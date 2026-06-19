"""Train + calibrate the per-sentence (claim-level) fusion for Option B, and measure the real goal.

Input : data/claims_<tag>.parquet  (one row per claim: features + factuality label, from build_claim_dataset)
Output: models/fusion_claim_<tag>.pkl  (+ thresholds) ; returns a metrics dict for the notebook to render.

Key choices:
  * GROUP split by `prompt` — all sentences of one answer go to the same side (no leakage).
  * Per-sentence metrics: AUROC/AUPR/F1 + confusion for SEP / HalluShift / TSV / FUSED.
  * Thresholds picked on TRAIN (F1-optimal), not test.
  * Localization metric ("find the wrong sentence"): for each test prompt with >=1 hallucinated claim,
    is the highest-FUSED claim actually hallucinated? (top-1 accuracy) + mean within-passage AUROC.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

from fusion import FusionModel
from metrics import detector_metrics, best_threshold, attach_curves

FEATS = ["sep_entropy", "hallushift", "tsv_margin"]
DETS = {"SEP": "sep_entropy", "HalluShift": "hallushift", "TSV": "tsv_margin"}


def _localization(df_test, score_col="fused"):
    """Top-1 'find the wrong sentence' accuracy + mean within-passage AUROC over multi-claim prompts."""
    top1_hits, top1_tot, aurocs = 0, 0, []
    for _, g in df_test.groupby("prompt"):
        y = g["label"].to_numpy()
        if y.sum() == 0 or len(g) < 2:
            continue
        s = g[score_col].to_numpy()
        top1_tot += 1
        if y[int(np.argmax(s))] == 1:
            top1_hits += 1
        if 0 < y.sum() < len(y):
            aurocs.append(roc_auc_score(y, s))
    return {"localization_top1": (top1_hits / top1_tot) if top1_tot else float("nan"),
            "n_multiclaim_prompts": top1_tot,
            "within_passage_auroc": float(np.mean(aurocs)) if aurocs else float("nan")}


def train(tag="v1", kind="logreg", C=0.5, test_size=0.25, seed=42, feats=None,
          t_high=None, t_med=None, save=True, verbose=True):
    """feats: which detector columns drive the fused score (default = all 3). Pass ["tsv_margin"] for the
    TSV-led demo fusion. t_high/t_med: override the calibrated thresholds (default = F1-optimal on train)."""
    feats = list(feats) if feats is not None else FEATS
    path = os.path.join(ROOT, "data", f"claims_{tag}.parquet")
    df = pd.read_parquet(path).reset_index(drop=True)
    y = df["label"].to_numpy().astype(int)
    if verbose:
        print(f"[train_claim] {path}: {len(df)} claims | halluc={y.mean()*100:.1f}% | "
              f"prompts={df['prompt'].nunique()} | feats={feats} | "
              f"sources={df['source'].value_counts().to_dict()}")

    if "split" in df.columns:
        # Option B: reuse the SAME held-out split the heads were trained on, so the TEST rows' detector
        # scores are out-of-sample (the heads never saw them) — no leakage into the fusion evaluation.
        tr = np.where(df["split"].to_numpy() == "train")[0]
        te = np.where(df["split"].to_numpy() == "test")[0]
        if verbose:
            print(f"[train_claim] using saved split: train={len(tr)} test={len(te)} (out-of-sample heads)")
    else:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
                      .split(df, y, groups=df["prompt"]))
    tr_df, te_df = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    y_tr, y_te = y[tr], y[te]

    fm = FusionModel(kind, feats, C=C).fit(tr_df.assign(hallucination=y_tr), label_col="hallucination")
    te_df = te_df.copy(); te_df["fused"] = fm.predict_proba(te_df)

    # thresholds: F1-optimal on TRAIN (no test peeking) unless explicitly overridden. The demo overrides
    # with a precision-favoring point because the F1-optimal cut (tuned on single-sentence TriviaQA) is far
    # too low for the multi-sentence regime — see the Phase-1 analysis.
    fused_tr = fm.predict_proba(tr_df)
    t_high = float(t_high) if t_high is not None else best_threshold(y_tr, fused_tr, metric="f1")
    t_med = float(t_med) if t_med is not None else round(0.6 * t_high, 3)

    # per-detector + fused metrics on TEST
    cand = {**{n: te_df[c].to_numpy() for n, c in DETS.items()}, "FUSED": te_df["fused"].to_numpy()}
    res = {}
    for name, s in cand.items():
        thr = t_high if name == "FUSED" else best_threshold(y_te, s)
        m = detector_metrics(y_te, s, threshold=thr)
        attach_curves(m, y_te, s)
        res[name] = m
    summary = pd.DataFrame([{"detector": n, "AUROC": m["AUROC"], "AUPR": m["AUPR"], "F1": m["F1"]}
                            for n, m in res.items()]).set_index("detector").round(3)
    loc = _localization(te_df)

    if verbose:
        print(summary.to_string())
        print(f"\nthresholds: T_MED={t_med}  T_HIGH={round(t_high,3)}")
        print(f"localization top-1 = {loc['localization_top1']:.3f} over {loc['n_multiclaim_prompts']} "
              f"multi-claim prompts | within-passage AUROC = {loc['within_passage_auroc']:.3f}")

    if save:
        os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
        mp = os.path.join(ROOT, "models", f"fusion_claim_{tag}.pkl")
        fm.save(mp)
        import json
        with open(os.path.join(ROOT, "models", f"fusion_claim_{tag}_thresholds.json"), "w") as f:
            json.dump({"t_med": float(t_med), "t_high": float(round(t_high, 3))}, f)
        if verbose:
            print(f"saved {os.path.relpath(mp, ROOT)} (+ thresholds json)")

    return {"summary": summary, "metrics": res, "localization": loc, "fusion": fm,
            "t_med": float(t_med), "t_high": float(round(t_high, 3)), "test_df": te_df}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--C", type=float, default=0.5)
    ap.add_argument("--kind", default="logreg", choices=["logreg", "gbm"])
    ap.add_argument("--feats", nargs="+", default=None,
                    help="detector cols driving the fused score (default all 3; demo uses: tsv_margin)")
    ap.add_argument("--t_high", type=float, default=None, help="override calibrated high threshold")
    ap.add_argument("--t_med", type=float, default=None, help="override calibrated medium threshold")
    args = ap.parse_args()
    train(tag=args.tag, kind=args.kind, C=args.C, feats=args.feats, t_high=args.t_high, t_med=args.t_med)
