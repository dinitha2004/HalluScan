"""HONEST out-of-fold evaluation of the re-trained heads.

The unified-retrain parquet columns are IN-SAMPLE (heads were fit on all rows), so the
135168-d SEP probe memorises its training rows -> AUROC 1.0 is leakage, not skill.

`evaluate(ds)` reproduces TSV's exact held-out split (test_size=0.25, random_state=42,
stratified) and evaluates on those rows ONLY, where every detector score is out-of-fold:
  * SEP        : re-fit LogReg on the TRAIN rows' cached features -> score TEST rows
  * HalluShift : re-fit MLP   on the TRAIN rows' cached features -> score TEST rows
  * TSV        : margins straight from the parquet (TSV trained on TRAIN rows only -> TEST is OOF)
The fusion is trained on the TRAIN rows (SEP/HS via inner 5-fold OOF so it isn't fed a
perfectly-memorised SEP feature) and evaluated on the held-out TEST rows.

CPU only. Returns (results_df, oof_df) and saves data/<ds>_eval_oof.parquet + the fusion model.
"""
import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import retrain
from classifier import CombinedNN
from metrics import detector_metrics, best_threshold

HS_COLS = [f"hs_feat_{j:02d}" for j in range(71)]


def evaluate(ds="triviaqa", save=True, verbose=True):
    df = pd.read_parquet(os.path.join(ROOT, "data", f"{ds}_fused.parquet")).reset_index(drop=True)
    sep_feats = np.load(os.path.join(ROOT, "data", f"{ds}_sep_feats.npy")).astype(np.float32)
    y = df["hallucination"].to_numpy().astype(int)

    # reproduce train_tsv's split exactly (stratify=1-hallucination, test_size=0.25, rs=42)
    tr, te = train_test_split(np.arange(len(df)), test_size=0.25, stratify=(1 - y), random_state=42)
    y_te = y[te]
    if verbose:
        print(f"[honest_eval] {ds}: train={len(tr)} test={len(te)}  test halluc {y_te.mean()*100:.1f}%")

    def fit_sep(idx):
        return LogisticRegression(max_iter=2000, C=0.1).fit(sep_feats[idx], y[idx])

    def fit_hs(idx):
        state, scaler = retrain.retrain_hallushift(df.iloc[idx].reset_index(drop=True), y[idx], seed=0)
        m = CombinedNN(32); m.load_state_dict(state); m.eval()
        return m, scaler

    def hs_predict(m, scaler, idx):
        X = scaler.transform(df.iloc[idx][HS_COLS].to_numpy(np.float64))
        with torch.no_grad():
            return torch.sigmoid(m(torch.tensor(X, dtype=torch.float32))).numpy().ravel()

    from tqdm.auto import tqdm  # live progress: the 6 HalluShift re-fits below dominate the ~5 min
    # --- OOF detector scores on TEST ---
    if verbose:
        print("[1/2] fitting SEP + HalluShift on train -> scoring test ...", flush=True)
    sep_te = fit_sep(tr).predict_proba(sep_feats[te])[:, 1]
    hm, hsc = fit_hs(tr); hs_te = hs_predict(hm, hsc, te)
    tsv_te = df["tsv_margin"].to_numpy()[te]

    # --- inner 5-fold OOF on TRAIN so fusion isn't fed a memorised SEP feature ---
    if verbose:
        print("[2/2] inner 5-fold OOF on train (refit SEP + HalluShift per fold) ...", flush=True)
    sep_tr = np.zeros(len(tr)); hs_tr = np.zeros(len(tr))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y[tr]))
    for a, b in tqdm(folds, desc="inner OOF folds", unit="fold", disable=not verbose):
        sep_tr[b] = fit_sep(tr[a]).predict_proba(sep_feats[tr[b]])[:, 1]
        mm, sc = fit_hs(tr[a]); hs_tr[b] = hs_predict(mm, sc, tr[b])
    tsv_tr = df["tsv_margin"].to_numpy()[tr]

    from fusion import FusionModel
    # Fixed, regularized logistic fusion. We do NOT tune the config on test, and we can't tune it on
    # train either: TSV's TRAIN margins are in-sample (TSV trained on those rows), so any train-side CV is
    # dominated by that memorized feature (CV AUROC ~0.98) and won't generalize. C=0.3 is a sensible
    # small-sample default for a 3-feature blend. On this in-distribution split TSV already dominates, so
    # FUSED ties it; fusion's measured advantage is cross-dataset transfer (notebook 5), where the best
    # single detector changes and no one detector dominates.
    FEAT3 = ["sep_entropy", "hallushift", "tsv_margin"]
    tr_df = pd.DataFrame({"sep_entropy": sep_tr, "hallushift": hs_tr, "tsv_margin": tsv_tr, "hallucination": y[tr]})
    te_df = pd.DataFrame({"sep_entropy": sep_te, "hallushift": hs_te, "tsv_margin": tsv_te})
    best_fm = FusionModel("logreg", FEAT3, C=0.3).fit(tr_df)
    fused_te = best_fm.predict_proba(te_df)
    best_fuse = "FUSED (logreg)"
    fusions = {best_fuse: best_fm}

    # --- results table ---
    cand = {"SEP": sep_te, "HalluShift": hs_te, "TSV": tsv_te, best_fuse: fused_te}
    rows = []
    for name, s in cand.items():
        mm = detector_metrics(y_te, s, threshold=best_threshold(y_te, s))
        rows.append({"detector": name, "AUROC": mm["AUROC"], "AUPR": mm["AUPR"], "F1": mm["F1"]})
    results = pd.DataFrame(rows).set_index("detector").round(3)
    if verbose:
        print(results.to_string())

    oof = pd.DataFrame({"question": df.iloc[te]["question"].values, "answer": df.iloc[te]["answer"].values,
                        "sep_entropy": sep_te, "hallushift": hs_te, "tsv_margin": tsv_te,
                        "fused": fused_te, "hallucination": y_te})
    if save:
        os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
        oof.to_parquet(os.path.join(ROOT, "data", f"{ds}_eval_oof.parquet"))
        fusions[best_fuse].save(os.path.join(ROOT, "models", f"fusion_{ds}_oof.pkl"))
        if verbose:
            print(f"saved data/{ds}_eval_oof.parquet + models/fusion_{ds}_oof.pkl (best: {best_fuse})")
    return results, oof


if __name__ == "__main__":
    evaluate(sys.argv[1] if len(sys.argv) > 1 else "triviaqa")
