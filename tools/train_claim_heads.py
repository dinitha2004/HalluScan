"""Option B — retrain the 3 detector HEADS on SENTENCE-level units with reference-match labels, and
report each head's HELD-OUT per-sentence AUROC. This is the decisive "do the signals actually separate
at sentence level?" checkpoint that comes BEFORE any fusion (the user's method: make each technique work
first, then fuse).

Why this is different from Option A (tools/unified_retrain.py): there the heads were trained on ~2-word
TriviaQA answers (BLEURT labels) — they track answer *form*, not sentence factuality, so the demo over-flags
full sentences. Here we train on the SAME regime the live demo uses (one forced factual sentence via the
Instruct chat template) with cheap, accurate reference-match labels — train == inference.

Flow (callable from notebook 8):
  build(tag, datasets, ...)         [GPU]  generate ONE factual sentence per question; cache RAW
                                           SEP(135168-d) + HalluShift(71-d) features + (question, answer)
                                           for TSV + reference-match labels. Refusals ("I don't know") are
                                           dropped (not claims, not hallucinations). -> data/claims_sent_<tag>.*
  train_heads(tag, ...)   [CPU + 1 GPU for TSV]  ONE shared stratified 75/25 split; refit SEP probe +
                                           HalluShift MLP + TSV vector on TRAIN; score the held-out TEST;
                                           print the per-head AUROC table; save Option-B head artifacts and a
                                           SCORED data/claims_<tag>.parquet (with a `split` col, TEST rows
                                           scored out-of-sample) ready for tools/train_claim_fusion.py.

Artifacts use a `_sentence_<tag>` suffix so they DON'T clobber the working Option-A short-QA artifacts:
  artifacts/sep/probes_sentence_<tag>.pkl
  artifacts/hallushift/hal_det_sentence_<tag>_{model.pth,scaler.pkl}
  artifacts/tsv/best_checkpoint_sentence_<tag>.pt

Run in se_probes_env:
  python tools/train_claim_heads.py --build --tag s1 --datasets triviaqa --n 1500 --offset 0
  python tools/train_claim_heads.py --tag s1 --epochs_tsv 40           # train heads from the cached build
"""
import argparse, os, pickle, sys
os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import retrain
from classifier import CombinedNN
from run_dataset import INSTRUCT_MODEL

DATA = os.path.join(ROOT, "data")
ART = os.path.join(ROOT, "artifacts")
HS_COLS = [f"hs_feat_{j:02d}" for j in range(71)]


def _raw_paths(tag):
    return (os.path.join(DATA, f"claims_sent_{tag}.parquet"),
            os.path.join(DATA, f"claims_sent_{tag}_sepfeats.npy"))


# ---------------------------------------------------------------- GPU: generate sentences + cache features
def build(tag="s1", datasets=("triviaqa",), n=1500, offset=0, max_new_tokens=64,
          instruct_model=INSTRUCT_MODEL, label_method="llm_judge", save=True, verbose=True):
    """Generate one factual sentence per question (sentence regime), cache RAW per-head features + labels,
    drop refusals. Saves data/claims_sent_<tag>.parquet + _sepfeats.npy.

    label_method: "llm_judge" (default, comparative QA judge — robust) | "reference" (substring, brittle) |
    "bleurt". The substring path mislabels ~40% of true sentences as hallucinated, so it is no longer default."""
    if isinstance(datasets, str):
        datasets = [datasets]
    dfs, feats = [], []
    for ds in datasets:
        df_i, sf_i = retrain.gen_and_cache(ds, n=n, offset=offset, max_new_tokens=max_new_tokens,
                                           instruct_model=instruct_model, regime="sentence",
                                           label_method=label_method, drop_refusals=True, verbose=verbose)
        df_i["source"] = f"qa:{ds}"
        dfs.append(df_i); feats.append(sf_i)
    df = pd.concat(dfs, ignore_index=True)
    sep_feats = np.vstack(feats)
    if save:
        os.makedirs(DATA, exist_ok=True)
        rawp, sepp = _raw_paths(tag)
        df.to_parquet(rawp)
        np.save(sepp, sep_feats)
        print(f"[build] saved {os.path.relpath(rawp, ROOT)} {df.shape} + "
              f"{os.path.relpath(sepp, ROOT)} {sep_feats.shape}", flush=True)
    return df, sep_feats


# ---------------------------------------------------------------- CPU + 1 GPU: retrain heads + AUROC
def train_heads(tag="s1", epochs_tsv=40, test_size=0.25, seed=42, save=True, verbose=True):
    """ONE shared split -> refit each head on TRAIN -> score held-out TEST -> per-head AUROC table.
    Saves Option-B head artifacts + a scored data/claims_<tag>.parquet (TEST rows out-of-sample)."""
    rawp, sepp = _raw_paths(tag)
    if not (os.path.exists(rawp) and os.path.exists(sepp)):
        raise FileNotFoundError(f"{rawp} / {sepp} not found — run build(tag='{tag}', ...) first (GPU).")
    df = pd.read_parquet(rawp).reset_index(drop=True)
    sep_feats = np.load(sepp)
    y = df["hallucination"].to_numpy().astype(int)
    if len(df) != len(sep_feats):
        raise RuntimeError(f"row mismatch: df={len(df)} sep_feats={len(sep_feats)}")
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"only one class present (halluc={y.mean()*100:.1f}%) — cannot train/eval. "
                           "Add more questions or check reference matching.")

    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=test_size, stratify=y,
                                      random_state=seed)
    print(f"[heads] n={len(df)} | train={len(tr_idx)} test={len(te_idx)} | "
          f"halluc={y.mean()*100:.1f}% | sources={df['source'].value_counts().to_dict()}", flush=True)

    auroc, aupr = {}, {}

    # ---- SEP probe (CPU): fit on TRAIN, score ALL rows, AUROC on TEST ----
    print("---- SEP probe (CPU) ----", flush=True)
    sep_probe = retrain.retrain_sep(sep_feats[tr_idx].astype(np.float32), y[tr_idx], name=f"sentence_{tag}")
    Xf = sep_feats.astype(np.float32)
    df["sep_entropy"] = sep_probe[0]["s_bmodel"].predict_proba(Xf)[:, 1]   # P(hallucinated)
    df["sep_accuracy"] = sep_probe[0]["s_amodel"].predict_proba(Xf)[:, 1]  # P(truthful)
    auroc["SEP"] = roc_auc_score(y[te_idx], df["sep_entropy"].to_numpy()[te_idx])
    aupr["SEP"] = average_precision_score(y[te_idx], df["sep_entropy"].to_numpy()[te_idx])

    # ---- HalluShift MLP (CPU): fit on TRAIN rows (internal val for early stop), score ALL, AUROC on TEST ----
    print("---- HalluShift MLP (CPU) ----", flush=True)
    hs_state, hs_scaler = retrain.retrain_hallushift(df.iloc[tr_idx].reset_index(drop=True), y[tr_idx],
                                                     seed=seed)
    Xhs = hs_scaler.transform(df[HS_COLS].to_numpy(dtype=np.float64))
    m = CombinedNN(32); m.load_state_dict(hs_state); m.eval()
    with torch.no_grad():
        df["hallushift"] = torch.sigmoid(m(torch.tensor(Xhs, dtype=torch.float32))).numpy().ravel()
    auroc["HalluShift"] = roc_auc_score(y[te_idx], df["hallushift"].to_numpy()[te_idx])
    aupr["HalluShift"] = average_precision_score(y[te_idx], df["hallushift"].to_numpy()[te_idx])

    # ---- TSV head (GPU): SHARED split so AUROC is on the SAME held-out questions ----
    print("---- TSV head (GPU, Instruct fp16) ----", flush=True)
    tsv_ckpt, tsv_margin = retrain.train_tsv(df, base_model=INSTRUCT_MODEL, epochs=epochs_tsv,
                                             tr_idx=tr_idx, te_idx=te_idx, verbose=verbose)
    df["tsv_margin"] = tsv_margin
    auroc["TSV"] = roc_auc_score(y[te_idx], tsv_margin[te_idx])
    aupr["TSV"] = average_precision_score(y[te_idx], tsv_margin[te_idx])

    # ---- the GO/NO-GO summary ----
    summary = pd.DataFrame([{"head": k, "heldout_AUROC": round(auroc[k], 3),
                             "heldout_AUPR": round(aupr[k], 3)} for k in auroc]).set_index("head")
    print("\n==== per-head HELD-OUT AUROC (sentence regime, reference-match labels) ====", flush=True)
    print(summary.to_string(), flush=True)
    print("\ndecision: a head that separates (AUROC >~0.65) is worth fusing; if all stay ~0.5 even with "
          "sentence-length units + accurate labels, the signal does not carry sentence-level factuality "
          "(an honest finding — scope accordingly).", flush=True)

    if save:
        for sub in ("sep", "hallushift", "tsv"):
            os.makedirs(os.path.join(ART, sub), exist_ok=True)
        with open(os.path.join(ART, "sep", f"probes_sentence_{tag}.pkl"), "wb") as f:
            pickle.dump(sep_probe, f)
        torch.save(hs_state, os.path.join(ART, "hallushift", f"hal_det_sentence_{tag}_model.pth"))
        with open(os.path.join(ART, "hallushift", f"hal_det_sentence_{tag}_scaler.pkl"), "wb") as f:
            pickle.dump(hs_scaler, f)
        torch.save(tsv_ckpt, os.path.join(ART, "tsv", f"best_checkpoint_sentence_{tag}.pt"))

        # scored table for the fusion step: TEST rows are out-of-sample (heads never saw them).
        split = np.array(["train"] * len(df), dtype=object)
        split[te_idx] = "test"
        scored = pd.DataFrame({
            "prompt": df["question"], "source": df["source"], "answer": df["answer"],
            "sep_entropy": df["sep_entropy"], "sep_accuracy": df["sep_accuracy"],
            "hallushift": df["hallushift"], "tsv_margin": df["tsv_margin"],
            "label": y, "split": split})
        scored.to_parquet(os.path.join(DATA, f"claims_{tag}.parquet"))
        print(f"\nsaved Option-B head artifacts (_sentence_{tag}) + data/claims_{tag}.parquet "
              f"(scored, with split col)", flush=True)

    return {"summary": summary, "auroc": auroc, "aupr": aupr, "tr_idx": tr_idx, "te_idx": te_idx,
            "sep_probe": sep_probe, "hs_state": hs_state, "hs_scaler": hs_scaler, "tsv_ckpt": tsv_ckpt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="GPU: generate sentences + cache features (run once)")
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--datasets", nargs="+", default=["triviaqa"], help="triviaqa / nq_open / squad")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--epochs_tsv", type=int, default=40)
    args = ap.parse_args()
    if args.build:
        build(tag=args.tag, datasets=args.datasets, n=args.n, offset=args.offset,
              max_new_tokens=args.max_new_tokens)
    else:
        train_heads(tag=args.tag, epochs_tsv=args.epochs_tsv)


if __name__ == "__main__":
    main()
