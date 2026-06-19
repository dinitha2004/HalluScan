"""ONE command to run the whole unified re-train (option A) and report results.

Pipeline (frozen LLM; two model loads, one at a time):
  1. [GPU, Instruct fp16] generate answers + cache SEP (135168-d) & HalluShift (71-d) features + BLEURT labels
  2. [CPU] re-fit SEP probe + HalluShift MLP on the cached features
  3. [GPU, base fp16]  re-train TSV steering vector + centroids (supervised) -> per-row margins
  4. [CPU] assemble the score table, train+CV the fusion meta-classifier, print AUROC/AUPR, save everything

Outputs:
  data/<ds>_fused.parquet            (per-example detector scores + fused + label)
  artifacts/sep/probes_retrained.pkl
  artifacts/hallushift/hal_det_retrained_<ds>_{model.pth,scaler.pkl}
  artifacts/tsv/best_checkpoint_retrained.pt
  models/fusion_<ds>_retrained.pkl

Run in se_probes_env:
    python tools/unified_retrain.py --dataset triviaqa --n 1200 --offset 1000 --epochs_tsv 40
"""
import argparse, os, pickle, sys
os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

import retrain
from classifier import CombinedNN
from fusion import FusionModel, SCORE_FEATURES, FEATURE_LEVEL
from run_dataset import INSTRUCT_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="triviaqa", choices=["triviaqa", "truthfulqa"])
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--offset", type=int, default=1000)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--epochs_tsv", type=int, default=40)
    args = ap.parse_args()
    ds = args.dataset
    DATA, ART, MODELS = (os.path.join(ROOT, d) for d in ("data", "artifacts", "models"))
    for d in (DATA, MODELS): os.makedirs(d, exist_ok=True)

    print("==== 1. generate + cache features (Instruct fp16) ====", flush=True)
    df, sep_feats = retrain.gen_and_cache(ds, n=args.n, offset=args.offset, max_new_tokens=args.max_new_tokens)
    np.save(os.path.join(DATA, f"{ds}_sep_feats.npy"), sep_feats)
    y = df["hallucination"].to_numpy()
    print(f"   n={len(df)}  balance: truthful={int((y==0).sum())} halluc={int(y.sum())}")

    print("==== 2. re-fit SEP probe + HalluShift MLP (CPU) ====", flush=True)
    sep_probe = retrain.retrain_sep(sep_feats, y)
    with open(os.path.join(ART, "sep", "probes_retrained.pkl"), "wb") as f:
        pickle.dump(sep_probe, f)
    df["sep_entropy"] = sep_probe[0]["s_bmodel"].predict_proba(sep_feats.astype(np.float32))[:, 1]
    df["sep_accuracy"] = sep_probe[0]["s_amodel"].predict_proba(sep_feats.astype(np.float32))[:, 1]

    hs_state, hs_scaler = retrain.retrain_hallushift(df, y)
    torch.save(hs_state, os.path.join(ART, "hallushift", f"hal_det_retrained_{ds}_model.pth"))
    with open(os.path.join(ART, "hallushift", f"hal_det_retrained_{ds}_scaler.pkl"), "wb") as f:
        pickle.dump(hs_scaler, f)
    cols = [f"hs_feat_{j:02d}" for j in range(71)]
    Xhs = hs_scaler.transform(df[cols].to_numpy(dtype=np.float64))
    m = CombinedNN(32); m.load_state_dict(hs_state); m.eval()
    with torch.no_grad():
        df["hallushift"] = torch.sigmoid(m(torch.tensor(Xhs, dtype=torch.float32))).numpy().ravel()

    print("==== 3. re-train TSV head (Instruct fp16 — same model as SEP/HalluShift) ====", flush=True)
    # Train TSV on the Instruct model so all three detectors share ONE model (single-model live demo).
    tsv_ckpt, tsv_margin = retrain.train_tsv(df, base_model=INSTRUCT_MODEL, epochs=args.epochs_tsv)
    torch.save(tsv_ckpt, os.path.join(ART, "tsv", "best_checkpoint_retrained.pt"))
    df["tsv_margin"] = tsv_margin

    print("==== 4. fusion + evaluation ====", flush=True)
    df.to_parquet(os.path.join(DATA, f"{ds}_fused.parquet"))
    dets = {"SEP": df["sep_entropy"], "HalluShift": df["hallushift"], "TSV": df["tsv_margin"]}
    print(f"\n{'detector':12s}  AUROC   AUPR  (re-trained heads, in-sample on full set)")
    for nme, s in dets.items():
        print(f"{nme:12s}  {roc_auc_score(y, s):.3f}   {average_precision_score(y, s):.3f}")
    # honest fused estimate via 5-fold CV
    for label, cols_ in [("score-level", SCORE_FEATURES), ("feature-level", FEATURE_LEVEL)]:
        X = df[cols_].to_numpy()
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
        print(f"FUSED ({label:13s}, 5-fold CV)  AUROC={roc_auc_score(y,p):.3f}  AUPR={average_precision_score(y,p):.3f}")
    # save a fusion model trained on all data (score-level)
    FusionModel("logreg", SCORE_FEATURES).fit(df).save(os.path.join(MODELS, f"fusion_{ds}_retrained.pkl"))
    print(f"\nSaved: data/{ds}_fused.parquet, re-trained artifacts, models/fusion_{ds}_retrained.pkl")


if __name__ == "__main__":
    main()
