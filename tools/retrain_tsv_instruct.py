"""Re-train the TSV head on the INSTRUCT model so all three detectors live on ONE model.

Originally TSV was trained on the BASE Llama-3.1-8B while SEP+HalluShift run on -Instruct, so the
live demo had to load TWO 8B models (OOMs a 12 GB GPU, and load/unload-per-question is unusable for
a live demo). TSV's steering vector + centroids are a tiny supervised head, so we just re-fit them
on the Instruct model — then a single Instruct load serves all three detectors and the demo scores
any question in seconds.

Cheap (~0.1 GPU-hr): reuses the answers + labels already in data/<ds>_fused.parquet (no
re-generation). Overwrites:
  * artifacts/tsv/best_checkpoint_retrained.pt   (now Instruct-trained)
  * the tsv_margin column of data/<ds>_fused.parquet
Then RE-RUN notebook 2 (honest_eval) to refresh models/fusion_<ds>_oof.pkl on the new margins.

Run in se_probes_env:
    python tools/retrain_tsv_instruct.py --dataset triviaqa --epochs_tsv 40
"""
import argparse, os, sys
os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

import retrain
from run_dataset import INSTRUCT_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="triviaqa", choices=["triviaqa", "truthfulqa"])
    ap.add_argument("--epochs_tsv", type=int, default=40)
    args = ap.parse_args()
    ds = args.dataset
    pq = os.path.join(ROOT, "data", f"{ds}_fused.parquet")
    ckpt_path = os.path.join(ROOT, "artifacts", "tsv", "best_checkpoint_retrained.pt")
    if not os.path.exists(pq):
        raise FileNotFoundError(f"{pq} not found — run notebook 1 (unified_retrain) on '{ds}' first.")

    df = pd.read_parquet(pq).reset_index(drop=True)
    y = df["hallucination"].to_numpy().astype(int)
    print(f"==== re-train TSV on INSTRUCT ({INSTRUCT_MODEL}) | {ds} n={len(df)} ====", flush=True)

    # same split/seed as before (test rows stay OOF for notebook 2's honest eval)
    tsv_ckpt, tsv_margin = retrain.train_tsv(df, base_model=INSTRUCT_MODEL, epochs=args.epochs_tsv)
    torch.save(tsv_ckpt, ckpt_path)

    df["tsv_margin"] = tsv_margin
    df.to_parquet(pq)

    auroc = roc_auc_score(y, tsv_margin)
    print(f"\n  Instruct-TSV in-sample AUROC = {auroc:.3f} (best held-out during train: "
          f"{tsv_ckpt.get('best_test_auroc', float('nan')):.3f})")
    print(f"  saved {os.path.relpath(ckpt_path, ROOT)} + updated tsv_margin in {os.path.relpath(pq, ROOT)}")
    print("\n  NEXT: re-run notebook 2 (honest_eval) to refresh models/fusion_%s_oof.pkl on the new "
          "margins, then the demo (notebook 4) runs on a SINGLE model." % ds)


if __name__ == "__main__":
    main()
