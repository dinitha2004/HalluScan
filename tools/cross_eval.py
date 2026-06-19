"""Cross-dataset TRANSFER evaluation (score-only — NO re-training).

Everything (the 3 heads + the fusion) was trained on `train_ds` (triviaqa). Here we generate a
FRESH answer for every question of a DIFFERENT dataset (`target_ds`, e.g. truthfulqa), score it
with those already-trained heads, fuse, and measure AUROC/AUPR/F1. Because nothing is re-fit on
the target, every row is genuinely held out — this is the honest test of whether HallKing
generalises across datasets, and whether the fusion stays near the top when the *best single
detector* may change.

Pipeline (frozen LLM, two model loads, one at a time):
  1. [GPU, Instruct fp16] generate + cache SEP (135168-d) & HalluShift (71-d) features + BLEURT labels
  2. [CPU] score SEP probe + HalluShift MLP (trained on train_ds) on the cached features
  3. [GPU, Instruct fp16] score the trained TSV steering vector + centroids -> tsv_margin
  4. [CPU] apply the trained fusion meta-classifier; report per-detector + fused AUROC/AUPR/F1

Run in se_probes_env:
    python tools/cross_eval.py --target truthfulqa --train triviaqa
"""
import argparse, os, pickle, sys
os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

import numpy as np, pandas as pd, torch

import retrain
from engine import HallKingEngine
from run_dataset import INSTRUCT_MODEL
from tsv_adapter import TSVAdapter
from classifier import CombinedNN
from fusion import FusionModel
from metrics import detector_metrics, best_threshold

HS_COLS = [f"hs_feat_{j:02d}" for j in range(71)]


def evaluate_cross(target_ds="truthfulqa", train_ds="triviaqa", n=None, offset=0,
                   max_new_tokens=64, save=True, verbose=True):
    ART = os.path.join(ROOT, "artifacts")
    sep_pkl = os.path.join(ART, "sep", "probes_retrained.pkl")
    hs_model = os.path.join(ART, "hallushift", f"hal_det_retrained_{train_ds}_model.pth")
    hs_scaler = os.path.join(ART, "hallushift", f"hal_det_retrained_{train_ds}_scaler.pkl")
    tsv_ckpt = os.path.join(ART, "tsv", "best_checkpoint_retrained.pt")
    fusion_pkl = os.path.join(ROOT, "models", f"fusion_{train_ds}_oof.pkl")
    for p in (sep_pkl, hs_model, hs_scaler, tsv_ckpt, fusion_pkl):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing trained artifact: {p}\n"
                                    f"Run notebook 1 + notebook 2 on '{train_ds}' first.")

    # 1. generate + cache features + BLEURT labels on the TARGET dataset (Instruct fp16)
    print(f"==== 1. generate + cache features on {target_ds} (Instruct fp16) ====", flush=True)
    df, sep_feats = retrain.gen_and_cache(target_ds, n=n, offset=offset, max_new_tokens=max_new_tokens)
    y = df["hallucination"].to_numpy().astype(int)
    print(f"   n={len(df)}  balance: truthful={int((y==0).sum())} halluc={int(y.sum())} "
          f"({y.mean()*100:.1f}% halluc)")

    # 2. score SEP + HalluShift heads trained on train_ds
    print(f"==== 2. score SEP + HalluShift heads (trained on {train_ds}) ====", flush=True)
    with open(sep_pkl, "rb") as f:
        sep_probe = pickle.load(f)
    df["sep_entropy"] = sep_probe[0]["s_bmodel"].predict_proba(sep_feats.astype(np.float32))[:, 1]

    with open(hs_scaler, "rb") as f:
        scaler = pickle.load(f)
    m = CombinedNN(32); m.load_state_dict(torch.load(hs_model, map_location="cpu", weights_only=True)); m.eval()
    Xhs = scaler.transform(df[HS_COLS].to_numpy(np.float64))
    with torch.no_grad():
        df["hallushift"] = torch.sigmoid(m(torch.tensor(Xhs, dtype=torch.float32))).numpy().ravel()

    # 3. score the trained TSV head on the INSTRUCT model (fp16) — TSV is now Instruct-trained (nb 1b),
    #    so all three detectors share one model and TSV margins are on the matching variant.
    print(f"==== 3. score TSV head (trained on {train_ds}) on Instruct model ====", flush=True)
    beng = HallKingEngine(model_name=INSTRUCT_MODEL, fp16_nonquant=True).load()
    tsv = TSVAdapter(beng, ckpt_path=tsv_ckpt).load()
    from tqdm.auto import tqdm   # live progress bar so the cell visibly advances
    margins = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="TSV scoring", unit="q"):
        margins.append(tsv.score_qa(r["question"], r["answer"])["tsv_margin"])
    df["tsv_margin"] = margins
    beng.unload(); del beng, tsv

    # 4. apply the trained fusion + report
    print("==== 4. fusion + evaluation ====", flush=True)
    fm = FusionModel.load(fusion_pkl)
    df["fused"] = fm.predict_proba(df[["sep_entropy", "hallushift", "tsv_margin"]])

    cand = {"SEP": df["sep_entropy"].to_numpy(), "HalluShift": df["hallushift"].to_numpy(),
            "TSV": df["tsv_margin"].to_numpy(), "FUSED": df["fused"].to_numpy()}
    rows = []
    for name, s in cand.items():
        mm = detector_metrics(y, s, threshold=best_threshold(y, s))
        rows.append({"detector": name, "AUROC": mm["AUROC"], "AUPR": mm["AUPR"], "F1": mm["F1"]})
    results = pd.DataFrame(rows).set_index("detector").round(3)
    print(f"\n=== {train_ds} heads -> {target_ds} (transfer, all rows held out) ===")
    print(results.to_string())

    if save:
        out = os.path.join(ROOT, "data", f"{target_ds}_cross_eval.parquet")
        keep = ["question", "answer", "sep_entropy", "hallushift", "tsv_margin", "fused",
                "bleurt", "hallucination"]
        df[keep].to_parquet(out)
        print(f"\nsaved {out}")
    return results, df


def evaluate_many(datasets=("nq_open", "squad", "triviaqa"), train_ds="triviaqa", n=300,
                  offsets=None, max_new_tokens=64, save=True, verbose=True):
    """Run evaluate_cross() over several target datasets (one notebook, many datasets).

    `offsets` maps dataset -> generation offset; the default keeps TriviaQA HELD-OUT (offset 3000, past
    the training range 1000-2200). Returns {dataset: (results_df, scored_df)} so the notebook can build a
    combined table + confusion matrices / ROC-PR from the per-dataset frames.
    """
    offsets = {"triviaqa": 3000, **(offsets or {})}
    out = {}
    for ds in datasets:
        off = offsets.get(ds, 0)
        if verbose:
            print(f"\n############## {ds}  (n={n}, offset={off}) ##############", flush=True)
        results, df = evaluate_cross(ds, train_ds=train_ds, n=n, offset=off,
                                     max_new_tokens=max_new_tokens, save=save, verbose=verbose)
        out[ds] = (results, df)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="nq_open", choices=["nq_open", "squad", "triviaqa", "truthfulqa"])
    ap.add_argument("--train", default="triviaqa", choices=["triviaqa", "truthfulqa"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    args = ap.parse_args()
    evaluate_cross(args.target, args.train, n=args.n, offset=args.offset,
                   max_new_tokens=args.max_new_tokens)
