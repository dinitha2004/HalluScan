"""Gate Step A — CPU-only audit of the sentence-head training labels + per-head separability.

NO GPU, NO model load. Reads the cached artifacts to answer "are the labels the culprit?":
  1. per-head held-out AUROC/AUPR on the scored claims_<tag>.parquet TEST split,
  2. the live-demo per-detector distributions (HalluShift flatness, SEP false-positive firing) from
     data/demo_scores.jsonl,
  3. label-inspection samples (both classes) for human review + a CSV worksheet for the gold-set check.

Run (any python with pandas + scikit-learn):
  python tools/diagnose_heads.py --tag s1
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

try:  # Windows consoles default to cp1252 and choke on non-ASCII; force UTF-8 output.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# (column, head-name, higher_means_hallucinated) — sep_accuracy is P(truthful) so it inverts.
HEAD_COLS = [("sep_entropy", "SEP (entropy)", True),
             ("sep_accuracy", "SEP (1-accuracy)", False),
             ("hallushift", "HalluShift", True),
             ("tsv_margin", "TSV (margin)", True)]


def _auroc(y, score, higher_halluc=True):
    s = np.asarray(score, dtype=float)
    if not higher_halluc:
        s = -s
    return roc_auc_score(y, s), average_precision_score(y, s)


def per_head_auroc(tag="s1"):
    """Per-head held-out AUROC/AUPR on the scored claims_<tag>.parquet test split (labels may be noisy)."""
    path = os.path.join(DATA, f"claims_{tag}.parquet")
    if not os.path.exists(path):
        print(f"[per-head] {os.path.relpath(path, ROOT)} not found — skipping AUROC")
        return
    df = pd.read_parquet(path)
    te = df[df["split"] == "test"] if "split" in df.columns else df
    y = te["label"].to_numpy().astype(int)
    print(f"\n=== per-head held-out AUROC (claims_{tag}.parquet, "
          f"n_test={len(te)}, halluc={y.mean()*100:.1f}%) ===")
    if len(np.unique(y)) < 2:
        print("  only one class in test split — cannot compute AUROC"); return
    print(f"  {'head':18} {'AUROC':>7} {'AUPR':>7}")
    for col, name, hh in HEAD_COLS:
        if col in te.columns:
            au, ap = _auroc(y, te[col].to_numpy(), hh)
            print(f"  {name:18} {au:7.3f} {ap:7.3f}")
    print("  (AUROC ~0.5 = no separation. Read against the label-noise estimate below — a head can look")
    print("   dead here purely because the labels it's scored against are wrong.)")


def demo_distributions(path=None):
    """Per-detector distribution over EVERY scored sentence in the live-demo log (no ground truth —
    shows flatness / firing-rate symptoms)."""
    path = path or os.path.join(DATA, "demo_scores.jsonl")
    if not os.path.exists(path):
        print(f"\n[demo] {os.path.relpath(path, ROOT)} not found — skipping"); return
    cols = {"sep_entropy": [], "sep_accuracy": [], "hallushift": [], "tsv_margin": []}
    n_records = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_records += 1
            rec = json.loads(line)
            for s in rec.get("sentences", []):
                if not s.get("is_claim"):
                    continue
                for k in cols:
                    v = s.get(k)
                    if v is not None:
                        cols[k].append(float(v))
    print(f"\n=== live-demo per-detector distribution ({n_records} queries, "
          f"{len(cols['hallushift'])} claim sentences) ===")
    print(f"  {'detector':14} {'mean':>7} {'std':>7} {'min':>7} {'max':>7}  note")
    notes = {"sep_entropy": "frac>0.5 (fires)=", "hallushift": "range width=",
             "sep_accuracy": "", "tsv_margin": ""}
    for k, v in cols.items():
        if not v:
            continue
        a = np.array(v)
        note = ""
        if k == "sep_entropy":
            note = f"{notes[k]}{(a > 0.5).mean():.2f}"
        elif k == "hallushift":
            note = f"{notes[k]}{a.max()-a.min():.2f} (small spread -> flat/dead)"
        print(f"  {k:14} {a.mean():7.3f} {a.std():7.3f} {a.min():7.3f} {a.max():7.3f}  {note}")


def label_inspection(tag="s1", n=20, write_worksheet=True):
    """Print a sample of each label class for human eyeballing + write a hand-label worksheet CSV
    (question, answer, current_label, blank true_label) for the gold-set agreement check."""
    raw = os.path.join(DATA, f"claims_sent_{tag}.parquet")
    if not os.path.exists(raw):
        print(f"\n[labels] {os.path.relpath(raw, ROOT)} not found — skipping"); return
    df = pd.read_parquet(raw)
    y = df["hallucination"].to_numpy().astype(int)
    print(f"\n=== label inspection (claims_sent_{tag}.parquet, n={len(df)}, "
          f"halluc={y.mean()*100:.1f}%) ===")
    for lab, title in [(1, "labelled HALLUCINATED (eyeball for TRUE sentences mislabelled)"),
                       (0, "labelled TRUTHFUL")]:
        sub = df[df["hallucination"] == lab][["question", "answer"]].head(n)
        print(f"\n  --- {title} ---")
        for _, r in sub.iterrows():
            print(f"   Q: {str(r['question'])[:72]}")
            print(f"     A: {str(r['answer'])[:104]}")
    if write_worksheet:
        rng = np.random.RandomState(0)
        idx = rng.permutation(len(df))[:min(60, len(df))]
        ws = df.iloc[idx][["question", "answer", "hallucination"]].rename(
            columns={"hallucination": "current_label"})
        ws["true_label"] = ""   # fill 0=truthful / 1=halluc by hand, then compute labeler accuracy
        out = os.path.join(DATA, f"label_audit_{tag}.csv")
        ws.to_csv(out, index=False, encoding="utf-8")
        print(f"\n  wrote hand-label worksheet -> {os.path.relpath(out, ROOT)} "
              f"({len(ws)} rows; fill 'true_label' to measure the labeler's accuracy)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--n", type=int, default=20, help="sample rows per class to print")
    args = ap.parse_args()
    print(f"HallKing — dataset & head audit (tag={args.tag})")
    per_head_auroc(args.tag)
    demo_distributions()
    label_inspection(args.tag, n=args.n)


if __name__ == "__main__":
    main()
