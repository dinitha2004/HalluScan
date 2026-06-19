"""Mini end-to-end evaluation (decisive validation before building the full notebooks).

Two-pass design (one 8B model in VRAM at a time):
  Pass 1 (Instruct): generate answer + SEP + HalluShift scores for N TruthfulQA questions.
  Pass 2 (base):     TSV margin for each (question, answer) on TSV's native base model.
Then BLEURT ground-truth (bleurt_env) -> labels, and per-detector AUROC/AUPR.

Goal: confirm each detector (especially TSV via cosine margin) carries real signal, and
that fusion >= best individual. Run in se_probes_env:  python tools/mini_eval.py
"""
import os, sys
os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, average_precision_score

N = int(os.environ.get("MINI_N", "80"))
OFFSET = int(os.environ.get("MINI_OFFSET", "0"))  # offset>=100 -> held out from HalluShift's training
MAXTOK = 64

# ----------------------------------------------------------------- data
ds = load_dataset("truthful_qa", "generation")["validation"]
ds = ds.select(range(OFFSET, min(OFFSET + N, len(ds))))
questions = [r["question"] for r in ds]
refs = [list(r["correct_answers"]) + [r["best_answer"]] for r in ds]
print(f"[mini_eval] {len(questions)} TruthfulQA questions")

# SINGLE_MODEL env -> run ALL three detectors on one model (decisive base-vs-instruct test).
from engine import HallKingEngine
from sep_adapter import SEPAdapter
from hallushift_adapter import HalluShiftAdapter
from tsv_adapter import TSVAdapter

SINGLE = os.environ.get("SINGLE_MODEL", "").strip()
if SINGLE:
    print(f"[mini_eval] SINGLE-MODEL mode: all 3 detectors on {SINGLE}")
    eng = HallKingEngine(model_name=SINGLE).load()
    sep = SEPAdapter(eng, probe_name="llama3-triviaqa").load()
    hs = HalluShiftAdapter(eng, dataset="truthfulqa").load()
    tsv = TSVAdapter(eng).load()
    rows = []
    for i, q in enumerate(questions):
        gen = eng.generate(q, max_new_tokens=MAXTOK)
        r = {"question": q, "answer": gen["answer_clean"]}
        r.update(sep.score(gen)); r.update(hs.score(gen)); r.update(tsv.score(gen))
        rows.append(r)
        if i % 20 == 0:
            print(f"  [{i}/{len(questions)}] A={r['answer'][:45]!r} tsvm={r['tsv_margin']:.3f}")
    eng.unload(); del eng, sep, hs, tsv
else:
    eng = HallKingEngine(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct").load()
    sep = SEPAdapter(eng, probe_name="llama3-triviaqa").load()
    hs = HalluShiftAdapter(eng, dataset="truthfulqa").load()
    rows = []
    for i, q in enumerate(questions):
        gen = eng.generate(q, max_new_tokens=MAXTOK)
        r = {"question": q, "answer": gen["answer_clean"]}
        r.update(sep.score(gen)); r.update(hs.score(gen))
        rows.append(r)
        if i % 20 == 0:
            print(f"  [pass1] {i}/{len(questions)}  A={r['answer'][:50]!r}")
    eng.unload(); del eng, sep, hs
    beng = HallKingEngine(model_name="meta-llama/Meta-Llama-3.1-8B", fp16_nonquant=True).load()
    tsv = TSVAdapter(beng).load()
    for i, r in enumerate(rows):
        r.update(tsv.score_qa(r["question"], r["answer"]))
        if i % 20 == 0:
            print(f"  [pass2/tsv] {i}/{len(rows)}  tsv_margin={r['tsv_margin']:.4f}")
    beng.unload(); del beng, tsv

df = pd.DataFrame(rows)

# ----------------------------------------------------------------- BLEURT GT
from gt_bleurt import bleurt_labels
labels, scores = bleurt_labels(df["answer"].tolist(), refs, threshold=0.5)
df["bleurt"] = scores
df["hallucination"] = labels
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
df.to_parquet(os.path.join(ROOT, "data", "mini_eval.parquet"))
print(f"\n[mini_eval] label balance: truthful={int((labels==0).sum())} hallucinated={int(labels.sum())}")

# ----------------------------------------------------------------- per-detector AUROC/AUPR
y = df["hallucination"].to_numpy()
detector_scores = {
    "SEP (entropy)":   df["sep_entropy"].to_numpy(),
    "SEP (1-accuracy)": 1.0 - df["sep_accuracy"].to_numpy(),
    "HalluShift":      df["hallushift"].to_numpy(),
    "TSV (margin)":    df["tsv_margin"].to_numpy(),
    "TSV (1-P_truth)": df["tsv"].to_numpy(),
}
print("\n================ per-detector (higher score = hallucinated) ================")
print(f"{'detector':20s}  AUROC   AUPR")
for name, s in detector_scores.items():
    if len(set(y)) < 2:
        print("  single-class labels -> cannot compute AUROC"); break
    print(f"{name:20s}  {roc_auc_score(y, s):.3f}   {average_precision_score(y, s):.3f}")

# quick in-sample score-level fusion sanity (LogReg, 5-fold CV AUROC)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    X = df[["sep_entropy", "sep_accuracy", "hallushift", "tsv_margin"]].to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    print(f"\n{'FUSED (score, 5-fold CV)':20s}  {roc_auc_score(y, proba):.3f}   {average_precision_score(y, proba):.3f}")
except Exception as e:
    print("fusion sanity skipped:", e)
