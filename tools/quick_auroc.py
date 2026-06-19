"""Print per-detector + quick-fusion AUROC/AUPR for a built dataset parquet.
Usage: python tools/quick_auroc.py data/triviaqa_mini.parquet
"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

df = pd.read_parquet(sys.argv[1])
y = df["hallucination"].to_numpy()
print(f"n={len(df)}  balance: truthful={int((y==0).sum())} hallucinated={int(y.sum())}")
dets = {"SEP (entropy)": df["sep_entropy"], "HalluShift": df["hallushift"], "TSV (margin)": df["tsv_margin"]}
print(f"\n{'detector':16s}  AUROC   AUPR")
for n, s in dets.items():
    print(f"{n:16s}  {roc_auc_score(y, s):.3f}   {average_precision_score(y, s):.3f}")
X = df[["sep_entropy", "sep_accuracy", "hallushift", "tsv_margin"]].to_numpy()
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
print(f"{'FUSED (CV)':16s}  {roc_auc_score(y, p):.3f}   {average_precision_score(y, p):.3f}")
