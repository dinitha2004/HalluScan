import pandas as pd, numpy as np, sys
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

path = sys.argv[1] if len(sys.argv) > 1 else "data/triviaqa_fused.parquet"
df = pd.read_parquet(path)
y = df["hallucination"].to_numpy()
print(f"N={len(df)}  truthful={int((y==0).sum())}  hallucinated={int(y.sum())}  ({y.mean()*100:.1f}% halluc)\n")
print(f"{'detector':12s}  AUROC   AUPR")
for nme, col in [("SEP", "sep_entropy"), ("HalluShift", "hallushift"), ("TSV", "tsv_margin")]:
    print(f"{nme:12s}  {roc_auc_score(y, df[col]):.3f}   {average_precision_score(y, df[col]):.3f}")
SCORE = ["sep_entropy", "sep_accuracy", "hallushift", "tsv_margin"]
FEAT = SCORE + [f"hs_feat_{j:02d}" for j in range(71)]
for label, cols in [("score-level", SCORE), ("feature-level", FEAT)]:
    X = df[cols].to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    p = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    print(f"FUSED ({label:13s} 5-fold CV)  {roc_auc_score(y, p):.3f}   {average_precision_score(y, p):.3f}")
print("\ninter-detector correlation (Spearman):")
print(df[["sep_entropy", "hallushift", "tsv_margin"]].corr(method="spearman").round(2).to_string())
