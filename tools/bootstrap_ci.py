"""Bootstrap CIs on the OOF test set to see whether detector differences are real or noise."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ds = sys.argv[1] if len(sys.argv) > 1 else "triviaqa"
df = pd.read_parquet(os.path.join(ROOT, "data", f"{ds}_eval_oof.parquet"))
y = df["hallucination"].to_numpy()
cols = {"SEP": "sep_entropy", "HalluShift": "hallushift", "TSV": "tsv_margin", "FUSED": "fused"}
S = {k: df[v].to_numpy() for k, v in cols.items()}
rng = np.random.RandomState(0)
n = len(y); B = 2000
boot = {k: [] for k in S}
diff = []  # FUSED - HalluShift
for _ in range(B):
    idx = rng.randint(0, n, n)
    if len(np.unique(y[idx])) < 2:
        continue
    for k in S:
        boot[k].append(roc_auc_score(y[idx], S[k][idx]))
    diff.append(roc_auc_score(y[idx], S["FUSED"][idx]) - roc_auc_score(y[idx], S["HalluShift"][idx]))
print(f"n_test={n}  bootstrap B={len(diff)}\n")
print(f"{'detector':12s} AUROC   95% CI")
for k in S:
    a = roc_auc_score(y, S[k]); lo, hi = np.percentile(boot[k], [2.5, 97.5])
    print(f"{k:12s} {a:.3f}   [{lo:.3f}, {hi:.3f}]")
diff = np.array(diff)
print(f"\nFUSED - HalluShift: mean {diff.mean():+.3f}  95% CI [{np.percentile(diff,2.5):+.3f}, {np.percentile(diff,97.5):+.3f}]")
print(f"P(FUSED > HalluShift) = {(diff>0).mean():.2f}   (0.5 = pure coin-flip; differences are within noise)")
