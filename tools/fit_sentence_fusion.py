"""Fit the lightweight 3-feature SENTENCE fusion for the Type-2 cross-dataset benchmark (notebook 5b).

Trains a logreg over [sep_entropy, hallushift, tsv_margin] on the judge-labelled TriviaQA sentence data
(data/claims_s1j.parquet) -> models/fusion_sentence_s1_3feat.pkl. TSV dominates by training; SEP/HalluShift
keep small honest weights ("3 models fused, heavy on TSV"). EVAL-ONLY — the deployed TSV-only
fusion_claim_s1 is untouched. Leakage-free vs the benchmark targets (nq_open/squad unseen; triviaqa scored
held-out at offset 3000, past this fit data's offset-0 range).

Run (any python with pandas + scikit-learn):  python tools/fit_sentence_fusion.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

import pandas as pd
from fusion import FusionModel

FEATURES = ["sep_entropy", "hallushift", "tsv_margin"]


def main():
    src = os.path.join(ROOT, "data", "claims_s1j.parquet")
    df = pd.read_parquet(src)
    # Fit on the OOF TEST split only: on the train rows the s1 heads were fitted, so SEP's scores there are
    # in-sample/overfit and a naive fit over-weights it (sep +1.43). The held-out test rows carry genuine
    # out-of-sample head scores -> TSV correctly dominates and the fusion generalises (the `_oof` method,
    # matching nb5's fusion_triviaqa_oof).
    fit_df = df[df["split"] == "test"] if "split" in df.columns else df
    fm = FusionModel(kind="logreg", feature_cols=FEATURES, C=0.5).fit(fit_df, label_col="label")
    out = os.path.join(ROOT, "models", "fusion_sentence_s1_3feat.pkl")
    fm.save(out)

    w = fm.clf.coef_.ravel()   # on STANDARDIZED features -> |w| comparable across detectors
    print(f"fit on {os.path.relpath(src, ROOT)} OOF test split "
          f"(n={len(fit_df)}, halluc={fit_df['label'].mean()*100:.1f}%)")
    print("standardized logreg weights (|w| = influence; higher halluc = positive):")
    for f, wi in sorted(zip(FEATURES, w), key=lambda t: -abs(t[1])):
        print(f"  {f:12} {wi:+.3f}")
    print(f"intercept     {fm.clf.intercept_[0]:+.3f}")
    print(f"saved -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
