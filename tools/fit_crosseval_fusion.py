"""Build the recalibrated CROSS-DATASET-BENCHMARK fusion (thesis notebook 11) — isolated from everything else.

The naive `models/fusion_triviaqa_oof.pkl` (a logreg fit on TriviaQA) collapses on transfer: TSV/HalluShift
dominate the TriviaQA fit so SEP gets ~0 weight, then the blend fails on datasets where SEP is the BEST
detector (squad), and the TriviaQA-fit StandardScaler miscalibrates the linear combination under cross-dataset
scale shift. The fused score ends up slightly BELOW TSV everywhere.

Fix = a parameter-free rank-mean ensemble (`FusionModel(kind="rankmean")`): each detector is converted to its
within-dataset percentile rank, then averaged. No weights, no scaler, no training data -> nothing to overfit
to TriviaQA, and per-detector scale shift can't dominate. We commit to this method A PRIORI (not chosen by
peeking at the eval sets), so the benchmark stays honest.

Writes ONLY `models/fusion_triviaqa_crosseval.pkl`. Does NOT touch the deployed demo's `fusion_claim_s1.pkl`
or the notebook-2/3/4 `fusion_triviaqa_oof.pkl`.

Run (any python with numpy + scikit-learn):  python tools/fit_crosseval_fusion.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

from fusion import FusionModel

FEATURES = ["sep_entropy", "hallushift", "tsv_margin"]
OUT = os.path.join(ROOT, "models", "fusion_triviaqa_crosseval.pkl")


def build(out_path: str = OUT, verbose: bool = True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fm = FusionModel(kind="rankmean", feature_cols=FEATURES)  # parameter-free: no fit needed
    fm.save(out_path)
    if verbose:
        print(f"saved {os.path.relpath(out_path, ROOT)}  (kind=rankmean, features={FEATURES})")
        print("parameter-free rank-mean ensemble: fused = mean(within-dataset percentile rank of each detector)")
    return fm


if __name__ == "__main__":
    build()
