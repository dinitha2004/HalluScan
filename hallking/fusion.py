"""Fusion meta-classifier over the three frozen detectors.

Two variants (the plan's "build both & compare"):
  * score-level   : 4 scalar scores  [sep_entropy, sep_accuracy, hallushift, tsv]
  * feature-level : the 4 scores PLUS HalluShift's raw 71-dim feature block (richer)

Both are tiny sklearn models (LogisticRegression or HistGradientBoosting — no extra
deps beyond what se_probes_env already has) trained on the BLEURT ground-truth label
(1 = hallucinated). Outputs a calibrated P(hallucination), higher = more hallucinated.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

SCORE_FEATURES = ["sep_entropy", "sep_accuracy", "hallushift", "tsv_margin"]
HS_FEATURES = [f"hs_feat_{j:02d}" for j in range(71)]
FEATURE_LEVEL = SCORE_FEATURES + HS_FEATURES


class FusionModel:
    def __init__(self, kind: str = "logreg", feature_cols=None, C: float = 1.0):
        assert kind in ("logreg", "gbm")
        self.kind = kind
        self.C = C  # logreg L2 strength; smaller = more regularized (helps a small-sample blend)
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(SCORE_FEATURES)
        self.scaler = StandardScaler()
        if kind == "logreg":
            self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
        else:
            self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                      l2_regularization=1.0, random_state=42)

    def fit(self, df, label_col: str = "hallucination"):
        X = df[self.feature_cols].to_numpy(dtype=np.float64)
        y = df[label_col].to_numpy(dtype=int)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        return self

    def predict_proba(self, df) -> np.ndarray:
        X = df[self.feature_cols].to_numpy(dtype=np.float64)
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]

    def predict_proba_row(self, row: dict) -> float:
        X = np.array([[row[c] for c in self.feature_cols]], dtype=np.float64)
        return float(self.clf.predict_proba(self.scaler.transform(X))[0, 1])

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"kind": self.kind, "feature_cols": self.feature_cols, "C": self.C,
                         "scaler": self.scaler, "clf": self.clf}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls(kind=d["kind"], feature_cols=d["feature_cols"], C=d.get("C", 1.0))
        m.scaler = d["scaler"]
        m.clf = d["clf"]
        return m
