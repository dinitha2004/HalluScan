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
        assert kind in ("logreg", "gbm", "rankmean")
        self.kind = kind
        self.C = C  # logreg L2 strength; smaller = more regularized (helps a small-sample blend)
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(SCORE_FEATURES)
        self.scaler = StandardScaler()
        if kind == "logreg":
            self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=C)
        elif kind == "gbm":
            self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                      l2_regularization=1.0, random_state=42)
        else:  # rankmean — parameter-free; nothing to fit
            self.clf = None

    def fit(self, df, label_col: str = "hallucination"):
        if self.kind == "rankmean":
            return self  # parameter-free ensemble: no weights, no scaler to fit
        X = df[self.feature_cols].to_numpy(dtype=np.float64)
        y = df[label_col].to_numpy(dtype=int)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        return self

    def _proba1(self, X: np.ndarray) -> np.ndarray:
        """P(hallucination) per row. For logreg, compute it directly from coef_/intercept_ + the scaler stats
        instead of calling clf.predict_proba(): the sklearn pickle's fitted ARRAYS survive across versions, but
        predict_proba's internals do not (e.g. sklearn>=1.7 removed `multi_class`, so a model pickled there
        crashes older sklearn). This keeps the served fusion version-independent. GBM falls back to the method."""
        X = np.asarray(X, dtype=np.float64)
        if self.kind == "logreg":
            Xs = (X - self.scaler.mean_) / self.scaler.scale_
            z = Xs @ self.clf.coef_.T + self.clf.intercept_       # (n, 1) for binary logreg
            return (1.0 / (1.0 + np.exp(-z))).ravel()             # sigmoid = P(class 1 = hallucinated)
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]

    @staticmethod
    def _rankmean(X: np.ndarray) -> np.ndarray:
        """Mean of each feature's within-batch percentile rank (transductive). Higher = more hallucinated.
        Scale-free per feature, so cross-dataset shift in any single detector's range can't dominate, and
        every detector contributes equally — the parameter-free, no-overfit ensemble used by the cross-dataset
        benchmark (kind='rankmean'). Needs the whole batch (a single row has no rank), see predict_proba_row."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        if n == 1:
            raise ValueError("rankmean fusion needs a batch (>=2 rows) to rank; a single row has no rank")
        # average rank over rows -> [0,1] percentile per feature, then mean across features
        order = np.argsort(np.argsort(X, axis=0), axis=0).astype(np.float64)  # 0..n-1 ranks per column
        pct = order / (n - 1)
        return pct.mean(axis=1)

    def predict_proba(self, df) -> np.ndarray:
        X = df[self.feature_cols].to_numpy(dtype=np.float64)
        if self.kind == "rankmean":
            return self._rankmean(X)
        return self._proba1(X)

    def predict_proba_row(self, row: dict) -> float:
        if self.kind == "rankmean":
            raise ValueError("rankmean fusion is batch-only (transductive ranking) — call predict_proba() on a "
                             "DataFrame, not predict_proba_row(). It must never serve the live demo.")
        return float(self._proba1(np.array([[row[c] for c in self.feature_cols]]))[0])

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
