"""Evaluation metrics + plots for hallucination detectors.

Convention everywhere: score in [0, 1], HIGHER = more likely HALLUCINATED;
ground-truth label 1 = hallucinated, 0 = truthful. This matches the BLEURT GT
(BLEURT <= 0.5 -> hallucinated) used by all three source papers.
"""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score)


def best_threshold(y_true, scores, metric: str = "f1") -> float:
    """Threshold that maximizes F1 (default) or Youden's J. Useful for a fair confusion matrix
    when a detector's raw scores are not calibrated to a 0.5 cut-off (e.g. TSV's margin)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    cands = np.unique(scores)
    if len(cands) > 200:
        cands = np.quantile(scores, np.linspace(0, 1, 200))
    best_t, best_v = 0.5, -1.0
    for t in cands:
        pred = (scores >= t).astype(int)
        if metric == "j":
            tp = ((pred == 1) & (y_true == 1)).sum(); fn = ((pred == 0) & (y_true == 1)).sum()
            tn = ((pred == 0) & (y_true == 0)).sum(); fp = ((pred == 1) & (y_true == 0)).sum()
            tpr = tp / (tp + fn + 1e-9); fpr = fp / (fp + tn + 1e-9); v = tpr - fpr
        else:
            v = f1_score(y_true, pred, zero_division=0)
        if v > best_v:
            best_v, best_t = v, float(t)
    return best_t


def detector_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    """All headline metrics for one detector. `scores` higher => hallucinated."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # rows=true [truthful,halluc]
    return {
        "AUROC": float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan"),
        "AUPR": float(average_precision_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan"),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm,             # [[TN, FP], [FN, TP]]
        "threshold": threshold,
        "n": int(len(y_true)),
        "positives(halluc)": int(y_true.sum()),
    }


def summary_table(results: dict):
    """`results` = {detector_name: metrics_dict}. Returns a tidy pandas DataFrame."""
    import pandas as pd
    rows = []
    for name, m in results.items():
        rows.append({"detector": name, "AUROC": m["AUROC"], "AUPR": m["AUPR"],
                     "Accuracy": m["Accuracy"], "Precision": m["Precision"],
                     "Recall": m["Recall"], "F1": m["F1"]})
    return pd.DataFrame(rows).set_index("detector").round(4)


# ------------------------------------------------------------------ plots
def plot_roc(ax, results: dict):
    for name, m in results.items():
        if "_roc" in m:
            fpr, tpr = m["_roc"]
        else:
            continue
        ax.plot(fpr, tpr, label=f"{name} (AUROC={m['AUROC']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC"); ax.legend(loc="lower right", fontsize=8)


def plot_pr(ax, results: dict):
    for name, m in results.items():
        if "_pr" in m:
            rec, prec = m["_pr"]
        else:
            continue
        ax.plot(rec, prec, label=f"{name} (AUPR={m['AUPR']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall"); ax.legend(loc="lower left", fontsize=8)


def attach_curves(metrics: dict, y_true, scores):
    """Add ROC/PR curve arrays to a metrics dict for plotting."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    metrics["_roc"] = (fpr, tpr)
    metrics["_pr"] = (rec, prec)
    return metrics


def plot_confusion(ax, cm, title="Confusion Matrix"):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["truthful", "halluc"]); ax.set_yticklabels(["truthful", "halluc"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
