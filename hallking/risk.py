"""Risk tiering for the live demo: map a fused hallucination probability -> a tier + label.

Convention (shared everywhere): the fused score is P(hallucination) in [0, 1], HIGHER = worse.
Two thresholds split it into three bands:
  * p <  T_MED   -> "ok"      / "Reliable"
  * p <  T_HIGH  -> "medium"  / "Uncertain"
  * else         -> "high"    / "Likely Hallucinated"

Defaults: T_MED=0.50, T_HIGH=0.74. T_HIGH is the FUSED **F1-optimal / Youden-J threshold measured on the
held-out TriviaQA OOF set** (0.739) — i.e. the cut that actually separates truthful from hallucinated, not
a guessed 0.5. Both are overridable so the demo can be re-tuned without touching the UI.
"""

T_MED = 0.50
T_HIGH = 0.74

# Tier -> the colour the frontend uses (kept here so backend + UI agree on the meaning).
TIER_COLORS = {"ok": "green", "medium": "yellow", "high": "red", "filler": "grey"}

_LABELS = {"ok": "Reliable", "medium": "Uncertain", "high": "Likely Hallucinated", "filler": "Not a claim"}


def tier(p: float, t_med: float = T_MED, t_high: float = T_HIGH) -> str:
    """Return 'ok' | 'medium' | 'high' for a fused hallucination probability."""
    if p is None:
        return "filler"
    if p >= t_high:
        return "high"
    if p >= t_med:
        return "medium"
    return "ok"


def risk_label(p: float, t_med: float = T_MED, t_high: float = T_HIGH) -> str:
    """Human-readable label for the aggregate answer-level score."""
    return _LABELS[tier(p, t_med, t_high)]
