"""Ground-truth labelling via BLEURT — the SHARED definition used by all three papers:
a generation is TRUTHFUL iff max BLEURT similarity to the reference answers > 0.5,
otherwise HALLUCINATED (label 1).

BLEURT runs in the isolated `bleurt_env` (transformers 4.x + bleurt_pytorch); this
module (in se_probes_env) just dumps the prediction/reference pairs to JSON, invokes
`bleurt_score_runner.py` there as a subprocess, and reads back the scores. This avoids
the transformers-5 BLEURT breakage and keeps TF/torch-cpu out of the main env.
"""
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BLEURT_PY = r"D:/Github Repositories/tsv/bleurt_env/Scripts/python.exe"
RUNNER = os.path.join(_HERE, "bleurt_score_runner.py")


def bleurt_scores(predictions, references, bleurt_python: str = DEFAULT_BLEURT_PY,
                  workdir: str = None, checkpoint: str = "lucadiliello/BLEURT-20") -> np.ndarray:
    """predictions: list[str]; references: list[list[str]]. Returns per-example max-BLEURT score."""
    assert len(predictions) == len(references)
    workdir = workdir or os.path.join(os.path.dirname(_HERE), "data")
    os.makedirs(workdir, exist_ok=True)
    pairs_path = os.path.join(workdir, "_bleurt_pairs.json")
    scores_path = os.path.join(workdir, "_bleurt_scores.npy")

    records = [{"prediction": str(p), "references": [str(r) for r in (refs or [])]}
               for p, refs in zip(predictions, references)]
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    if not os.path.exists(bleurt_python):
        raise FileNotFoundError(
            f"bleurt_env python not found at {bleurt_python}. Pass bleurt_python=... "
            f"(on Colab, point it at a separate bleurt venv).")

    cmd = [bleurt_python, RUNNER, "--in", pairs_path, "--out", scores_path, "--checkpoint", checkpoint]
    print("[gt_bleurt] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return np.load(scores_path)


def bleurt_labels(predictions, references, threshold: float = 0.5, **kw):
    """Returns (labels, scores): label 1 = hallucinated (score <= threshold)."""
    scores = bleurt_scores(predictions, references, **kw)
    labels = (scores <= threshold).astype(int)  # 1 = hallucinated
    return labels, scores
