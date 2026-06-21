"""Pre-push deploy check (no creds, no GPU): would a fresh clone of this repo actually run the backend?

Verifies (1) every file the Colab/Vercel deploy needs is present on disk, (2) those files are NOT git-ignored
(so they'd land in a clone), and (3) the heavy/regenerable dirs ARE git-ignored (so they won't bloat the repo).
Exit code 0 = ready to push; non-zero = something would break the friend's clone.

    python tools/check_deploy_ready.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files a fresh clone MUST contain for `notebooks/7_backend_colab.ipynb` -> backend/app.py to serve Option B.
REQUIRED = [
    # Option-B detector artifacts (small, committed — no Git LFS)
    # 8B (tag s1)
    "artifacts/sep/probes_sentence_s1.pkl",
    "artifacts/hallushift/hal_det_sentence_s1_model.pth",
    "artifacts/hallushift/hal_det_sentence_s1_scaler.pkl",
    "artifacts/tsv/best_checkpoint_sentence_s1.pt",
    "models/fusion_claim_s1.pkl",
    "models/fusion_claim_s1_thresholds.json",
    # 1B (tag l1b) — the second selectable model
    "artifacts/sep/probes_sentence_l1b.pkl",
    "artifacts/hallushift/hal_det_sentence_l1b_model.pth",
    "artifacts/hallushift/hal_det_sentence_l1b_scaler.pkl",
    "artifacts/tsv/best_checkpoint_sentence_l1b.pt",
    "models/fusion_claim_l1b.pkl",
    "models/fusion_claim_l1b_thresholds.json",
    # backend + the pipeline modules it imports
    "backend/app.py",
    "backend/requirements_colab.txt",
    "hallking/pipeline.py", "hallking/engine.py", "hallking/localize.py", "hallking/fusion.py",
    "hallking/risk.py", "hallking/claim_filter.py", "hallking/sep_adapter.py",
    "hallking/hallushift_adapter.py", "hallking/tsv_adapter.py", "hallking/sentence_segmenter.py",
    "hallking/functions.py", "hallking/classifier.py", "hallking/llm_layers.py", "hallking/metrics.py",
    # frontend (Vercel) + the deploy notebook
    "frontend/vercel.json", "frontend/package.json", "frontend/src/App.jsx",
    "notebooks/7_backend_colab.ipynb",
]

# Dirs/globs that MUST be git-ignored (heavy or regenerable — keep them out of the clone).
MUST_BE_IGNORED = [
    "frontend/node_modules", "frontend/dist", "trash",
    "data/demo_scores.jsonl", "models/fusion_claim_s1.pkl.sepled.bak",
]


def _ignored(path):
    """True if git would ignore `path` (check-ignore exits 0 when ignored)."""
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT)
    return r.returncode == 0


def main():
    problems = []

    print("== required files present & committable ==")
    for rel in REQUIRED:
        on_disk = os.path.exists(os.path.join(ROOT, rel))
        ignored = _ignored(rel)
        ok = on_disk and not ignored
        flag = "ok " if ok else "XX "
        note = "" if ok else (" MISSING" if not on_disk else " (git-IGNORED — won't be in the clone!)")
        print(f"  [{flag}] {rel}{note}")
        if not ok:
            problems.append(rel)

    print("\n== heavy/regenerable paths are git-ignored ==")
    for rel in MUST_BE_IGNORED:
        on_disk = os.path.exists(os.path.join(ROOT, rel))
        ignored = _ignored(rel)
        # only a problem if it exists on disk AND is not ignored (would get committed)
        ok = ignored or not on_disk
        flag = "ok " if ok else "XX "
        note = "" if ignored else (" (absent)" if not on_disk else " NOT IGNORED — would bloat the repo!")
        print(f"  [{flag}] {rel}{note}")
        if not ok:
            problems.append(rel)

    print()
    if problems:
        print(f"NOT READY — {len(problems)} issue(s): " + ", ".join(problems))
        return 1
    print("READY — a fresh clone has everything the backend needs, and no heavy dirs would be committed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
