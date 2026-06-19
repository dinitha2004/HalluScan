"""CHEAP test: does each detector's AUROC rise or fall under CORRECT (judge) labels — WITHOUT re-running
the heads? Re-labels the existing data/<ds>_cross_eval.parquet (BLEURT) with the LLM-judge and recomputes
SEP/HalluShift/TSV/FUSED AUROC vs OLD(bleurt) vs NEW(judge), reusing the already-cached detector scores.
Only the 8B judge runs (a few min/dataset) — no generation, no head scoring. Settles the "does SEP just
track BLEURT's noise?" question directly.

Greedy generation is deterministic, so these scores == what a full judge re-run would produce; the only
difference is a full re-run also DROPS refusals. So this is a faithful, fast preview.

Run (se_probes_env, GPU):  python tools/relabel_cross_eval.py
"""
import os
import re
import sys

os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

OFFSETS = {"nq_open": 0, "squad": 0, "triviaqa": 3000}   # offset used in nb5 (for ref alignment)
DETS = [("sep_entropy", "SEP"), ("hallushift", "HalluShift"), ("tsv_margin", "TSV"), ("fused", "FUSED")]


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def _auroc(y, s):
    y = np.asarray(y).astype(int)
    return roc_auc_score(y, np.asarray(s, dtype=float)) if len(np.unique(y)) > 1 else float("nan")


def main():
    from engine import HallKingEngine
    from run_dataset import load_qa, INSTRUCT_MODEL
    from claim_label import label_hybrid
    eng = HallKingEngine(model_name=INSTRUCT_MODEL).load()
    summary = []
    for ds, off in OFFSETS.items():
        path = os.path.join(ROOT, "data", f"{ds}_cross_eval.parquet")
        if not os.path.exists(path):
            print(f"[skip] {os.path.relpath(path, ROOT)} not found"); continue
        df = pd.read_parquet(path).reset_index(drop=True)
        qs, rfs = load_qa(ds, n=len(df) + off + 500, offset=0)
        qref = {_norm(q): r for q, r in zip(qs, rfs)}
        refs = [qref.get(_norm(q), []) for q in df["question"]]
        cov = sum(1 for r in refs if r)
        print(f"\n############## {ds}  (n={len(df)}, refs matched {cov}/{len(df)}) ##############", flush=True)
        new_y, _ = label_hybrid(df["question"].tolist(), df["answer"].tolist(), refs, eng)
        old_y = df["hallucination"].to_numpy().astype(int)
        print(f"  halluc rate: BLEURT={old_y.mean()*100:.0f}%  ->  judge={new_y.mean()*100:.0f}% "
              f"| agreement={(old_y == new_y).mean()*100:.0f}%", flush=True)
        print(f"  {'detector':12} {'AUROC_bleurt':>12} {'AUROC_judge':>12}  delta")
        for col, name in DETS:
            if col not in df.columns:
                continue
            ab, aj = _auroc(old_y, df[col]), _auroc(new_y, df[col])
            print(f"  {name:12} {ab:12.3f} {aj:12.3f}  {aj-ab:+.3f}")
            summary.append({"dataset": ds, "detector": name, "AUROC_bleurt": round(ab, 3),
                            "AUROC_judge": round(aj, 3), "delta": round(aj - ab, 3)})
        df["label_bleurt"] = old_y
        df["label_judge"] = new_y
        df.to_parquet(os.path.join(ROOT, "data", f"{ds}_cross_eval_judged.parquet"))
    eng.unload()
    if summary:
        print("\n==== summary (AUROC: BLEURT labels vs judge labels) ====")
        print(pd.DataFrame(summary).pivot_table(index="detector", columns="dataset",
              values="delta").to_string())
        print("(positive delta = AUROC RISES under correct labels; negative = the detector was tracking "
              "BLEURT's noise.)")


if __name__ == "__main__":
    main()
