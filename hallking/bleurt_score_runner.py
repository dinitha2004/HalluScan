"""BLEURT scoring runner — MUST be run in the isolated `bleurt_env`
(transformers 4.x + bleurt_pytorch), NOT in se_probes_env.

Reads a JSON list of records [{"prediction": str, "references": [str, ...]}, ...],
scores each prediction against its references with BLEURT-20 (taking the MAX over
references, exactly like the TSV/HalluShift ground-truth pipelines), and writes an
.npy array of per-example BLEURT scores.

Usage (called as a subprocess by hallking/gt_bleurt.py):
    bleurt_env/Scripts/python.exe bleurt_score_runner.py --in pairs.json --out scores.npy
"""
import argparse
import json

import numpy as np
import torch
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", dest="out", required=True)
ap.add_argument("--checkpoint", default="lucadiliello/BLEURT-20")
args = ap.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[bleurt] loading {args.checkpoint} on {device} ...", flush=True)
model = BleurtForSequenceClassification.from_pretrained(args.checkpoint).to(device).eval()
tokenizer = BleurtTokenizer.from_pretrained(args.checkpoint)

with open(args.inp, "r", encoding="utf-8") as f:
    records = json.load(f)

scores = np.zeros(len(records), dtype=np.float64)
with torch.no_grad():
    for i, rec in enumerate(records):
        pred = rec["prediction"]
        refs = rec.get("references", []) or []
        if len(refs) == 0:
            scores[i] = 0.0
            continue
        inputs = tokenizer([pred] * len(refs), refs, padding="longest", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits.flatten().tolist()
        scores[i] = float(np.max(logits))
        if i % 50 == 0:
            print(f"  scored {i}/{len(records)} (last={scores[i]:.3f})", flush=True)

np.save(args.out, scores)
print(f"[bleurt] saved {args.out} shape={scores.shape} "
      f"min={scores.min():.3f} max={scores.max():.3f} mean={scores.mean():.3f}", flush=True)
