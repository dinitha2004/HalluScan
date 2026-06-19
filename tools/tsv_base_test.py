"""Confirm TSV works on its NATIVE base model (meta-llama/Meta-Llama-3.1-8B).
If truthful answers score low and wrong answers score high, the saturation seen on the
Instruct model is purely the base->Instruct variant mismatch, and the fix is to score TSV
on the base model in a decoupled pass.

Run in se_probes_env:  python tools/tsv_base_test.py
"""
import os, sys
os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hallking"))

from engine import HallKingEngine
from tsv_adapter import TSVAdapter

PAIRS = [
    ("What is the capital of France?", " Paris.", "truthful"),
    ("What is the capital of France?", " Berlin.", "hallucinated"),
    ("Who was the first person to walk on the moon?", " Neil Armstrong.", "truthful"),
    ("Who was the first person to walk on the moon?", " Buzz Aldrin was the first.", "hallucinated"),
]

import torch, torch.nn.functional as F
eng = HallKingEngine(model_name="meta-llama/Meta-Llama-3.1-8B").load()
tsv = TSVAdapter(eng).load()
print("\n=== TSV on BASE model (full precision; higher tsv = more hallucinated) ===")
for q, a, tag in PAIRS:
    prompt = f"Answer the question concisely. Q: {q} A:{a}"
    rep = tsv._last_token_rep(prompt)
    repn = F.normalize(rep.float(), p=2, dim=-1)
    sims_raw = torch.matmul(repn, tsv.centroids.T)            # cosine to [halluc, truthful]
    sims = sims_raw / tsv.cos_temp
    p_truth = torch.softmax(sims, dim=-1)[1].item()
    print(f"  [{tag:12s}] tsv={1-p_truth:.6f}  P_truth={p_truth:.6e}  "
          f"cos(halluc)={sims_raw[0]:.4f} cos(truth)={sims_raw[1]:.4f}  A={a!r}")
