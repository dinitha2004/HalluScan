"""Precaution (option C): does SEP improve in float16 vs bfloat16?
Re-scores the FROZEN SEP probe on the EXACT answers already generated in
data/triviaqa_mini.parquet (no new generation — just one forward per row), under fp16,
and compares AUROC to the bf16 sep_entropy already stored. Isolates the dtype effect.

Run in se_probes_env:  python tools/sep_fp16_test.py
"""
import os, sys
os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
sys.path.insert(0, r"D:/Github Repositories/HallKing/hallking")

import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from engine import HallKingEngine, GEN_PROMPT
from sep_adapter import SEPAdapter

df = pd.read_parquet(r"D:/Github Repositories/HallKing/data/triviaqa_mini.parquet")
y = df["hallucination"].to_numpy()
print(f"n={len(df)} balance: truthful={int((y==0).sum())} halluc={int(y.sum())}")
print(f"SEP bf16 (stored)   AUROC={roc_auc_score(y, df['sep_entropy']):.3f}")

eng = HallKingEngine(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", fp16_nonquant=True).load()
sep = SEPAdapter(eng, probe_name="llama3-triviaqa").load()
tok = eng.tokenizer

ent_fp16, acc_fp16 = [], []
for i, r in enumerate(df.itertuples()):
    prompt = GEN_PROMPT.format(question=r.question)
    full = prompt + (" " + r.answer if not str(r.answer).startswith(" ") else r.answer)
    ids = tok(full, return_tensors="pt").input_ids
    plen = tok(prompt, return_tensors="pt").input_ids.shape[1]
    gen_result = {"sequences": ids, "prompt_len": plen}
    s = sep.score(gen_result)
    ent_fp16.append(s["sep_entropy"]); acc_fp16.append(s["sep_accuracy"])
    if i % 40 == 0:
        print(f"  {i}/{len(df)}", flush=True)

ent_fp16 = np.array(ent_fp16); acc_fp16 = np.array(acc_fp16)
print(f"\nSEP fp16 entropy     AUROC={roc_auc_score(y, ent_fp16):.3f}")
print(f"SEP fp16 1-accuracy  AUROC={roc_auc_score(y, 1 - acc_fp16):.3f}")
print(f"(corr fp16 vs bf16 entropy: {np.corrcoef(ent_fp16, df['sep_entropy'])[0,1]:.3f})")
