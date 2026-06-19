"""Reproduce HalluShift's own number by matching the notebook config EXACTLY
(Instruct, 4-bit NF4, NO torch_dtype => bfloat16 residual stream, use_fast=False,
greedy, max_new_tokens=64). If the frozen classifier recovers ~0.86 AUROC on freshly
extracted features, then frozen reuse works and the dtype/tokenizer config is the culprit.
"""
import os, sys
os.environ["HF_HOME"] = "D:/LLAMA CACHE/huggingface"
sys.path.insert(0, r"D:/Github Repositories/HallKing/hallking")

import numpy as np, pandas as pd, torch, pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
import functions
from classifier import CombinedNN

A = r"D:/Github Repositories/HallKing/artifacts/hallushift"
N = 100
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
tok = AutoTokenizer.from_pretrained(MODEL_ID, token=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=True, quantization_config=bnb,
                                             device_map="auto", low_cpu_mem_usage=True,
                                             attn_implementation="eager")  # NB: no torch_dtype
model.eval()
num_layers = model.config.num_hidden_layers
print("loaded", MODEL_ID, "num_layers", num_layers, "| residual dtype:",
      model.model.embed_tokens.weight.dtype)

ds = load_dataset("truthful_qa", "generation")["validation"].select(range(N))
base_prompt = "Answer the question concisely. Q: {question} A:"
rows = []
for i, row in enumerate(ds):
    prompt = tok(base_prompt.format(question=row["question"]), return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**prompt, do_sample=False, max_new_tokens=64,
                             pad_token_id=tok.eos_token_id, return_dict_in_generate=True,
                             output_hidden_states=True, output_attentions=True, output_logits=True)
    rows.append(functions.plot_internal_state_2(gen, num_layers)
                + functions.plot_internal_state_2(gen, num_layers, state="attention")
                + functions.probability_function(gen))
    del gen, prompt; torch.cuda.empty_cache()
    if i % 25 == 0:
        print(f"  {i}/{N}", flush=True)

df = pd.DataFrame(rows)  # 62 cols (cols 60,61 are prob lists)
# saved labels from the original run
resp = pd.read_csv(f"D:/Github Repositories/hallushift/results/truthfulqa_processed/"
                   f"hal_det_llama3_8B_truthfulqa_responses_with_bleurt.csv")
df_bleurt = pd.DataFrame({"hallucination": resp["hallucination"].values[:N]})
data = functions.data_preparation(df.copy(), df_bleurt, num_layers)
X = data.iloc[:, :-1].values.astype("float32"); y = data.iloc[:, -1].values.astype(int)

sd = torch.load(f"{A}/hal_det_llama3_8B_truthfulqa_model.pth", map_location="cpu")
scaler = pickle.load(open(f"{A}/hal_det_llama3_8B_truthfulqa_scaler.pkl", "rb"))
clf = CombinedNN(num_layers); clf.load_state_dict(sd); clf.eval()
with torch.no_grad():
    p = torch.sigmoid(clf(torch.tensor(scaler.transform(X), dtype=torch.float32))).numpy().ravel()
print(f"\n>>> FRESH-FEATURES AUROC (exact-config, n={N}): {roc_auc_score(y, p):.4f}  (target ~0.86)")

# compare to SAVED features
saved = pd.read_parquet("D:/Github Repositories/hallushift/results/truthfulqa_processed/"
                        "hal_det_llama3_8B_truthfulqa_dataset.pq")
Xs = saved.iloc[:N, :-1].values.astype("float64")
corr = np.corrcoef(X.ravel(), Xs.ravel())[0, 1]
print(f">>> feature correlation fresh-vs-saved: {corr:.4f}  (1.0 = identical)")
print(f">>> mean|fresh-saved|: {np.mean(np.abs(X - Xs)):.4f}")
