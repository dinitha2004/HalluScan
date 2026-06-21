"""Build notebooks/13_train_1b_heads.ipynb — train the 3 sentence-regime heads (SEP, HalluShift, TSV) +
fusion for a configurable ~1B chat model, with a decoupled strong judge for labels, and validate per-head
held-out AUROC IN the notebook before anything touches the demo.

This notebook reuses the DEPLOYED Option-B recipe (hallking/retrain.py + tools/train_claim_fusion.py) and
replicates the train_claim_heads orchestration inline, so the 8B trainer (tools/train_claim_heads.py) is left
untouched. The only shared-code change this relies on is retrain.py being model-aware (HalluShift feature
count from num_layers; optional decoupled judge_model) — both backward-compatible.

Run:  python tools/build_nb13.py   ->  writes notebooks/13_train_1b_heads.ipynb
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "13_train_1b_heads.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = []

cells.append(md(
"""# 13 · Train the 3 heads + fusion for a ~1B chat model (then validate before deploy)

Adds a **second, smaller model** to HalluScan. The 8B's heads can't be reused — they read that model's
internal states — so we **retrain SEP + HalluShift + TSV** for the new model on TriviaQA, fit the fusion,
and check **per-head held-out AUROC** here. **Nothing touches the live demo until these numbers look good.**

Same proven Option-B recipe as the 8B (`hallking/retrain.py` + `tools/train_claim_fusion.py`); this notebook
replicates the `train_claim_heads` orchestration inline so the deployed 8B trainer is left untouched.

- **Model under test:** `Llama-3.2-1B-Instruct` (a real instruction-tuned chat model — there is no "Llama 3.1 1B").
- **Labels:** generate with the 1B, then judge correctness with a **decoupled** strong model (the 8B) —
  a 1B is too weak to label its own data. Substring-match handles the easy cases; the judge only rescues misses.
- **Robustness:** SEP and TSV are already model-agnostic; HalluShift's feature count follows the layer count
  (39 for a 16-layer 1B vs 71 for the 8B) and is derived automatically below.

Run in `se_probes_env`. Cells 5 + 7 are the **GPU** passes. `SAVE=False` by default — review the AUROC table first.
"""))

cells.append(code(
"""import os, sys
os.environ.setdefault('HF_HOME', r'D:/LLAMA CACHE/huggingface')
# Use the HF token cached by `huggingface_hub.login` (the account with Llama-3.2 access). Env-var tokens
# OVERRIDE the cached one in huggingface_hub, so drop any stale HF_TOKEN / HUGGING_FACE_HUB_TOKEN here or
# loading the gated model would 401. (token=True in engine.py then falls through to the cached token.)
for _v in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
    os.environ.pop(_v, None)
sys.path.insert(0, os.path.abspath(os.path.join('..', 'hallking')))
sys.path.insert(0, os.path.abspath(os.path.join('..', 'tools')))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, pickle, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import retrain
from classifier import CombinedNN
ROOT = os.path.abspath('..')
DATA = os.path.join(ROOT, 'data'); ART = os.path.join(ROOT, 'artifacts')
SEED = 42
from huggingface_hub import get_token
print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '| HF token', (get_token() or '')[:6] + '...')
"""))

cells.append(md(
"""## 0 · Config
Swap the model, dataset, sample count, tag, etc. here. `BUILD=True` runs the GPU generate+cache once; set it
`False` afterwards to re-train heads from the cache without regenerating. `SAVE` gates writing the head
artifacts (keep `False` until the AUROC table looks good)."""))

cells.append(code(
"""# == CONFIG =================================================================================
MODEL_ID       = 'meta-llama/Llama-3.2-1B-Instruct'        # model under test (any chat/instruct HF id)
JUDGE_MODEL    = 'meta-llama/Meta-Llama-3.1-8B-Instruct'   # decoupled judge (strong). None -> 1B self-judges.
DATASET        = 'triviaqa'                                # any key in run_dataset.load_qa
N_SAMPLES      = 1500                                      # questions (refusal-drop yields fewer rows)
OFFSET         = 0
TAG            = 'l1b'                                     # artifact suffix: *_sentence_<TAG>, fusion_claim_<TAG>
MAX_NEW_TOKENS = 64
EPOCHS_TSV     = 40
TSV_STR_LAYER  = None                                      # None -> ~0.28*num_layers (8B used 9/32); or pin an int
FUSION_FEATS   = None                                      # None -> all 3 heads; e.g. ['tsv_margin'] for TSV-led
BUILD          = True                                      # True -> GPU generate+cache (run once); False -> reuse cache
SAVE           = False                                     # True -> write head artifacts + fusion (gate before deploy)
# ===========================================================================================
print(f'MODEL_ID    = {MODEL_ID}')
print(f'JUDGE_MODEL = {JUDGE_MODEL}')
print(f'DATASET={DATASET} N={N_SAMPLES} offset={OFFSET} TAG={TAG} | BUILD={BUILD} SAVE={SAVE}')
"""))

cells.append(md(
"""## 1 · (GPU) Generate one factual sentence per question + cache features + decoupled-judge labels
`gen_and_cache` (sentence regime) generates with the **1B**, caches the raw SEP (all-layer SLT stack) and
HalluShift features, drops refusals, then unloads the 1B and loads the **8B judge** to label
(substring-truthful + QA-judge rescue). One model is resident at a time."""))

cells.append(code(
"""RAW_PARQUET = os.path.join(DATA, f'claims_sent_{TAG}.parquet')
RAW_SEPFEAT = os.path.join(DATA, f'claims_sent_{TAG}_sepfeats.npy')

if BUILD:
    df, sep_feats = retrain.gen_and_cache(
        DATASET, n=N_SAMPLES, offset=OFFSET, max_new_tokens=MAX_NEW_TOKENS,
        instruct_model=MODEL_ID, judge_model=JUDGE_MODEL,
        regime='sentence', label_method='llm_judge', drop_refusals=True, verbose=True)
    df['source'] = f'qa:{DATASET}'
    os.makedirs(DATA, exist_ok=True)
    df.to_parquet(RAW_PARQUET); np.save(RAW_SEPFEAT, sep_feats)
    print(f'saved {os.path.relpath(RAW_PARQUET, ROOT)} {df.shape} + '
          f'{os.path.relpath(RAW_SEPFEAT, ROOT)} {sep_feats.shape}')
else:
    df = pd.read_parquet(RAW_PARQUET).reset_index(drop=True)
    sep_feats = np.load(RAW_SEPFEAT)
    print(f'loaded cache {df.shape} + sep_feats {sep_feats.shape}')

y = df['hallucination'].to_numpy().astype(int)
hs_cols = [c for c in df.columns if c.startswith('hs_feat_')]
n_hs = len(hs_cols)
NUM_LAYERS = 2 * ((n_hs - 11) // 4 + 1)      # invert HalluShift count 4*((L/2)-1)+11 -> L
assert len(df) == len(sep_feats), f'row mismatch df={len(df)} sep_feats={len(sep_feats)}'
assert len(np.unique(y)) == 2, f'need both classes; halluc rate={y.mean()*100:.1f}%'
print(f'rows={len(df)} | halluc={y.mean()*100:.1f}% | HalluShift feats={n_hs} -> NUM_LAYERS={NUM_LAYERS} '
      f'| SEP width={sep_feats.shape[1]}')
"""))

cells.append(md(
"""## 2 · (CPU + 1 GPU) Train the 3 heads on ONE shared split, score the held-out test
Stratified 75/25 split (seed 42), shared across heads so every per-head AUROC is on the **same** held-out
questions. SEP + HalluShift fit on CPU from the cached features; TSV trains on the **1B** (GPU). HalluShift
uses `NUM_LAYERS` from this model; TSV's steering layer scales with depth."""))

cells.append(code(
"""tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.25, stratify=y, random_state=SEED)
print(f'split: train={len(tr_idx)} test={len(te_idx)}')
auroc, aupr = {}, {}

# ---- SEP probe (CPU): fit on TRAIN, score all rows, AUROC on TEST ----
print('---- SEP probe (CPU) ----', flush=True)
sep_probe = retrain.retrain_sep(sep_feats[tr_idx].astype(np.float32), y[tr_idx], name=f'sentence_{TAG}')
Xf = sep_feats.astype(np.float32)
df['sep_entropy']  = sep_probe[0]['s_bmodel'].predict_proba(Xf)[:, 1]   # P(hallucinated)
df['sep_accuracy'] = sep_probe[0]['s_amodel'].predict_proba(Xf)[:, 1]   # P(truthful)
auroc['SEP'] = roc_auc_score(y[te_idx], df['sep_entropy'].to_numpy()[te_idx])
aupr['SEP']  = average_precision_score(y[te_idx], df['sep_entropy'].to_numpy()[te_idx])

# ---- HalluShift MLP (CPU): num_layers from THIS model -> correct feature count ----
print('---- HalluShift MLP (CPU) ----', flush=True)
hs_state, hs_scaler = retrain.retrain_hallushift(df.iloc[tr_idx].reset_index(drop=True), y[tr_idx],
                                                 num_layers=NUM_LAYERS, seed=SEED)
Xhs = hs_scaler.transform(df[hs_cols].to_numpy(dtype=np.float64))
m = CombinedNN(NUM_LAYERS); m.load_state_dict(hs_state); m.eval()
with torch.no_grad():
    df['hallushift'] = torch.sigmoid(m(torch.tensor(Xhs, dtype=torch.float32))).numpy().ravel()
auroc['HalluShift'] = roc_auc_score(y[te_idx], df['hallushift'].to_numpy()[te_idx])
aupr['HalluShift']  = average_precision_score(y[te_idx], df['hallushift'].to_numpy()[te_idx])

# ---- TSV head (GPU): the 1B is the TSV base model; steering layer scaled to depth ----
str_layer = int(TSV_STR_LAYER) if TSV_STR_LAYER is not None else max(1, round(0.28 * NUM_LAYERS))
print(f'---- TSV head (GPU, {MODEL_ID} fp16) | str_layer={str_layer}/{NUM_LAYERS} ----', flush=True)
tsv_ckpt, tsv_margin = retrain.train_tsv(df, base_model=MODEL_ID, epochs=EPOCHS_TSV,
                                         str_layer=str_layer, tr_idx=tr_idx, te_idx=te_idx, verbose=True)
df['tsv_margin'] = tsv_margin
auroc['TSV'] = roc_auc_score(y[te_idx], tsv_margin[te_idx])
aupr['TSV']  = average_precision_score(y[te_idx], tsv_margin[te_idx])

summary = pd.DataFrame([{'head': k, 'heldout_AUROC': round(auroc[k], 3), 'heldout_AUPR': round(aupr[k], 3)}
                        for k in auroc]).set_index('head')
print('\\n==== per-head HELD-OUT AUROC (sentence regime, decoupled-judge labels) ====')
print(summary.to_string())
summary
"""))

cells.append(md(
"""## 3 · GO / NO-GO checkpoint
A head that **separates** (held-out AUROC ≳ 0.65) is worth fusing. If all three sit near 0.5 even with clean
labels, the signal does not carry sentence-level factuality for this small model — that's an honest finding;
report it rather than shipping a flat detector. (The 8B demo ended up **TSV-led** because SEP/HalluShift were
weak there.) Use the table above to decide `FUSION_FEATS` in the config — e.g. set it to `['tsv_margin']` if
only TSV separates, then re-run the fusion cell."""))

cells.append(code(
"""# Scored table for the fusion step — always written (intermediate data, not a deployed head).
split = np.array(['train'] * len(df), dtype=object); split[te_idx] = 'test'
scored = pd.DataFrame({'prompt': df['question'], 'source': df['source'], 'answer': df['answer'],
                       'sep_entropy': df['sep_entropy'], 'sep_accuracy': df['sep_accuracy'],
                       'hallushift': df['hallushift'], 'tsv_margin': df['tsv_margin'],
                       'label': y, 'split': split})
os.makedirs(DATA, exist_ok=True)
scored.to_parquet(os.path.join(DATA, f'claims_{TAG}.parquet'))
print(f'wrote data/claims_{TAG}.parquet (scored, with split col)')

if SAVE:
    for sub in ('sep', 'hallushift', 'tsv'):
        os.makedirs(os.path.join(ART, sub), exist_ok=True)
    with open(os.path.join(ART, 'sep', f'probes_sentence_{TAG}.pkl'), 'wb') as f:
        pickle.dump(sep_probe, f)
    torch.save(hs_state, os.path.join(ART, 'hallushift', f'hal_det_sentence_{TAG}_model.pth'))
    with open(os.path.join(ART, 'hallushift', f'hal_det_sentence_{TAG}_scaler.pkl'), 'wb') as f:
        pickle.dump(hs_scaler, f)
    torch.save(tsv_ckpt, os.path.join(ART, 'tsv', f'best_checkpoint_sentence_{TAG}.pt'))
    print(f'saved _sentence_{TAG} head artifacts (SEP / HalluShift / TSV)')
else:
    print('SAVE=False - head artifacts NOT written (review AUROC first). Parquet written for the fusion test.')
"""))

cells.append(md(
"""## 4 · Fit + calibrate the fusion, and measure it (reuses `tools/train_claim_fusion.py`)
Group-split-safe fusion over the chosen heads on the out-of-sample test rows, with F1-calibrated thresholds.
Prints per-detector vs fused AUROC/AUPR/F1 + localization. Writes `models/fusion_claim_<TAG>.pkl` (+ thresholds)
only when `SAVE=True`."""))

cells.append(code(
"""import train_claim_fusion as TCF
res = TCF.train(tag=TAG, feats=FUSION_FEATS, save=SAVE, verbose=True)
print('\\nfusion thresholds:', {'t_med': res['t_med'], 't_high': res['t_high']})
res['summary']
"""))

cells.append(md(
"""## 5 · (GPU, optional) End-to-end check on the SERVED path
Loads the full `HallKingPipeline` for the 1B exactly as the backend would (per-tag artifacts, fusion,
thresholds) and scores a few questions. The inference adapters auto-match this model's layer count, so a
clean run here means the heads are ready to wire into the demo. Needs `SAVE=True` (artifacts on disk)."""))

cells.append(code(
"""fusion_pkl = os.path.join(ROOT, 'models', f'fusion_claim_{TAG}.pkl')
if SAVE and os.path.exists(fusion_pkl):
    from pipeline import HallKingPipeline
    from fusion import FusionModel
    pipe = HallKingPipeline(model_name=MODEL_ID, dataset=DATASET, separate_tsv=False,
                            sentence_tag=TAG, hs_tag=TAG).load()
    pipe.fusion = FusionModel.load(fusion_pkl)
    thr = json.load(open(os.path.join(ROOT, 'models', f'fusion_claim_{TAG}_thresholds.json')))
    pipe.t_med, pipe.t_high = float(thr['t_med']), float(thr['t_high'])
    for q in ['What is the capital of France?',
              'Who wrote the play Hamlet?',
              'In what year did the first man land on the moon?']:
        out = pipe.score_with_sentences(q, max_new_tokens=128)
        agg = out['aggregate']
        print('\\nQ:', q)
        print('A:', out['answer'])
        print('  ->', agg['label'], '| fused =', agg['fused'],
              '| sep=%.3f hs=%.3f tsv=%.3f' % (agg['sep_entropy'], agg['hallushift'], agg['tsv_margin']))
else:
    print('skip: set SAVE=True and run the save + fusion cells first (needs artifacts on disk).')
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "se_probes_env (3.12.9.final.0)", "language": "python",
                       "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py",
                          "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python",
                          "pygments_lexer": "ipython3", "version": "3.12.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {os.path.relpath(OUT, ROOT)} ({len(cells)} cells)")
