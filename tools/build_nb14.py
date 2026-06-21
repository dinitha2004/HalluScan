"""Build notebooks/14_fused_benchmark_1b_crossdataset.ipynb — the cross-dataset transfer benchmark for the
**1B** (Llama-3.2-1B) model's heads + fusion, mirroring notebook 11 (the 8B thesis benchmark).

For each of 4 datasets: generate FRESH answers with the 1B, score the TriviaQA-trained l1b SEP/HalluShift/TSV
heads, apply the 1B's fusion, label with a DECOUPLED 8B judge, and report AUROC/AUPR/F1 + the FUSED-on-top
pivot. Nothing is re-fit on the targets -> every row is held out. Reuses tools/cross_eval.py (now model-aware)
and the parameter-free rank-mean fusion (tools/fit_crosseval_fusion.py).

Run:  python tools/build_nb14.py   ->  writes notebooks/14_fused_benchmark_1b_crossdataset.ipynb
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "14_fused_benchmark_1b_crossdataset.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}


cells = []

cells.append(md(
"""# 14 · Cross-dataset transfer benchmark — the 1B (Llama-3.2-1B) fused detector

Same robustness test as **notebook 11**, but for the **1B** model's heads + fusion (`tag l1b`, trained in
nb13). For each of 4 datasets we generate *fresh* answers with the 1B, score the TriviaQA-trained
SEP / HalluShift / TSV heads, apply the 1B's fusion, and report the full metric set — **nothing is re-fit on
the targets**, so every row is held out.

- **Model under test:** `Llama-3.2-1B-Instruct` with its `sentence_l1b` heads.
- **Labels:** a **decoupled 8B judge** grades correctness (a 1B is too weak to grade itself) — substring-match
  handles the easy cases, the judge only rescues misses.
- **Robustness:** `tools/cross_eval.py` is now model-aware (HalluShift layer count derived from the cached
  feature width), so the exact 8B benchmark code runs the 1B unchanged.

Run in `se_probes_env` (GPU). The 1B and the 8B judge load one at a time per dataset.
"""))

cells.append(code(
"""import os, sys
os.environ.setdefault('HF_HOME', r'D:/LLAMA CACHE/huggingface')
# Use the HF token cached by huggingface_hub.login; env-var tokens override it, so drop stale ones or the
# gated Llama-3.2 would 401 (same guard as nb13).
for _v in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
    os.environ.pop(_v, None)
sys.path.insert(0, os.path.abspath(os.path.join('..', 'hallking')))
sys.path.insert(0, os.path.abspath(os.path.join('..', 'tools')))
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
print('paths set')
"""))

cells.append(md(
"""## 0 · Config
Swap the model / datasets / sample count here. `FUSION_PKL` is the 1B's **deployed** fused model (the headline);
`RANK_PKL` is the parameter-free rank-mean used only in the ablation."""))

cells.append(code(
"""# == CONFIG =================================================================================
MODEL_ID    = 'meta-llama/Llama-3.2-1B-Instruct'        # model under test (its l1b heads)
JUDGE_MODEL = 'meta-llama/Meta-Llama-3.1-8B-Instruct'   # decoupled judge for labels (strong)
HEAD_SET    = 'sentence_l1b'                            # the 1B's per-sentence heads (artifacts *_sentence_l1b)
DATASETS    = ['triviaqa', 'squad', 'popqa', 'sciq']   # triviaqa held-out (offset 3000); the rest are transfer
N           = 100                                       # questions per dataset (use 50 for a quick smoke test)
OFFSETS     = {'triviaqa': 3000}                        # keep TriviaQA out of the heads' training range
FUSION_PKL  = 'models/fusion_claim_l1b.pkl'             # the 1B's DEPLOYED fused model -> the headline FUSED
RANK_PKL    = 'models/fusion_triviaqa_crosseval.pkl'    # parameter-free rank-mean (model-independent; ablation)
# ===========================================================================================
print(f'MODEL_ID={MODEL_ID}')
print(f'HEAD_SET={HEAD_SET} | DATASETS={DATASETS} | N={N} | FUSION_PKL={FUSION_PKL}')
"""))

cells.append(md(
"""## 1 · Build the parameter-free rank-mean fusion (ablation only; model-independent)
Rank-mean has no weights and no training data, so the SAME artifact works for any model — we just (re)write it
here for the ablation cell. It does NOT touch the 1B's deployed `fusion_claim_l1b.pkl`."""))

cells.append(code(
"""from fit_crosseval_fusion import build
build()   # writes models/fusion_triviaqa_crosseval.pkl (rank-mean of within-dataset percentile ranks)
"""))

cells.append(md(
"""## 2 · Run the benchmark (generate with the 1B → score l1b heads → fuse → label with the 8B judge)
GPU. Per dataset: the 1B generates fresh answers, its SEP/HalluShift/TSV heads score them, the 1B's fusion is
applied, and labels come from the decoupled 8B judge. The 1B and the 8B load one at a time (fits a T4)."""))

cells.append(code(
"""from cross_eval import evaluate_many
SCORED = evaluate_many(DATASETS, train_ds='triviaqa', n=N, offsets=OFFSETS,
                       label_method='llm_judge', head_set=HEAD_SET,
                       model_name=MODEL_ID, judge_model=JUDGE_MODEL, fusion_pkl=FUSION_PKL)
print('done:', list(SCORED.keys()))
"""))

cells.append(md(
"""## 3 · Per-dataset metrics (AUROC / AUPR / Accuracy / Precision / Recall / F1)"""))

cells.append(code(
"""import metrics as M
DETS = {'SEP':'sep_entropy', 'HalluShift':'hallushift', 'TSV':'tsv_margin', 'FUSED':'fused'}
METRICS = {}
for ds, (_, df) in SCORED.items():
    y = df['hallucination'].to_numpy()
    res = {}
    for name, col in DETS.items():
        s = df[col].to_numpy()
        m = M.detector_metrics(y, s, threshold=M.best_threshold(y, s))
        M.attach_curves(m, y, s)
        res[name] = m
    METRICS[ds] = res
    print(f'\\n=== {ds}  (n={len(df)}, halluc={y.mean()*100:.1f}%) ===')
    print(M.summary_table(res).to_string())
"""))

cells.append(md(
"""## 4 · ROC / PR curves + confusion matrices per dataset"""))

cells.append(code(
"""import metrics as M
import matplotlib.pyplot as plt
for ds, res in METRICS.items():
    fig, ax = plt.subplots(1, 2, figsize=(11,4))
    M.plot_roc(ax[0], res); M.plot_pr(ax[1], res)
    fig.suptitle(f'{ds} — ROC / PR (1B)'); plt.tight_layout(); plt.show()
    fig, axes = plt.subplots(1, 4, figsize=(15,3.4))
    for axx,(name,m) in zip(axes, res.items()):
        M.plot_confusion(axx, m['confusion_matrix'], title=f"{ds}:{name}\\nF1={m['F1']:.2f}")
    plt.tight_layout(); plt.show()
"""))

cells.append(md(
"""## 5 · Headline — does the 1B's FUSED detector stay on top across datasets?
Rows = dataset, cols = detector. Watch the **best single detector change** between rows while **FUSED stays
≥ the best single** — the robustness story for the 1B. `FUSED_wins` is True when fused ≥ the best individual head."""))

cells.append(code(
"""import pandas as pd
for metric in ['AUROC','AUPR','F1']:
    piv = pd.DataFrame({ds:{name:res[name][metric] for name in DETS}
                        for ds,res in METRICS.items()}).T.round(3)
    piv['best_single'] = piv[['SEP','HalluShift','TSV']].idxmax(axis=1)
    piv['FUSED_wins'] = piv['FUSED'] >= piv[['SEP','HalluShift','TSV']].max(axis=1)
    print(f'\\n=== {metric} (rows=dataset, cols=detector) ===')
    print(piv.to_string())
"""))

cells.append(md(
"""## 6 · Ablation — the 1B's deployed logreg fusion vs parameter-free rank-mean
Both applied to the **same** held-out detector scores (no re-generation), isolating the fusion method. Shows
whether the 1B's logreg fusion (fit on TriviaQA sentences) holds up across datasets, or whether the
parameter-free rank-mean transfers better (the nb11 thesis, re-checked for the 1B)."""))

cells.append(code(
"""from fusion import FusionModel
from sklearn.metrics import roc_auc_score
import os, pandas as pd
LOGREG = FusionModel.load(os.path.join('..', 'models', 'fusion_claim_l1b.pkl'))         # 1B deployed fusion
RANK   = FusionModel.load(os.path.join('..', 'models', 'fusion_triviaqa_crosseval.pkl'))  # rank-mean
F3 = ['sep_entropy', 'hallushift', 'tsv_margin']
rows = []
for ds, (_, df) in SCORED.items():
    y = df['hallucination'].to_numpy().astype(int)
    rows.append({'dataset': ds,
                 'FUSED_logreg_l1b': round(roc_auc_score(y, LOGREG.predict_proba(df[F3])), 3),
                 'FUSED_rankmean':   round(roc_auc_score(y, RANK.predict_proba(df[F3])), 3),
                 'best_single_head': round(max(roc_auc_score(y, df[c]) for c in F3), 3)})
ab = pd.DataFrame(rows).set_index('dataset')
print('AUROC — the 1B deployed logreg fusion vs the parameter-free rank-mean:')
print(ab.to_string())
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
