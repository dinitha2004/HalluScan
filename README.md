# HalluScan — Fused LLM Hallucination Detector (SEP + HalluShift + TSV)

HalluScan combines three published hallucination detectors into one calibrated detector on a
**single shared generation** from **Llama-3.1-8B**, and can **localize which sentence** of a long
answer is hallucinated. A plain-language overview for non-experts is in
[`docs/HallKing_Overview.docx`](docs/HallKing_Overview.docx).

## The three detectors (frozen, reused as scorers)

| Detector | "Lens" | Signal it reads | Output |
|---|---|---|---|
| **SEP** (Semantic Entropy Probes, [2406.15927](https://arxiv.org/abs/2406.15927)) | semantic uncertainty | logistic probe on all-layer hidden states at the second-last token | `sep_entropy`, `sep_accuracy` |
| **HalluShift** ([2504.09482](https://arxiv.org/abs/2504.09482)) | internal-state dynamics | Wasserstein/cosine shifts across layers + token-prob stats → MLP | `hallushift` ∈ [0,1] |
| **TSV** (Steer LLM Latents, [2503.01917](https://arxiv.org/abs/2503.01917)) | learned latent geometry | steering vector at layer 9 + truthful/hallucinated centroids | `tsv_margin` = cos(halluc) − cos(truth) |

A small **fusion meta-classifier** (logistic-regression / gradient-boosting) combines the detector
scores into one calibrated `P(hallucination)`. Ground truth = **BLEURT-20 > 0.5 ⇒ truthful** (the
shared definition all three papers use).

### Fusion vs. a hand-weighted average of the three scores

A natural question: why not just pick weights and compute `w₁·SEP + w₂·HalluShift + w₃·TSV`? **The fusion
model *is* that weighted combination — only the weights are learned, not guessed.** Concretely it is a
`LogisticRegression` over the standardized `[sep_entropy, sep_accuracy, hallushift, tsv_margin]`
([fusion.py](hallking/fusion.py)): it fits one coefficient per detector (the "weight") plus an intercept,
then passes the weighted sum through a sigmoid to produce a **probability in [0,1]**. So vs. a manual
weighted average it adds three things: (1) weights chosen by data to best separate truthful/hallucinated
rather than by hand, (2) per-feature standardization so detectors on different scales (e.g. TSV's cosine
margin vs SEP's entropy) are comparable, and (3) probability **calibration** so the output is a usable risk,
not an arbitrary blended number. The demo shows all four — the three raw detector scores **and** the fused
probability — so you can see each lens and the learned combination side by side.

## Approach: re-train the three lightweight heads on ONE shared config (LLM frozen)

We first tried pure **frozen reuse** (run the published artifacts as-is, fuse). Empirically that fails
because each frozen artifact only works on its own original dataset + dtype and they **don't co-align**
on one shared generation (evidence below). So HalluScan keeps the **8B LLM completely frozen** but
**re-fits the tiny heads** on ONE shared dataset + config (fp16 throughout), which puts all three
in-distribution together and makes the fusion legitimate. The heads are cheap:
SEP probe (logistic regression, seconds), HalluShift MLP (~minutes), TSV steering vector + centroids
(supervised, ~0.1 GPU-hr). This is `tools/unified_retrain.py` (notebook 1).

## Key engineering findings (the evidence)

- **Model variant + dtype matter, bit-for-bit.** Reproducing HalluShift's exact config
  (`-Instruct`, **bfloat16**, `use_fast=False`) makes its features *bit-identical* to its training set
  (AUROC 0.89). **SEP needs float16** (its backend config): on the *same* answers, fp16 gives AUROC
  **0.66** vs bf16 **0.51**. TSV was trained on the **base** model in **float16**.
- **No single frozen config satisfies all three** (SEP wants fp16, HalluShift wants bf16, TSV wants its
  native dataset). Re-fitting all heads on one fp16 config removes the conflict.
- **TSV saturates in probability** (`cos_temp=0.1`); we use the rank-equivalent **cosine margin**
  (`cos_halluc − cos_truth`) for fusion + AUROC. The *frozen* TSV checkpoint was base-trained (its
  centroids don't separate cross-variant on Instruct), but when we **re-fit** the TSV head we train it
  **on the Instruct model** (notebook 1b) so all three detectors share ONE model — the live demo then
  loads a single 8B (~6 GB, fits 12 GB) and scores any question in seconds.
- **Eval on held-out TriviaQA** (SEP probe in-distribution; HalluShift trained on first 1000; OFFSET≥1000
  keeps it held out).

## Results (honest, out-of-fold on held-out TriviaQA)

Scoring the re-trained heads on their own training rows is **leakage** (the 135168-dim SEP probe
memorises → AUROC 1.0). The real evaluation is **out-of-fold**: re-fit SEP+HalluShift on the train
split, use TSV's own held-out margins, evaluate all three + fusion on the 300 held-out rows where
every score is OOF (`tools/honest_eval.py`, notebook 2).

| Detector | AUROC | AUPR | F1 |
|---|---|---|---|
| SEP | 0.676 | 0.535 | 0.601 |
| HalluShift | 0.810 | 0.740 | 0.682 |
| TSV | **0.830** | 0.717 | 0.683 |
| FUSED (logreg, C=0.3) | 0.809 | 0.707 | 0.680 |

After re-training TSV on the **Instruct** model (notebook 1b, single-model demo) TSV becomes the
**strongest single detector** (0.830, up from 0.799 on the base model). On this *single in-distribution*
dataset the fusion **ties** the top detector rather than beating it — expected once one detector dominates,
and the gap is within the 300-row bootstrap noise band. The signals are still genuinely complementary
(a parameter-free rank-average of HalluShift+TSV reaches **0.866**); the fusion's *measured advantage* is
**cross-dataset robustness** (notebook 5): it stays at/near the top when the best single detector changes
across datasets, which you can't predict in advance on a new distribution. (Fusion selection is fixed, not
tuned on test; a per-fold OOF TSV refit — needed to let the in-distribution fusion shine — is deferred.)

## Repo layout

```
hallking/      engine.py (shared 4-bit load + toggleable TSV steering) , {sep,hallushift,tsv}_adapter.py,
               pipeline.py, fusion.py, metrics.py, gt_bleurt.py, run_dataset.py, localize.py,
               + copied source modules (sentence_segmenter, claim_filter, functions, classifier, llm_layers)
artifacts/     trained detector files copied from the 3 repos (SEP probes, HalluShift .pth+scaler, TSV .pt)
hallking/      … + risk.py (demo risk tiers), localize.py (per-sentence), sentence_segmenter.py (pysbd),
               claim_filter.py (DeBERTa NLI claim judge)
backend/       app.py (FastAPI: 1 Instruct model serves all 3 detectors + fusion; serves frontend/dist),
               requirements_colab.txt
frontend/      Vite + React + Tailwind chatbot UI (adapted from the SEP web app): 3-tier per-sentence
               highlight + hover metrics + aggregate risk + in-app backend-URL field
notebooks/     0_overview · 1_build_and_retrain · 1b_retrain_tsv_instruct · 2_evaluate_oof · 3_evaluate ·
               4_demo_live · 5_cross_dataset_eval · 6_truthfulqa_judge · 7_backend_colab
run_local.py / run_local.ps1   one-command local demo (build frontend → backend → open browser)
docs/          HallKing_Overview.docx (newcomer-friendly writeup)
tools/         build_notebooks.py, make_overview_doc.py, sanity/eval helper scripts
reference/     original source files kept for reference (sep_engine, tsv_detector, hallushift pipeline, …)
data/ models/  generated feature tables + saved fusion models
```

## Environments (reuse the existing venvs — do NOT create new ones)

- **Main pipeline** → `se_probes_env` (`D:/Github Repositories/semantic-entropy-probes/se_probes_env`;
  transformers 5.x, torch cu118, bitsandbytes). Jupyter kernel registered as **`HallKing (se_probes_env)`**.
- **BLEURT ground truth** → `bleurt_env` (`D:/Github Repositories/tsv/bleurt_env`; TF CPU + `bleurt_pytorch`),
  invoked as a subprocess by `gt_bleurt.py`. BLEURT-20-D12 checkpoint is path-referenced from the
  hallushift repo (not copied — 646 MB).
- The gated Llama weights are reused from `D:/LLAMA CACHE/huggingface` (set `HF_HOME`).

`requirements.txt` is for a fresh env (e.g. **Colab**) only.

## How to run

1. `notebooks/0_overview_doc.ipynb` → regenerate the plain-language `.docx`. (instant, CPU)
2. `notebooks/1_build_and_retrain.ipynb` → **the long GPU job**: generate answers, cache features,
   re-fit all three heads, save `data/<dataset>_fused.parquet` + re-trained artifacts. Equivalent CLI:
   `python tools/unified_retrain.py --dataset triviaqa --n 1200 --offset 1000 --epochs_tsv 40`.
3. `notebooks/2_evaluate_oof.ipynb` → **honest out-of-fold** evaluation: re-fit SEP+HalluShift on the
   train split, fuse, report AUROC/AUPR/F1 on held-out rows. Saves `data/<ds>_eval_oof.parquet` +
   `models/fusion_<ds>_oof.pkl`. (~5 min, CPU). CLI: `python tools/honest_eval.py triviaqa`.
4. `notebooks/3_evaluate.ipynb` → per-detector vs fused **AUROC / AUPR / confusion matrices** + ROC/PR
   plots, vs the paper baselines. (instant, CPU)
4b. `notebooks/1b_retrain_tsv_instruct.ipynb` → re-fit the TSV head on the **Instruct** model so all
   three detectors share ONE model (single-model live demo). Reuses the parquet (no re-generation,
   ~0.1 GPU-hr), then refreshes the fusion. CLI: `python tools/retrain_tsv_instruct.py --dataset triviaqa`.
   Run this once before the demo.
5. `notebooks/4_demo_live.ipynb` → **single-model** interactive scoring (~6 GB, fits 12 GB; load once,
   score in seconds) + **per-sentence localization**. Needs the Instruct-TSV from notebook 1b.
6. `notebooks/5_cross_dataset_eval.ipynb` → **cross-dataset transfer over MANY datasets in one notebook**
   (the strongest held-out test): score the TriviaQA-trained heads + fusion on **fresh** answers from
   `nq_open` + `squad` + **held-out** `triviaqa` (offset 3000), ~300 Q each, and report confusion matrix +
   AUROC/AUPR/F1/Accuracy/Precision/Recall per dataset + a datasets×detectors pivot. TruthfulQA is excluded
   (broken BLEURT → notebook 6). **GPU pass.** CLI (single dataset): `python tools/cross_eval.py --target
   nq_open --train triviaqa`.
7. `notebooks/6_truthfulqa_judge.ipynb` → **proper TruthfulQA label**. BLEURT>0.5 mislabels TruthfulQA's
   free-form answers, so this re-labels with a comparative judge (correct-vs-incorrect): `nli` (DeBERTa-v3
   MNLI, ~1 GB, cached) or `bleurtacc` (TruthfulQA's own metric), then recomputes AUROC against the
   detector scores from notebook 5 — no 8B re-generation. CLI: `python tools/truthfulqa_judge.py nli`.
8. `notebooks/7_backend_colab.ipynb` → **host the web-demo backend on a Colab GPU + ngrok** (single Instruct
   model). Clone the repo, HF login, start `backend/app.py` on the fixed ngrok URL (the frontend auto-connects — nothing to paste).

## Live web demo (chatbot UI with hallucination detection)

A React chatbot UI (`frontend/`) talks to a FastAPI backend (`backend/app.py`) that runs **one** Instruct
model for all three detectors + the fusion. The demo serves the **Option-B per-sentence heads** (sentence
regime, tag `s1`): the model answers with one fact per short sentence, the answer is **defragmented into
sentences → claim-filtered → each factual claim scored** by SEP/HalluShift/TSV + the fusion. It shows a
**per-sentence** 3-tier highlight (green = ok / yellow = medium / red = high risk; only **factual-claim**
sentences are scored, fillers are dimmed), **hover** any sentence for SEP/HalluShift/TSV/FUSED, and an
**answer-level risk** = the **worst claim sentence** (so a long answer only lights up where it actually goes
wrong — the over-flagging of the earlier short-QA "Option A" path is gone). Risk tiers use the fusion's
**own calibrated thresholds** (`models/fusion_claim_s1_thresholds.json`), not the generic defaults. Per-sentence
scores remain *indicative* (no per-sentence ground truth). Needs the Option-B artifacts from
`notebooks/8_train_claim_detector.ipynb` (`artifacts/*/*_sentence_s1.*` + `models/fusion_claim_s1.pkl`).

**Run locally (one command):** `python run_local.py` (or `./run_local.ps1`) → builds the frontend, starts
the backend on your GPU, opens `http://localhost:8000`. `--dev` for Vite hot-reload, `--no-build` to reuse
the last build. Fits a 12 GB GPU (~6 GB model + ~0.7 GB NLI claim judge).

### Deploy for the viva (Vercel frontend + Colab GPU backend)

The split: the **frontend** is static on **Vercel**; the **backend** (8B + detectors) runs on a **Colab GPU**
exposed at a **fixed ngrok URL** (a reserved static domain). Because the URL never changes, the frontend is
hardcoded to target it (`DEFAULT_BACKEND_URL` in `frontend/src/App.jsx`) — **so there's nothing to paste.**
Repo: `https://github.com/dinitha2004/HalluScan`.

**One-time setup**
1. **Reserve a free static ngrok domain** — dashboard.ngrok.com → Domains → Create. It's already wired into both
   the notebook CONFIG cell (`NGROK_DOMAIN`) and the frontend default. (To change it, edit those two spots.)
2. **Push the repo** (`python tools/check_deploy_ready.py` first to confirm a clone has everything — artifacts
   `artifacts/*/*_sentence_s1.*` + `models/fusion_claim_s1.*` are committed, all < 1.1 MB, no Git LFS).
3. **Frontend → Vercel.** Import the repo, set **Root Directory = `frontend/`** (Vite auto-detected;
   `vercel.json` handles the SPA build). Deploy. This URL is permanent.

**Every session (≈ "Run all and go")**
4. Open `notebooks/7_backend_colab.ipynb` in Colab with a **T4 GPU** runtime. In the **CONFIG cell** paste your
   **HF token** (Llama-3.1 license accepted) and **ngrok authtoken** — *do not commit the notebook with tokens
   filled in.* Then **Runtime ▸ Restart session and run all**. First run downloads Llama-3.1-8B (~16 GB, a few min). **Keep the tab
   open** — the tunnel dies when it stops.
5. Open the Vercel site — it **auto-connects** to the fixed backend URL (the status dot turns green once the model
   is loaded). Nothing to paste; the top-bar field is only a manual override.

**To share back after a GPU job:** `data/<dataset>_fused.parquet` (and the printed AUROC table) — the per-detector
scores + labels, enough to verify the fusion vs. each individual.
