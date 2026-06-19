"""Build the HallKing notebooks (.ipynb) with nbformat. Re-run to regenerate.
Keeps notebook source in one maintainable place; notebooks stay thin and call hallking/*.

Run in se_probes_env:  python tools/build_notebooks.py
"""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks")
os.makedirs(NB, exist_ok=True)

SETUP = (
    "import os, sys\n"
    "os.environ.setdefault('HF_HOME', r'D:/LLAMA CACHE/huggingface')  # reuse local LLaMA cache\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..', 'hallking')))\n"
    "import warnings; warnings.filterwarnings('ignore')\n"
)


# Stream a child process's output to the cell LIVE (line-buffered, unbuffered child) instead of
# subprocess.run(..., check=True), which collects everything and shows nothing until the process
# exits — that made a 6 hr TSV retrain look frozen. tqdm bars + prints now appear as they happen.
STREAM = (
    "import subprocess\n"
    "_env = {**os.environ, 'PYTHONUNBUFFERED': '1'}\n"
    "print('running:', ' '.join(cmd), flush=True)\n"
    "_p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
    "                      text=True, bufsize=1, env=_env)\n"
    "for _line in _p.stdout:\n"
    "    print(_line, end='', flush=True)\n"
    "if _p.wait() != 0:\n"
    "    raise RuntimeError(f'subprocess failed (rc={_p.returncode})')\n"
)


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "hallking", "display_name": "HallKing (se_probes_env)",
                                 "language": "python"}
    path = os.path.join(NB, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


# ============================================================ 0_overview_doc
write("0_overview_doc.ipynb", [
    md("# 0 · Project overview (newcomer-friendly)\n\n"
       "Generates **`docs/HallKing_Overview.docx`** — a plain-language explanation of what each "
       "technique is, how it detects hallucinations, and what is novel about combining them.\n\n"
       "HallKing fuses three published detectors on a single shared generation:\n"
       "- **SEP** – *does the model sound unsure of the meaning?* (semantic-entropy probe)\n"
       "- **HalluShift** – *does the internal train-of-thought lurch between layers?* (distribution shift)\n"
       "- **TSV** – *does the answer point toward the learned truthful direction?* (steering vector)"),
    code(SETUP +
         "sys.path.insert(0, os.path.abspath(os.path.join('..', 'tools')))\n"
         "from make_overview_doc import build\n"
         "build()\n"
         "print('Open docs/HallKing_Overview.docx')"),
])

# ============================================================ 1_build_and_retrain
write("1_build_and_retrain.ipynb", [
    md("# 1 · Build dataset + re-train the 3 heads (the long GPU job)\n\n"
       "Runs `tools/unified_retrain.py` end-to-end on ONE shared dataset + config (fp16), LLM frozen:\n"
       "1. generate one answer/question (Instruct fp16) and cache SEP (135168-d) + HalluShift (71-d) features + BLEURT label,\n"
       "2. re-fit the SEP probe + HalluShift MLP on those features (CPU),\n"
       "3. re-train the TSV steering vector + centroids (base fp16),\n"
       "4. assemble scores, train+CV the fusion, print per-detector vs fused AUROC/AUPR.\n\n"
       "Why re-train: the *frozen* artifacts each only work on their own dataset/dtype and don't co-align "
       "on one shared generation; re-fitting the tiny heads on one config makes all three in-distribution "
       "together (the LLM stays frozen). Output → `data/<dataset>_fused.parquet` + re-trained artifacts.\n\n"
       "**This is the long GPU pass** (two 8B model loads, ~minutes/question with attentions)."),
    code(SETUP +
         "DATASET='triviaqa'; N=1200; OFFSET=1000; EPOCHS_TSV=40\n"
         "cmd=[sys.executable, os.path.join('..','tools','unified_retrain.py'),\n"
         "     '--dataset',DATASET,'--n',str(N),'--offset',str(OFFSET),'--epochs_tsv',str(EPOCHS_TSV)]\n"
         + STREAM),
    code("import pandas as pd\n"
         "df = pd.read_parquet(os.path.join('..','data',f'{DATASET}_fused.parquet'))\n"
         "print(df.shape, df['hallucination'].value_counts().to_dict())\n"
         "df[['question','answer','sep_entropy','hallushift','tsv_margin','bleurt','hallucination']].head(8)"),
])

# ============================================================ 2_evaluate_oof
write("2_evaluate_oof.ipynb", [
    md("# 2 · Honest out-of-fold evaluation + fusion\n\n"
       "The re-trained heads in notebook 1 were fit on **all** rows, so scoring those same rows is "
       "**in-sample** — the 135168-dim SEP probe memorises its training set (AUROC 1.0 = leakage, not "
       "skill). This notebook evaluates **out-of-fold** instead:\n\n"
       "1. reproduce TSV's exact 25% held-out split (so TSV's margins on the test rows are already OOF),\n"
       "2. re-fit SEP + HalluShift on the **train** rows only and score the **test** rows,\n"
       "3. train the fusion on train (with an inner 5-fold OOF so it isn't fed a memorised SEP feature),\n"
       "4. report honest AUROC / AUPR / F1 on the held-out test rows.\n\n"
       "Saves `data/<ds>_eval_oof.parquet` (for notebook 3) + `models/fusion_<ds>_oof.pkl`. **CPU, ~5 min.**"),
    code(SETUP +
         "sys.path.insert(0, os.path.join('..','tools'))\n"
         "from honest_eval import evaluate\n"
         "DATASET='triviaqa'\n"
         "results, oof = evaluate(DATASET)   # prints the honest table; goal: FUSED >= best individual\n"
         "results"),
])

# ============================================================ 5_cross_dataset_eval
write("5_cross_dataset_eval.ipynb", [
    md("# 5 · Cross-dataset transfer evaluation (MANY datasets, one notebook)\n\n"
       "Everything — the 3 heads **and** the fusion — was trained on **TriviaQA** (notebooks 1–2). Here we "
       "generate *fresh* answers for **several** datasets, score them with those already-trained heads, fuse, "
       "and report the full metric set per dataset. **Nothing is re-fit on the targets**, so every row is held "
       "out. The headline question: does **FUSED stay on top even when the best *single* detector changes** "
       "between datasets?\n\n"
       "**Datasets** (set in the config cell): `nq_open` + `squad` are unseen by the heads (BLEURT label "
       "valid → real transfer); `triviaqa` is the training set, scored **held-out** via a high `OFFSET` (3000, "
       "past the training range 1000–2200). **TruthfulQA is excluded here** (BLEURT label invalid) — use "
       "notebook 6's comparative NLI judge for it.\n\n"
       "**This is a GPU pass** (Instruct fp16; ~2 model loads per dataset). You'll see a `generate+cache` tqdm "
       "bar, a `scoring BLEURT…` line, then a `TSV scoring` bar, per dataset — if a bar advances it's alive.\n\n"
       "Each dataset is also saved to `data/<ds>_cross_eval.parquet` so you can re-plot without re-running."),
    code(SETUP +
         "sys.path.insert(0, os.path.join('..','tools'))\n"
         "from cross_eval import evaluate_many\n"
         "# ---- CONFIG -------------------------------------------------------------------------------\n"
         "DATASETS = ['nq_open', 'squad', 'triviaqa']   # truthfulqa -> notebook 6 (broken BLEURT here)\n"
         "TRAIN    = 'triviaqa'                          # heads + fusion were trained on this\n"
         "N        = 300                                 # questions per dataset\n"
         "OFFSETS  = {'triviaqa': 3000}                  # keep TriviaQA held-out (training used 1000-2200)\n"
         "# ------------------------------------------------------------------------------------------\n"
         "SCORED = evaluate_many(DATASETS, train_ds=TRAIN, n=N, offsets=OFFSETS)  # live tqdm per dataset\n"
         "print('done:', list(SCORED.keys()))"),
    md("### Per-dataset metrics (AUROC / AUPR / Accuracy / Precision / Recall / F1)"),
    code(SETUP +
         "import metrics as M, pandas as pd, numpy as np\n"
         "DETS = {'SEP':'sep_entropy', 'HalluShift':'hallushift', 'TSV':'tsv_margin', 'FUSED':'fused'}\n"
         "METRICS = {}\n"
         "for ds, (_, df) in SCORED.items():\n"
         "    y = df['hallucination'].to_numpy()\n"
         "    res = {}\n"
         "    for name, col in DETS.items():\n"
         "        s = df[col].to_numpy()\n"
         "        m = M.detector_metrics(y, s, threshold=M.best_threshold(y, s))\n"
         "        M.attach_curves(m, y, s)\n"
         "        res[name] = m\n"
         "    METRICS[ds] = res\n"
         "    print(f'\\n=== {ds}  (n={len(df)}, halluc={y.mean()*100:.1f}%) ===')\n"
         "    print(M.summary_table(res).to_string())"),
    md("### ROC / PR curves + confusion matrices per dataset"),
    code("import matplotlib.pyplot as plt\n"
         "for ds, res in METRICS.items():\n"
         "    fig, ax = plt.subplots(1, 2, figsize=(11,4))\n"
         "    M.plot_roc(ax[0], res); M.plot_pr(ax[1], res)\n"
         "    fig.suptitle(f'{ds} — ROC / PR'); plt.tight_layout(); plt.show()\n"
         "    fig, axes = plt.subplots(1, 4, figsize=(15,3.4))\n"
         "    for axx,(name,m) in zip(axes, res.items()):\n"
         "        M.plot_confusion(axx, m['confusion_matrix'], title=f\"{ds}:{name}\\nF1={m['F1']:.2f}\")\n"
         "    plt.tight_layout(); plt.show()"),
    md("### Headline — does FUSED stay on top across datasets?\n\n"
       "Pivot tables (rows = dataset, cols = detector). Watch the **best single detector change** between "
       "rows while **FUSED stays ≥ the best single** — that is the robustness story."),
    code("for metric in ['AUROC','AUPR','F1']:\n"
         "    piv = pd.DataFrame({ds:{name:res[name][metric] for name in DETS}\n"
         "                        for ds,res in METRICS.items()}).T.round(3)\n"
         "    piv['best_single'] = piv[['SEP','HalluShift','TSV']].idxmax(axis=1)\n"
         "    piv['FUSED_wins'] = piv['FUSED'] >= piv[['SEP','HalluShift','TSV']].max(axis=1)\n"
         "    print(f'\\n=== {metric} (rows=dataset, cols=detector) ===')\n"
         "    print(piv.to_string())"),
])

# ============================================================ 6_truthfulqa_judge
write("6_truthfulqa_judge.ipynb", [
    md("# 6 · TruthfulQA — proper truthfulness judge (fix the broken BLEURT label)\n\n"
       "On TruthfulQA the `BLEURT(answer, correct_refs) > 0.5` label is **invalid**: answers are "
       "free-form and adversarial, so they pile up at the 0.5 cut and get mislabelled (correct "
       "*'rainbows have no taste'* → halluc; wrong *'marry your grandparent'* → truthful). That's why "
       "every detector scored ≈0.5 or below in notebook 5 — they were graded against a broken key.\n\n"
       "TruthfulQA ships **both `correct_answers` AND `incorrect_answers`**, so the right test is "
       "**comparative**: is the answer closer to a CORRECT reference than to an INCORRECT one? Two "
       "free, locally-cached judges do this:\n"
       "- **`nli`** — a strong entailment model (DeBERTa-v3 MNLI, ~1 GB): does the answer *entail* a "
       "correct reference more than any incorrect one? (the 'strong model' check, tiny VRAM)\n"
       "- **`bleurtacc`** — TruthfulQA's own automatic metric: max BLEURT to correct > max BLEURT to "
       "incorrect. (free, reuses the BLEURT env, no new model)\n\n"
       "The detector **scores** already live in `data/truthfulqa_cross_eval.parquet` (from notebook 5) "
       "and don't depend on the label, so this just **swaps the label and recomputes AUROC** — no 8B "
       "re-generation. **Run notebook 5 with `TARGET='truthfulqa'` first** to produce that file."),
    code(SETUP +
         "sys.path.insert(0, os.path.join('..','tools'))\n"
         "from truthfulqa_judge import relabel_and_eval\n"
         "# ---- CONFIG --------------------------------------------------------------------------\n"
         "JUDGE = 'nli'        # 'nli' (strong entailment model, ~1GB) or 'bleurtacc' (free, BLEURT env)\n"
         "N     = None         # None = all 817; set e.g. 50 for a quick smoke test\n"
         "# --------------------------------------------------------------------------------------\n"
         "results, judged = relabel_and_eval(JUDGE, n=N)\n"
         "results"),
    md("### Read it\n"
       "`AUROC (BLEURT-0.5)` is the broken-label number from notebook 5; `AUROC (judge)` is against the "
       "proper label. If the detectors jump from ≈0.5 toward the TriviaQA range, that confirms the "
       "collapse was a **labelling artifact**, not classifier overfitting — and gives a real TruthfulQA "
       "transfer number. `agreement%` shows how often the BLEURT-0.5 label and the judge disagree."),
])

# ============================================================ 3_evaluate
write("3_evaluate.ipynb", [
    md("# 3 · Evaluation — per-detector vs. fused (AUROC, AUPR, confusion matrices)\n\n"
       "Headline metrics are **AUROC** and **AUPR** (threshold-free, the numbers the source papers "
       "report). Confusion matrices use each individual detector's best-F1 threshold and **0.5** for the "
       "calibrated fused model. Goal: **fused ≥ best individual**, in range of the published baselines."),
    code(SETUP +
         "import pandas as pd, numpy as np, matplotlib.pyplot as plt\n"
         "import metrics as M\n"
         "DATASET='triviaqa'\n"
         "test = pd.read_parquet(os.path.join('..','data',f'{DATASET}_eval_oof.parquet'))  # honest OOF, from nb2\n"
         "y = test['hallucination'].to_numpy()\n"
         "# detector score columns (higher = more hallucinated)\n"
         "detectors = {\n"
         "  'SEP':        test['sep_entropy'].to_numpy(),\n"
         "  'HalluShift': test['hallushift'].to_numpy(),\n"
         "  'TSV':        test['tsv_margin'].to_numpy(),\n"
         "  'FUSED':      test['fused'].to_numpy(),\n"
         "}"),
    code("res = {}\n"
         "for name, s in detectors.items():\n"
         "    thr = M.best_threshold(y, s, metric='f1')\n"
         "    m = M.detector_metrics(y, s, threshold=thr)\n"
         "    M.attach_curves(m, y, s)\n"
         "    res[name] = m\n"
         "summary = M.summary_table(res)\n"
         "summary"),
    code("# paper-reported AUROC for context (LLaMA-3.1-8B, TriviaQA; from each paper's own generation)\n"
         "paper = pd.DataFrame({'paper_AUROC_TriviaQA':{\n"
         "    'SEP':'~0.78 (Llama-2 in-dist; no 3.1-8B number)',\n"
         "    'HalluShift':0.99,   # their own run; our held-out reproduction is lower\n"
         "    'TSV':0.84,          # tqa-trained, transfers to triviaqa ~0.80\n"
         "    'FUSED':'(this work)'}})\n"
         "paper"),
    code("fig, ax = plt.subplots(1,2, figsize=(11,4))\n"
         "M.plot_roc(ax[0], res); M.plot_pr(ax[1], res)\n"
         "plt.tight_layout(); plt.show()"),
    code("fig, axes = plt.subplots(1,4, figsize=(15,3.4))\n"
         "for ax,(name,m) in zip(axes, res.items()):\n"
         "    M.plot_confusion(ax, m['confusion_matrix'], title=f\"{name}\\nF1={m['F1']:.2f} thr={m['threshold']:.2f}\")\n"
         "plt.tight_layout(); plt.show()"),
    code("# inter-detector correlation (low correlation => complementary => fusion helps)\n"
         "test[['sep_entropy','hallushift','tsv_margin']].corr().round(2)"),
])

# ============================================================ 1b_retrain_tsv_instruct
write("1b_retrain_tsv_instruct.ipynb", [
    md("# 1b · Re-train TSV on the Instruct model (one-model demo)\n\n"
       "After notebooks 1–2, TSV's steering vector is still trained on the **base** model while "
       "SEP+HalluShift run on **-Instruct** — so the live demo would need TWO 8B models loaded at once "
       "(OOMs a 12 GB GPU; load/unload per question is unusable). This re-fits the tiny TSV head on the "
       "**Instruct** model so all three detectors share ONE model and the demo scores any question in "
       "seconds.\n\n"
       "**Cheap (~0.1 GPU-hr):** reuses the answers + labels already in `data/triviaqa_fused.parquet` "
       "(no re-generation). Overwrites `best_checkpoint_retrained.pt` (now Instruct-trained) and the "
       "`tsv_margin` column, then refreshes the fusion. Run cell 1 (GPU) then cell 2 (CPU)."),
    code(SETUP +
         "DATASET='triviaqa'; EPOCHS_TSV=40\n"
         "cmd=[sys.executable, os.path.join('..','tools','retrain_tsv_instruct.py'),\n"
         "     '--dataset',DATASET,'--epochs_tsv',str(EPOCHS_TSV)]\n"
         + STREAM),
    md("### Refresh the honest OOF eval + fusion on the new Instruct-TSV margins\n"
       "Re-runs notebook 2's evaluation so `models/fusion_<ds>_oof.pkl` matches the new TSV. (CPU, ~5 min)"),
    code(SETUP +
         "sys.path.insert(0, os.path.join('..','tools'))\n"
         "from honest_eval import evaluate\n"
         "DATASET='triviaqa'\n"
         "results, oof = evaluate(DATASET)\n"
         "results"),
])

# ============================================================ 4_demo_live
write("4_demo_live.ipynb", [
    md("# 4 · Live demo — answer-level score + per-sentence hallucination localization\n\n"
       "Scores questions end-to-end (answer + 3 detector scores + fused probability) and "
       "**highlights which sentence** of a longer answer is likely hallucinated.\n\n"
       "**Single model (fast, interactive).** All three detectors run on ONE Instruct model (~6 GB, fits "
       "12 GB easily) — load it once, then every `pipe.score(q)` takes a couple of seconds. This needs "
       "the **Instruct-trained TSV** from notebook 1b; if you haven't run 1b, run it first (otherwise TSV "
       "is on the base model and you'd need the slow two-model path).\n\n"
       "*(Per-sentence scores are indicative; the calibrated number is the answer-level fused score.)*"),
    code(SETUP +
         "import torch, gc\n"
         "from pipeline import HallKingPipeline\n"
         "from fusion import FusionModel\n"
         "from localize import localize, render_highlight\n"
         "DATASET='triviaqa'\n"
         "gc.collect(); torch.cuda.empty_cache()\n"
         "if torch.cuda.is_available():\n"
         "    free, total = torch.cuda.mem_get_info()\n"
         "    print(f'VRAM before load: {free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total')\n"
         "# separate_tsv=False -> ONE Instruct model for SEP+HalluShift+TSV (TSV re-trained in nb 1b).\n"
         "pipe = HallKingPipeline(dataset=DATASET, separate_tsv=False, retrained=True).load()\n"
         "pipe.fusion = FusionModel.load(os.path.join('..','models',f'fusion_{DATASET}_oof.pkl'))\n"
         "print('pipeline + fusion ready (single model)')"),
    md("### Answer-level scoring (interactive — each call is ~seconds, no reload)"),
    code("for q in ['What is the capital of Australia?',\n"
         "          'Who invented the telephone?',\n"
         "          'What is the largest planet in the solar system?']:\n"
         "    r = pipe.score(q)\n"
         "    print(f\"Q: {q}\\n  A: {r['answer']!r}\")\n"
         "    print(f\"  sep_entropy={r['sep_entropy']:.2f} hallushift={r['hallushift']:.2f} \"\n"
         "          f\"tsv_margin={r['tsv_margin']:.3f}  ->  FUSED P(halluc)={r['fused']:.2f}\\n\")"),
    md("### Score your own question\nChange `q` and re-run — the model stays loaded, so it's instant."),
    code("q = 'What year did the first human land on Mars?'\n"
         "r = pipe.score(q)\n"
         "print('A:', repr(r['answer']))\n"
         "print(f\"sep_entropy={r['sep_entropy']:.2f} hallushift={r['hallushift']:.2f} \"\n"
         "      f\"tsv_margin={r['tsv_margin']:.3f}  ->  FUSED P(halluc)={r['fused']:.2f}\")"),
    md("### Per-sentence localization on a longer answer\n"
       "`use_claim_filter=True` skips filler/meta sentences (loads a small DeBERTa NLI judge once); set it "
       "to `False` to score every sentence without the extra model."),
    code("q = 'Tell me three facts about the planet Mars.'\n"
         "res = localize(pipe, q, max_new_tokens=200, use_claim_filter=True)\n"
         "print('ANSWER:\\n', res['answer'], '\\n')\n"
         "print('PER-SENTENCE (tier · fused probability; fillers shown as filler/None):')\n"
         "for s in res['sentences']:\n"
         "    f = 'filler' if s['fused'] is None else f\"{s['fused']:.2f}\"\n"
         "    print(f\"  [{s['tier']:6s} {f:>6}] {s['sentence']}\")"),
])

# ============================================================ 7_backend_colab
write("7_backend_colab.ipynb", [
    md("# 7 · Host the HalluScan backend on Colab (GPU) + ngrok\n\n"
       "Runs the **same** `backend/app.py` as the local demo on a Colab **T4** (single Instruct model ~6 GB + "
       "DeBERTa NLI ~0.7 GB), exposed at a **fixed ngrok URL**. The Vercel frontend already points at that URL, "
       "so there is **nothing to paste** — just keep this notebook running.\n\n"
       "It serves the **Option-B per-sentence detector** automatically (`HALLKING_SENTENCE_TAG=s1`), loading "
       "`artifacts/*/*_sentence_s1.*` + `models/fusion_claim_s1.pkl`. A long answer is split into sentences and "
       "each factual claim is scored.\n\n"
       "**Every session:** fill the two tokens in the CONFIG cell → **Runtime ▸ Run all**. "
       "**Keep this tab open** — closing it drops the tunnel."),
    md("### 0 · CONFIG — the only cell you edit\n"
       "Paste your two tokens, then run all. ⚠️ **These tokens are runtime-only — never commit this notebook with "
       "them filled in** (the GitHub copy must stay blank). `NGROK_DOMAIN` is the reserved static domain so the URL "
       "never changes; leave it as-is."),
    code("# ===== CONFIG (edit the two tokens, then Runtime > Run all) ==============================\n"
         "REPO            = 'https://github.com/dinitha2004/HalluScan.git'\n"
         "HF_TOKEN        = ''   # huggingface.co/settings/tokens (Read). Llama-3.1 license must be accepted.\n"
         "NGROK_AUTHTOKEN = ''   # dashboard.ngrok.com -> Your Authtoken\n"
         "NGROK_DOMAIN    = 'declared-angular-matchbox.ngrok-free.dev'  # reserved static domain (leave as-is)\n"
         "# =======================================================================================\n"
         "assert HF_TOKEN.strip(), 'Paste HF_TOKEN above (huggingface.co/settings/tokens).'\n"
         "assert NGROK_AUTHTOKEN.strip(), 'Paste NGROK_AUTHTOKEN above (dashboard.ngrok.com -> Your Authtoken).'\n"
         "print('config set; domain =', NGROK_DOMAIN or '(random)')"),
    md("### 1 · Get the code + install deps\n"
       "The repo is **public**, so no token is needed to clone. (`git clone` makes a `HalluScan/` folder.) The ML "
       "stack is installed explicitly (Colab already has a compatible torch)."),
    code("import os\n"
         "if not os.path.isdir('HalluScan'):\n"
         "    !git clone $REPO\n"
         "%cd HalluScan\n"
         "!pip install -q transformers accelerate bitsandbytes sentencepiece scikit-learn pandas pyarrow\n"
         "!pip install -q fastapi 'uvicorn[standard]' pyngrok pysbd nest_asyncio requests"),
    md("### 2 · Hugging Face login (gated Llama-3.1-8B)\n"
       "Uses `HF_TOKEN` from the CONFIG cell (must have accepted the Llama-3.1 license). First load downloads the "
       "8B weights ~16 GB — a few minutes on a fresh runtime."),
    code("from huggingface_hub import login\n"
         "login(token=HF_TOKEN) if HF_TOKEN.strip() else login()"),
    md("### 3 · Start the API (background thread) + open the fixed ngrok tunnel\n"
       "Sets the ngrok authtoken, then opens the tunnel on your **reserved static domain** so the URL is always the "
       "same. `nest_asyncio` lets uvicorn run inside Colab's event loop; the model loads on startup (~1–2 min on T4 "
       "after weights are cached)."),
    code("import sys, threading, uvicorn, nest_asyncio\n"
         "nest_asyncio.apply()\n"
         "from pyngrok import ngrok, conf\n"
         "conf.get_default().auth_token = NGROK_AUTHTOKEN.strip()\n"
         "sys.path.insert(0, 'backend'); sys.path.insert(0, 'hallking')\n"
         "import app  # backend/app.py (FastAPI instance = app.app); auto-serves Option B (tag s1)\n"
         "def _serve():\n"
         "    uvicorn.run(app.app, host='0.0.0.0', port=8000, log_level='info')\n"
         "threading.Thread(target=_serve, daemon=True).start()\n"
         "ngrok.kill()  # drop any tunnel from a previous run before reclaiming the static domain\n"
         "public = (ngrok.connect(8000, 'http', domain=NGROK_DOMAIN) if NGROK_DOMAIN.strip()\n"
         "          else ngrok.connect(8000, 'http'))\n"
         "print('\\n========================================================')\n"
         "print(' PUBLIC URL (the frontend already targets this):')\n"
         "print('  ', public.public_url)\n"
         "print('========================================================\\n')\n"
         "# (no ngrok? fallback: !pip install cloudflared && use a cloudflared quick tunnel on port 8000)"),
    md("### 4 · Wait for model load + smoke-test\n"
       "Polls `/status` until the model is loaded (check `regime=sentence`, `sentence_tag=s1`, and the calibrated "
       "`t_med`/`t_high`), then runs a short and a long question. The long one should split into multiple "
       "sentences — confirming the per-sentence pipeline works end to end. After this, open the Vercel site (no "
       "pasting needed) and ask away."),
    code("import requests, time\n"
         "for _ in range(60):\n"
         "    try:\n"
         "        s = requests.get('http://localhost:8000/status', timeout=5).json()\n"
         "        if s.get('model_loaded'):\n"
         "            print('status:', s); break\n"
         "    except Exception:\n"
         "        pass\n"
         "    time.sleep(5)\n"
         "else:\n"
         "    print('model still loading — re-run this cell in a moment')\n"
         "for q in ['Who painted the Mona Lisa?', 'Tell me about the Sigiriya rock fortress.']:\n"
         "    out = requests.post('http://localhost:8000/infer', json={'question': q}, timeout=120).json()\n"
         "    agg = out['aggregate']\n"
         "    print(f\"\\nQ: {q}\\nA: {out['answer'][:160]}\")\n"
         "    print(f\"  -> {agg['label']} (fused={agg['fused']}) | {agg['n_flagged']}/{agg['n_sentences']} flagged\")"),
])

# ============================================================ 8_train_claim_detector
write("8_train_claim_detector.ipynb", [
    md("# 8 · Re-train the 3 heads at SENTENCE level, then fuse (Option B)\n\n"
       "The Option-A heads were trained on **2-word** TriviaQA answers (BLEURT string-match) — they track "
       "answer *form*, not sentence factuality, so the demo over-flags full sentences. Here we retrain each "
       "head on the **same regime the live demo uses** — ONE forced factual sentence per question (Instruct "
       "chat template) — with cheap, accurate **reference-match** labels (the answer's known aliases). "
       "Train == inference, so the per-sentence scores are finally in-distribution.\n\n"
       "**Method (the right order): make each technique work FIRST, then fuse.**\n"
       "1. **Config** — every knob in one cell.\n"
       "2. **Build dataset** *(GPU)* — generate one sentence per question; cache RAW SEP (135168-d) + "
       "HalluShift (71-d) features + `(question, answer)` for TSV; label by reference match; drop refusals "
       "(\"I don't know\" is not a claim). Cached so head re-training needs no GPU re-gen.\n"
       "3. **Re-train each head + per-head HELD-OUT AUROC** *(CPU + 1 GPU for TSV)* — the **decisive "
       "checkpoint**: if a head separates (AUROC >~0.65) it's worth fusing; if all stay ~0.5 that's the honest "
       "finding.\n"
       "4. **Fuse + evaluate** *(CPU)* — only if the heads separate: fuse the 3 scores on the SAME held-out "
       "split (test rows scored by heads that never saw them — no leakage); AUROC/AUPR/F1 + confusion + ROC/PR.\n"
       "5. **Smoke-test the product** *(GPU)* — flag each sentence of a fresh answer via the Option-B heads.\n\n"
       "**Re-running with different settings = edit the CONFIG cell only.** Live progress bars throughout.\n\n"
       "*Honest note:* empirical run — no guaranteed accuracy; the per-head AUROC cell **measures** whether "
       "the signals carry sentence-level factuality before we invest in fusion."),
    code(SETUP +
         "sys.path.insert(0, os.path.join('..','tools'))\n"
         "# ===== CONFIG (edit here) ==================================================================\n"
         "TAG       = 's1'           # tag -> artifacts/*/*_sentence_<TAG>.*, data/claims_<TAG>.parquet, fusion_claim_<TAG>.pkl\n"
         "DATASETS  = ['triviaqa']   # named-entity QA where reference-match works; add 'nq_open' for more data\n"
         "N         = 1500           # questions per dataset\n"
         "OFFSET    = 0              # start index into the dataset\n"
         "MAX_NEW_TOKENS = 64        # one factual sentence\n"
         "EPOCHS_TSV = 40            # TSV steering-vector epochs\n"
         "C         = 0.5            # fusion L2 regularization\n"
         "# ==========================================================================================\n"
         "print('config set; TAG =', TAG, '| datasets =', DATASETS, '| N =', N)"),
    md("### 2 · Build the sentence dataset — **GPU pass** (generate + RAW features + reference labels)\n"
       "Generates one factual sentence per question, caches RAW per-head features + reference-match labels, "
       "drops refusals. Saves `data/claims_sent_<TAG>.parquet` + `_sepfeats.npy` so step 3 can re-run heads "
       "**without** re-generating. Start with a small `N` to smoke-test before scaling."),
    code(SETUP +
         "sys.path.insert(0, os.path.join('..','tools'))\n"
         "from train_claim_heads import build\n"
         "df, sep_feats = build(tag=TAG, datasets=DATASETS, n=N, offset=OFFSET, max_new_tokens=MAX_NEW_TOKENS)\n"
         "print('label balance (0=truthful, 1=halluc):', df['hallucination'].value_counts().to_dict())\n"
         "df[['question','answer','hallucination','source']].head(8)"),
    md("### 3 · Re-train each head + **per-head HELD-OUT AUROC** — the decisive checkpoint *(CPU + 1 GPU for TSV)*\n"
       "ONE shared 75/25 split; refits the SEP probe + HalluShift MLP + TSV vector on TRAIN and scores the "
       "held-out TEST. **Read the table before going further:** a head with AUROC >~0.65 carries sentence-level "
       "factuality and is worth fusing; if all three hover ~0.5, these signals don't separate sentences and "
       "fusion won't rescue them (the honest finding). Saves the `_sentence_<TAG>` head artifacts + a scored "
       "`data/claims_<TAG>.parquet`."),
    code(SETUP +
         "sys.path.insert(0, os.path.join('..','tools'))\n"
         "from train_claim_heads import train_heads\n"
         "H = train_heads(tag=TAG, epochs_tsv=EPOCHS_TSV)\n"
         "H['summary']"),
    md("### 4 · Fuse + evaluate *(CPU)* — run only if at least one head separated\n"
       "Fuses the 3 scores using the SAME split (TEST rows are out-of-sample for the heads → no leakage); "
       "thresholds picked on TRAIN. Prints per-detector vs FUSED AUROC/AUPR/F1 and saves "
       "`models/fusion_claim_<TAG>.pkl` (+ thresholds)."),
    code(SETUP +
         "sys.path.insert(0, os.path.join('..','tools'))\n"
         "from train_claim_fusion import train\n"
         "R = train(tag=TAG, C=C)\n"
         "R['summary']"),
    md("### Confusion matrices + ROC / PR"),
    code(SETUP +
         "import matplotlib.pyplot as plt, metrics as M\n"
         "res = R['metrics']\n"
         "fig, ax = plt.subplots(1, 2, figsize=(11,4)); M.plot_roc(ax[0], res); M.plot_pr(ax[1], res)\n"
         "fig.suptitle('Per-sentence detection — ROC / PR'); plt.tight_layout(); plt.show()\n"
         "fig, axes = plt.subplots(1, 4, figsize=(15,3.4))\n"
         "for axx,(name,m) in zip(axes, res.items()):\n"
         "    M.plot_confusion(axx, m['confusion_matrix'], title=f\"{name}\\nF1={m['F1']:.2f}\")\n"
         "plt.tight_layout(); plt.show()"),
    md("### Thresholds + a note on localization\n"
       "Training is one sentence per question, so within-passage 'find-the-wrong-sentence' is **not** measured "
       "on this set (each prompt has a single claim) — it is demonstrated on a real multi-sentence answer in "
       "the smoke test below."),
    code("loc = R['localization']\n"
         "if loc['n_multiclaim_prompts'] > 0:\n"
         "    print(f\"Find-the-wrong-sentence (top-1): {loc['localization_top1']:.3f} over \"\n"
         "          f\"{loc['n_multiclaim_prompts']} multi-claim passages | within-passage AUROC \"\n"
         "          f\"{loc['within_passage_auroc']:.3f}\")\n"
         "else:\n"
         "    print('within-passage localization not measured here (one claim per question) — see smoke test')\n"
         "print(f\"Calibrated thresholds: T_MED={R['t_med']}  T_HIGH={R['t_high']}\")"),
    md("### 5 · Smoke-test the product — **GPU** — Option-B heads, per-sentence flags\n"
       "Loads the `_sentence_<TAG>` heads (`sentence_tag=TAG` → sentence-regime generation, train==inference) "
       "and the per-claim fusion, then flags each claim sentence of a fresh answer. Obscure entities elicit "
       "more hallucinations."),
    code(SETUP +
         "import torch, gc, json\n"
         "from pipeline import HallKingPipeline\n"
         "from fusion import FusionModel\n"
         "from localize import localize\n"
         "gc.collect(); torch.cuda.empty_cache()\n"
         "pipe = HallKingPipeline(dataset='triviaqa', separate_tsv=False, sentence_tag=TAG).load()\n"
         "pipe.fusion = FusionModel.load(os.path.join('..','models',f'fusion_claim_{TAG}.pkl'))\n"
         "THR = json.load(open(os.path.join('..','models',f'fusion_claim_{TAG}_thresholds.json')))\n"
         "for Q in ['Who painted the Mona Lisa?', 'What is the capital of Australia?',\n"
         "          'Tell me about the Sigiriya rock fortress.']:\n"
         "    res = localize(pipe, Q, max_new_tokens=80, use_claim_filter=True)\n"
         "    print('Q:', Q); print('A:', res['answer'])\n"
         "    for s in res['sentences']:\n"
         "        if s['fused'] is None:\n"
         "            tg='filler   '; fv='   '\n"
         "        else:\n"
         "            fv=f\"{s['fused']:.2f}\"\n"
         "            tg = 'HALLUC   ' if s['fused']>=THR['t_high'] else ('UNCERTAIN' if s['fused']>=THR['t_med'] else 'ok       ')\n"
         "        print(f\"   [{tg} {fv}] {s['sentence']}\")\n"
         "    print()"),
    md("### Wire it into the live demo (after the numbers check out)\n"
       "The pipeline already supports Option B: `HallKingPipeline(..., sentence_tag='<TAG>')` loads the "
       "`_sentence_<TAG>` heads and generates in the sentence regime. To switch the web demo, set "
       "`SENTENCE_TAG = '<TAG>'` in `backend/app.py` (it loads `models/fusion_claim_<TAG>.pkl` + thresholds). "
       "Until the numbers are good, the backend stays on the Option-A short-QA path."),
])

print("\nAll notebooks built.")
