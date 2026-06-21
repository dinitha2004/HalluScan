"""Build notebooks/12_eat_experiment.ipynb with nbformat. Re-run to regenerate.

Exact Answer Token (EAT) experiment — the VALIDATION that gates the demo highlight (Phase 3).
Two staged gates, mirroring the plan:

  * Phase 2a (cheap, no retrain) — extraction quality. Does the LLM self-extractor return a usable
    verbatim answer span, and does it match the gold answer? If poor, STOP (the idea is unsound here).
  * Phase 2b (GPU, gated behind RUN_2B) — anchor comparison. Re-anchor SEP / HalluShift / TSV at the
    EAT token (vs the current end-of-sequence anchor), retrain the two probe heads, and compare
    held-out AUROC + the answer-length correlation (the documented length confound). This tests the
    paper's claim that probing AT the answer token detects errors better.

Reuses hallking/eat.py (unit-tested mapping), tools/eat_eval.py (2a), retrain.py (head re-fits), and
the frozen adapters' arbitrary-token methods. Heavy passes are the user's GPU runs; this file only
builds the notebook. Datasets: triviaqa + squad (web_questions gold unusable — docs/head_audit.md).

Run:  python tools/build_nb12.py
"""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks")
os.makedirs(NB, exist_ok=True)


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


cells = []

cells.append(md(
    "# 12 · Exact Answer Token (EAT) experiment — validate, *then* (separately) ship\n\n"
    "**Idea.** Locate the verbatim span that actually answers the question (\"Paris\", \"1969\") and, for a "
    "flagged answer, point at *that* span. From Orgad et al., *LLMs Know More Than They Show* (ICLR 2025): "
    "probing the hidden state **at the exact answer token** detects errors far better than at the last token.\n\n"
    "**Why it matters here.** Every HallKing detector currently anchors at an *end-of-sequence* token "
    "(SEP=second-last, TSV=last, HalluShift=whole range). That end-anchor is exactly the suspect behind two "
    "documented weaknesses: the weak SEP/HalluShift heads and the **length confound** (heads ride answer "
    "length — see `docs/head_audit.md`). Re-anchoring at the EAT is a new lever for both.\n\n"
    "**Gate (do NOT skip).** The demo highlight (Phase 3) is only built *after* this notebook verifies. "
    "**2a** (extraction quality) is the first gate; **2b** (anchor vs AUROC) is the real test. Run in "
    "se_probes_env."
))

# ---- setup ----
cells.append(code(
    "import os, sys, json\n"
    "os.environ.setdefault('HF_HOME', r'D:/LLAMA CACHE/huggingface')\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..', 'hallking')))\n"
    "sys.path.insert(0, os.path.abspath(os.path.join('..', 'tools')))\n"
    "import warnings; warnings.filterwarnings('ignore')\n"
    "import numpy as np, pandas as pd, torch\n"
    "from types import SimpleNamespace\n"
    "from scipy.stats import spearmanr\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.metrics import roc_auc_score\n"
    "import eat, eat_eval, retrain, metrics as M\n"
    "ROOT = os.path.abspath('..'); DATA = os.path.join(ROOT, 'data'); os.makedirs(DATA, exist_ok=True)\n"
    "SEED = 42\n"
    "print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
))

# ====================================================================== PHASE 2a
cells.append(md(
    "## Phase 2a · Extraction quality — the first gate (cheap, no retrain)\n"
    "Generate a single-sentence answer per question, run the LLM self-extractor, and measure: extraction "
    "rate, EAT==gold (exact match), EAT↔gold (contains), and the failure breakdown. Per-row output is saved "
    "to `data/eat_eval_<ds>_n<N>.jsonl`.\n\n"
    "**Read the gate like this:** a high *extraction rate* with a high *EAT↔gold among correct answers* "
    "means the extractor works. If extraction rate is low, or it frequently picks the wrong span when the "
    "answer is right (`wrong span` failures), STOP and fix the extractor before 2b."
))
cells.append(code(
    "DS_2A = ['triviaqa', 'squad']   # web_questions excluded (gold unusable)\n"
    "N_2A  = 200                     # questions per dataset (GPU; a few minutes each)\n"
    "rows_2a = {}\n"
    "for ds in DS_2A:\n"
    "    rows_2a[ds] = eat_eval.evaluate(dataset=ds, n=N_2A, save=True, verbose=True)"
))
cells.append(md(
    "### 2a · Inspect examples + failure cases\n"
    "Eyeball that the extracted span is the actual answer phrase, and look at where it goes wrong "
    "(model paraphrased / no clean answer / span landed on the wrong token)."
))
cells.append(code(
    "def show(ds, k=12):\n"
    "    rs = rows_2a[ds]\n"
    "    print(f'=== {ds}: {k} examples ===')\n"
    "    for r in rs[:k]:\n"
    "        tag = 'OK ' if r['eat_contains_gold'] else ('x  ' if r['extracted'] else '-- ')\n"
    "        print(f\"  {tag} EAT={str(r['eat'])[:30]!r:32} gold0={str(r['refs'][0])[:24]!r:26} | A={r['answer'][:60]!r}\")\n"
    "    bad = [r for r in rs if r['answer_contains_gold'] and not r['eat_contains_gold']]\n"
    "    print(f'\\n  -- {len(bad)} WRONG-SPAN cases (answer was correct but EAT missed it) --')\n"
    "    for r in bad[:8]:\n"
    "        print(f\"     EAT={str(r['eat'])[:30]!r:32} raw={str(r['raw_eat'])[:30]!r:32} | A={r['answer'][:55]!r}\")\n"
    "for ds in DS_2A:\n"
    "    show(ds); print()"
))
cells.append(md(
    "### 2a · GATE decision\n"
    "Rough bar to proceed to 2b: extraction rate ≳ 85%, and among answers that contain the gold, EAT↔gold "
    "≳ 85% (few wrong-span failures). Note the numbers here in `docs/eat_audit.md`. **If this fails, stop — "
    "do not run 2b and do not touch the demo.**"
))

# ====================================================================== PHASE 2b
cells.append(md(
    "## Phase 2b · Anchor comparison — the real test (GPU, gated)\n"
    "> ⚠️ **Untested locally** (no GPU in the build env). Run only after 2a passes. Set `RUN_2B=True`.\n\n"
    "One Instruct (fp16) pass caches, per question, **both** anchorings of every head plus a hallucination "
    "label, then we retrain the two probe heads on each anchoring and compare:\n"
    "- **SEP** — all-layer hidden stack at the **EAT token** vs the **second-last** token (`_slt_stack`).\n"
    "- **HalluShift** — 71-d features over the **EAT step window** vs the **whole** range (`_features_from_gen`).\n"
    "- **TSV** — frozen centroid margin read at the **EAT token** vs the **last** token (`score_sentences`).\n\n"
    "We report held-out AUROC (end vs EAT) and each head's Spearman correlation with answer length (the "
    "confound — lower |ρ| is better). Labels via the same hybrid judge used by `retrain.gen_and_cache`."
))
cells.append(code(
    "RUN_2B   = False                # <-- flip to True on the GPU box, after 2a passes\n"
    "DS_2B    = 'triviaqa'           # cache+retrain dataset\n"
    "N_2B     = 400\n"
    "OFFSET_2B = 0\n"
    "MAXTOK_2B = 64\n"
    "CACHE_2B = os.path.join(DATA, f'eat_anchor_{DS_2B}_n{N_2B}.npz')\n"
    "print('RUN_2B =', RUN_2B, '| cache ->', os.path.relpath(CACHE_2B, ROOT))"
))
cells.append(md(
    "### 2b · Cache both anchorings (GPU). \n"
    "Reuses `eat.locate_eat_span` + `eat.eat_token_range` (unit-tested) to find the EAT token, then reads "
    "each head at the EAT anchor and at its usual end anchor from the SAME generation. SEP vectors are "
    "stored fp16 (135168-d × N × 2)."
))
cells.append(code(
    "def _eat_step_window(eng, gen, answer, eat_text, min_steps=2):\n"
    "    '''Return (eat_abs_idx, (a_step, b_step)) for HalluShift's step-range view, or None.'''\n"
    "    span = eat.locate_eat_span(answer, eat_text) if eat_text else None\n"
    "    if span is None:\n"
    "        return None\n"
    "    t0, t1 = eat.eat_token_range(eng, gen, span[0], span[1])      # absolute token indices\n"
    "    plen = gen['prompt_len']\n"
    "    a, b = t0 - plen, t1 - plen                                   # -> generation-step indices\n"
    "    while b - a < min_steps:                                      # widen so HalluShift doesn't fall back\n"
    "        a = max(0, a - 1); b = b + 1\n"
    "    return t1 - 1, (max(0, a), b)                                 # anchor = last EAT token\n"
    "\n"
    "def cache_2b():\n"
    "    from sep_adapter import SEPAdapter\n"
    "    from hallushift_adapter import HalluShiftAdapter\n"
    "    from run_dataset import load_qa, INSTRUCT_MODEL\n"
    "    from engine import HallKingEngine\n"
    "    from claim_label import label_hybrid\n"
    "    qs, refs = load_qa(DS_2B, n=N_2B, offset=OFFSET_2B)\n"
    "    eng = HallKingEngine(model_name=INSTRUCT_MODEL, fp16_nonquant=True).load()\n"
    "    sep = SEPAdapter(eng); hs = HalluShiftAdapter(eng); hs.num_layers = eng.num_layers\n"
    "    keep_q, keep_r, answers = [], [], []\n"
    "    sep_end, sep_eat, hs_end, hs_eat, ans_len, has_eat, eat_end = [], [], [], [], [], [], []\n"
    "    from tqdm.auto import tqdm\n"
    "    for q, rf in tqdm(list(zip(qs, refs)), desc=f'EAT cache ({DS_2B})', unit='q'):\n"
    "        gen = eng.generate_sentence(q, max_new_tokens=MAXTOK_2B)\n"
    "        ans = gen['answer_full'].strip()\n"
    "        if not ans:\n"
    "            continue\n"
    "        eat_txt = eat.extract_eat_text(eng, q, ans)\n"
    "        sp_eat = eat.locate_eat_span(ans, eat_txt) if eat_txt else None\n"
    "        win = _eat_step_window(eng, gen, ans, eat_txt)\n"
    "        seq = gen['sequences']; plen = gen['prompt_len']\n"
    "        slt = max(seq.shape[1] - 2, plen)\n"
    "        H = eng.forward_hidden_states(seq)                        # one clean forward (SEP)\n"
    "        e_idx = win[0] if win else slt                            # fall back to SLT when no EAT\n"
    "        sep_end.append(sep._slt_stack(H, slt).astype(np.float16).reshape(-1))\n"
    "        sep_eat.append(sep._slt_stack(H, e_idx).astype(np.float16).reshape(-1))\n"
    "        hs_end.append(np.asarray(hs.features(gen), np.float64))\n"
    "        if win:\n"
    "            a, b = win[1]; go = gen['gen_output']\n"
    "            view = SimpleNamespace(hidden_states=go.hidden_states[a:b],\n"
    "                                   attentions=go.attentions[a:b], logits=go.logits[a:b])\n"
    "            try:\n"
    "                hs_eat.append(np.asarray(hs._features_from_gen(view), np.float64))\n"
    "            except Exception:\n"
    "                hs_eat.append(np.asarray(hs.features(gen), np.float64))\n"
    "        else:\n"
    "            hs_eat.append(np.asarray(hs.features(gen), np.float64))\n"
    "        keep_q.append(q); keep_r.append(rf); answers.append(ans)\n"
    "        ans_len.append(len(ans)); has_eat.append(win is not None)\n"
    "        eat_end.append(sp_eat[1] if sp_eat else -1)   # EAT end char offset (reused by the TSV cell)\n"
    "    print(f'cached {len(answers)} rows | EAT found for {sum(has_eat)} ({100*np.mean(has_eat):.0f}%)')\n"
    "    labels, _ = label_hybrid(keep_q, answers, keep_r, eng, verbose=True)\n"
    "    eng.unload()\n"
    "    y = np.asarray(labels).astype(int)\n"
    "    np.savez_compressed(CACHE_2B, y=y, ans_len=np.asarray(ans_len), has_eat=np.asarray(has_eat),\n"
    "                        eat_end=np.asarray(eat_end),\n"
    "                        sep_end=np.stack(sep_end), sep_eat=np.stack(sep_eat),\n"
    "                        hs_end=np.stack(hs_end), hs_eat=np.stack(hs_eat),\n"
    "                        question=np.array(keep_q, dtype=object), answer=np.array(answers, dtype=object))\n"
    "    print('saved', os.path.relpath(CACHE_2B, ROOT))\n"
    "\n"
    "if RUN_2B and not os.path.exists(CACHE_2B):\n"
    "    cache_2b()\n"
    "else:\n"
    "    print('skip cache:', 'RUN_2B=False' if not RUN_2B else 'cache exists')"
))
cells.append(md(
    "### 2b · Retrain the probe heads on each anchor + compare AUROC\n"
    "Shared 75/25 split. SEP probes via `retrain.retrain_sep`; HalluShift via `retrain.retrain_hallushift`. "
    "Each head trained twice (end-anchored vs EAT-anchored), evaluated on the same held-out rows."
))
cells.append(code(
    "def auroc(y, s): return roc_auc_score(y, s) if len(set(y)) > 1 else float('nan')\n"
    "if RUN_2B:\n"
    "    Z = np.load(CACHE_2B, allow_pickle=True)\n"
    "    y, ans_len = Z['y'], Z['ans_len']\n"
    "    tr, te = train_test_split(np.arange(len(y)), test_size=0.25, stratify=y, random_state=SEED)\n"
    "    def sep_auc(feats):\n"
    "        probe = retrain.retrain_sep(feats[tr].astype(np.float32), y[tr])[0]\n"
    "        Xb = feats.astype(np.float32)\n"
    "        s = probe['s_bmodel'].predict_proba(Xb[te])[:, 1]\n"
    "        return auroc(y[te], s), s\n"
    "    sep_end_auc, sep_end_s = sep_auc(Z['sep_end'])\n"
    "    sep_eat_auc, sep_eat_s = sep_auc(Z['sep_eat'])\n"
    "    def hs_auc(feats):\n"
    "        df = pd.DataFrame({f'hs_feat_{j:02d}': feats[:, j] for j in range(feats.shape[1])})\n"
    "        state, scaler = retrain.retrain_hallushift(df.iloc[tr], y[tr])\n"
    "        from classifier import CombinedNN\n"
    "        m = CombinedNN(32); m.load_state_dict(state); m.eval()\n"
    "        with torch.no_grad():\n"
    "            s = torch.sigmoid(m(torch.tensor(scaler.transform(feats), dtype=torch.float32))).numpy().ravel()\n"
    "        return auroc(y[te], s[te]), s[te]\n"
    "    hs_end_auc, _ = hs_auc(Z['hs_end'])\n"
    "    hs_eat_auc, _ = hs_auc(Z['hs_eat'])\n"
    "    print(f'{\"head\":12} {\"END anchor\":>11} {\"EAT anchor\":>11}')\n"
    "    print(f'{\"SEP\":12} {sep_end_auc:11.3f} {sep_eat_auc:11.3f}')\n"
    "    print(f'{\"HalluShift\":12} {hs_end_auc:11.3f} {hs_eat_auc:11.3f}')\n"
    "else:\n"
    "    print('RUN_2B=False')"
))
cells.append(md(
    "### 2b · TSV (frozen) end vs EAT read\n"
    "No retrain — just move the read position. `score_qa` reads the last token; `score_sentences(.,.,[eat_end])` "
    "reads the EAT token of the steered forward. Uses the deployed sentence TSV checkpoint."
))
cells.append(code(
    "if RUN_2B:\n"
    "    from engine import HallKingEngine\n"
    "    from run_dataset import INSTRUCT_MODEL\n"
    "    from tsv_adapter import TSVAdapter\n"
    "    Z = np.load(CACHE_2B, allow_pickle=True); qs = Z['question']; ans = Z['answer']; ee = Z['eat_end']\n"
    "    ckpt = os.path.join(ROOT, 'artifacts', 'tsv', 'best_checkpoint_sentence_s1.pt')\n"
    "    eng = HallKingEngine(model_name=INSTRUCT_MODEL, fp16_nonquant=True).load()\n"
    "    tsv = TSVAdapter(eng, ckpt_path=ckpt).load()\n"
    "    tsv_end, tsv_eat = [], []\n"
    "    from tqdm.auto import tqdm\n"
    "    for i, (q, a) in enumerate(tqdm(list(zip(qs, ans)), desc='TSV end/EAT', unit='q')):\n"
    "        m_end = tsv.score_qa(q, a)['tsv_margin']   # last-token read\n"
    "        tsv_end.append(m_end)\n"
    "        # reuse the cached EAT span (same span SEP/HalluShift anchored on); fall back to last token\n"
    "        tsv_eat.append(tsv.score_sentences(q, a, [int(ee[i])])[0] if ee[i] >= 0 else m_end)\n"
    "    eng.unload()\n"
    "    print(f'TSV       END={auroc(y[te], np.asarray(tsv_end)[te]):.3f}  EAT={auroc(y[te], np.asarray(tsv_eat)[te]):.3f}')\n"
    "else:\n"
    "    print('RUN_2B=False')"
))
cells.append(md(
    "### 2b · Length-confound diagnostic\n"
    "Spearman ρ between each head's score and answer length, end vs EAT. The hypothesis: anchoring at the "
    "answer token (not the sequence end, whose hidden state drifts with position/length) **shrinks |ρ|**. "
    "Compare against `docs/head_audit.md`'s end-anchored numbers."
))
cells.append(code(
    "if RUN_2B:\n"
    "    def rho(s): return spearmanr(s, ans_len[te]).correlation\n"
    "    print('Spearman |rho| with answer length (lower is better):')\n"
    "    print(f'  SEP        end={rho(sep_end_s):+.3f}  EAT={rho(sep_eat_s):+.3f}')\n"
    "    # (HalluShift/TSV: recompute full-vector scores above if you want their rho too)\n"
    "else:\n"
    "    print('RUN_2B=False')"
))
cells.append(md(
    "## Verdict → `docs/eat_audit.md`\n"
    "Fill the table with the 2a gate numbers and the 2b end-vs-EAT AUROC + length-ρ. **Demo (Phase 3) "
    "proceeds only if** 2a passed AND EAT anchoring helps (or at least matches without misleading). If EAT "
    "≈ END on AUROC but the highlight is still wanted as pure UX, that is allowed — but it must be labelled "
    "'exact answer phrase', not 'the hallucinated token' (the flag stays sentence-level)."
))

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "hallking", "display_name": "HallKing (se_probes_env)",
                             "language": "python"}
path = os.path.join(NB, "12_eat_experiment.ipynb")
with open(path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", path)
