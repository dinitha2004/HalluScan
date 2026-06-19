# Head audit — why the cross-dataset AUROCs collapsed

Diagnosis-only pass (no retraining). Triggered by `notebooks/5_cross_dataset_eval50questions_1.ipynb`
showing surprisingly low AUROCs. Question: is it **labelling**, **weak heads**, or **bad training**?
Short answer: **mostly none of those on triviaqa/squad — it's a length/regime confound plus one
unusable dataset (web_questions). The labels on triviaqa & squad are basically fine.**

All numbers below are CPU-recomputed from the saved parquets and artifacts. No GPU steps were run
(notebook 5b was training during this audit).

---

## 0. The displayed numbers were stale — ignore them

The notebook *cell source* says `DATASETS=['triviaqa','squad','web_questions']`, `N=200`, but the
*shown output* is `nq_open/squad/triviaqa @ n=50` with TSV ≈0.77–0.90. Those don't match — the output
is from an older run and was never re-executed after the config changed. The **real** current state is
the saved `data/<ds>_cross_eval_llm_judge.parquet` files (n≈200), recomputed with `metrics.py`:

| head set | dataset | halluc% | SEP | HalluShift | TSV | FUSED |
|---|---|--:|--:|--:|--:|--:|
| Type-1 (short) | triviaqa (held-out, off 3000) | 19.5% | 0.541 | 0.676 | 0.626 | 0.616 |
| Type-1 (short) | squad | 41.7% | 0.675 | 0.670 | 0.627 | 0.599 |
| Type-1 (short) | web_questions | 15.1% | 0.665 | **0.488** | **0.439** | **0.416** |
| Type-2 (sentence s1) | triviaqa (n=20) | 30.0% | 0.762 | 0.333 | 0.750 | 0.750 |
| Type-2 (sentence s1) | squad (n=139) | 33.1% | 0.732 | 0.540 | 0.691 | 0.702 |

The web_questions row (TSV/FUSED **below chance**) is what looked alarming.

---

## 1. The heads have real signal *in-distribution* — so training isn't broken

On the heads' **own** held-out test split (sentence regime, `diagnose_heads.py`):

| labels | SEP | HalluShift | TSV |
|---|--:|--:|--:|
| `s1` (substring) | 0.717 | 0.653 | **0.817** |
| `s1j` (judge-relabelled) | 0.680 | 0.648 | **0.804** |

Type-1 TSV trained AUROC = **0.873** (`artifacts/tsv/best_checkpoint_retrained.pt`). So TSV genuinely
separates at ~0.80–0.87 on its training distribution; SEP ~0.68–0.72; HalluShift ~0.65. **The training
process is not the primary problem.** (HalluShift is the weak one — see §4.)

---

## 2. The real culprit: the heads ride **answer length**, not truth

`corr(head_score, answer_length)` vs `corr(answer_length, hallucination)` on the Type-1 cross-eval:

| dataset | corr(len, **halluc**) | corr(SEP,len) | corr(HalluShift,len) | corr(TSV,len) |
|---|--:|--:|--:|--:|
| triviaqa | **+0.035** | +0.161 | +0.578 | +0.540 |
| squad | **+0.059** | +0.235 | +0.479 | +0.603 |
| web_questions | **−0.080** | +0.245 | +0.723 | +0.637 |

TSV and HalluShift correlate **0.5–0.7 with answer length**, but length's correlation with actual
hallucination is ~0 **and flips sign across datasets**. So the heads are effectively length detectors:

- triviaqa/squad: longer ≈ very slightly more hallucinated → length-driven heads land just above chance (~0.63).
- web_questions: longer ≈ slightly more *truthful* → the same heads **invert** → AUROC < 0.5.

**Mechanism confirmed** by restricting to short (training-like) answers — the inversion disappears:

| dataset | TSV full | TSV (len≤40) |
|---|--:|--:|
| web_questions | 0.439 | **0.534** |
| triviaqa | 0.626 | 0.637 |
| squad | 0.627 | 0.659 |

Why length leaks in: the heads were trained on uniformly short (~2-word) TriviaQA answers, so length had
no variance to key on. At eval time the Instruct model rambles into long, truncated paragraphs on the
messier datasets (web_questions max 387 chars), and the TSV last-token representation + HalluShift
internal-dynamics features both scale with sequence length. **Type-2 sentence heads show a weaker
confound (TSV-len corr 0.36–0.42 vs 0.54–0.64) and no inversion** — regime-matching is the fix direction,
and it's already where the deployed heads went.

---

## 3. web_questions is unusable — drop it (the nq_open problem again)

The substring rule fails on **86/199** web_questions rows and the LLM-judge "rescued" **56** of them to
truthful — because the **Freebase gold answers are themselves wrong/irrelevant**:

- *"who plays ken barlow in coronation street?"* gold = **"Tony Warren"** (the show's *creator*, not the
  actor). Answer "William Roache" is **correct**; gold is wrong.
- *"who did mozart write his four horn concertos for?"* gold = **"…story by pierre beaumarchais"**
  (irrelevant). Answer "Ignaz Leutgeb" is correct.
- *"what happened after mr. sugihara died?"* gold = **"Yaotsu"**; the answer fabricates a museum in
  Kaohsiung, Taiwan and was still **rescued to truthful** (a false negative).

Plus opinion/temporal/ill-posed questions ("best Sandals resort in St Lucia", "what to do today in
Atlanta", "my timezone in Louisiana", "minority leader *now*"). With 15% base rate, this noise swings
AUROC wildly. **web_questions should be excluded, for the same reason nq_open was.** Its numbers are not
evidence of head quality either way.

**By contrast, triviaqa & squad labels are sound.** The judge's rescues there are correct paraphrase /
alias / number-word fixes ("The Patriots" → New England Patriots; "4 teams" → "four"; 1995 Clause IV →
"public ownership of the means of production" = Nationalization). A few squad rescues are dubious (a
fabricated "2005" date rescued as correct), but the labelling is not what's dragging triviaqa/squad down.

---

## 4. Secondary issues

- **Fusion is pathologically TSV-dominated.** Type-1 fusion logreg coefs = `SEP −0.30, HalluShift +0.17,
  TSV **+3.41**` → FUSED ≈ TSV, so it inherits and *amplifies* the inversion on web_questions (FUSED
  0.416, the worst of all). The Type-2 fusion is far healthier (`0.10 / 0.21 / 0.89`).
- **HalluShift (Type-2 sentence) is *broken by training*, not dead — see §6 for the proven cause + fix.**
  The deployed head outputs a near-constant value (demo std 0.088); on transfer it flags everything. But
  the features carry real signal (logreg 0.79); a fixed retrain recovers it.
- **Residual label noise even in `s1j`.** A handful of true sentences are still in the hallucinated class
  ("stereo records … 1950s", "Stansfield Turner … 1977-81"). Minor; depresses measured AUROC a little.

---

## 5. Honest cross-dataset number (drop web_questions)

Pooled triviaqa+squad, Type-1 (n=399, 30.6% halluc): SEP **0.648**, HalluShift **0.688**, TSV **0.622**,
FUSED **0.601**. That is the real state: **weak but above chance**, dominated by a length proxy, with TSV
no longer the best single head once length variance is in play.

---

## Verdict on the three hypotheses

1. **"Labelling is wrong"** — partly. **True for web_questions** (bad Freebase gold + opinion/temporal
   Qs → drop it). **False for triviaqa/squad** — labels there are sound.
2. **"Heads are weak"** — partly. They have genuine in-distribution signal (TSV 0.80, trained 0.87) but
   are **length-confounded**, so they degrade to ~0.62–0.69 on variable-length cross-dataset answers and
   *invert* when length anti-correlates with truth. Type-2 HalluShift specifically was **broken by a
   training/preprocessing bug** (§6), not weak — its features carry ~0.79 AUROC.
3. **"Training is bad"** — no, not the training *procedure*. The flaw is **regime/length mismatch**:
   heads trained on uniformly short answers, evaluated on long truncated paragraphs.

## Recommendation (out of scope to execute here)
- **Drop web_questions** from the cross-eval (keep triviaqa held-out + squad; consider truthfulqa via nb6).
- **Fix the regime mismatch**, not the training algorithm: evaluate (and, if retraining, train) on a
  length-controlled regime — the Type-2 sentence heads already do this and show no inversion. Either
  length-match the eval answers or move the cross-eval onto the sentence heads.
- **Re-fit the Type-1 fusion** or stop using it — coef 3.41 on TSV makes it a TSV passthrough that
  amplifies the worst failure.
- **Consider a length-decorrelated TSV** (e.g. control for sequence length) since length is the dominant
  nuisance variable.

### Cheap GPU confirmations (run when 5b is done — user's machine)
- `tools/relabel_cross_eval.py` — re-judge cached scores, print per-head AUROC old(bleurt) vs new(judge)
  to confirm SEP isn't just tracking label noise. (No generation; reuses cached scores.)
- These were **not** run during this audit to avoid throttling the in-progress 5b GPU job.

---

## 6. Type-2 HalluShift collapse — PROVEN cause + fix (notebook 10)

**What trained it:** TriviaQA only (`data/claims_sent_s1.parquet`, 500 rows, offset 0), on the **noisy
substring labels** (`s1`; the judge later found ~6.8% were true sentences wrongly tagged hallucinated).
The corrected `s1j` labels were never trained on. So squad/web_questions are pure transfer.

**Why it collapses on transfer (two compounding defects):**
- **A. StandardScaler landmine.** In the sentence regime ~45/71 HalluShift divergence/cosine features are
  near-constant (std≈0), so the scaler stored `scale_` down to **1.6e-13**. On any distribution shift those
  features deviate from the training mean and `(x−mean)/1.6e-13` explodes to ~1e8–1e12, saturating the
  LayerNorm `CombinedNN` → constant output. *Demonstrated*: perturbing only the dead columns by **+1e-4**
  flatlines the deployed head (std 0.11→0.00, AUROC 0.65→0.50).
- **B. Underfit MLP.** In-dist held-out the deployed `CombinedNN` reaches 0.653 (flat) while a plain logreg
  on the same features+split hits **0.789** (full-batch GD, no minibatching, coarse early stop).

**The fix (`notebooks/10_retrain_hallushift.ipynb`, CPU + one GPU transfer pass):** neutralise the dead
features (`scale_=1.0` where train `var_<1e-12`), retrain on the **`s1j`** labels with proper minibatched
training. Results:

| | OLD CombinedNN (s1) | NEW CombinedNN (s1j, fixed) | NEW LogReg (s1j) |
|---|--:|--:|--:|
| in-dist held-out AUROC | 0.618 | **0.766** | 0.761 |
| output std (flat→spread) | 0.11 | 0.30 | 0.31 |
| **squad TRANSFER AUROC** | **0.548** | **0.690** | 0.680 |
| squad transfer confusion | [[0,120],[0,68]] (flags all) | [[71,49],[16,52]] | [[66,54],[15,53]] |
| +1e-4 perturbation | collapses to 0.50 | **stable** | stable |

Fixed `CombinedNN` wins. **HalluShift is fixable, not signal-dead** — this revises the earlier
"intrinsically weak, drop it" conclusion. The new artifacts are not deployed yet (notebook save cell is
gated `SAVE=False`); to wire them into the pipeline/cross-eval the inference path must apply the same
neutralised scaler (kept-feature list saved alongside). Transfer set cached at
`data/claims_sent_squad_transfer.parquet`.
