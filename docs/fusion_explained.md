# How the fused hallucination detector works (HalluScan)

A plain-English + math walkthrough of how HalluScan turns one model answer into a single
**hallucination probability**, for both deployed models (the 8B and the 1B). Uses the *actual*
weights shipped in `models/fusion_claim_s1.pkl` (8B) and `models/fusion_claim_l1b.pkl` (1B).

---

## 1. The big picture

```
                          ┌── SEP head        → sep_entropy   (0..1, higher = worse)
   answer / sentence ───► ├── HalluShift head → hallushift    (0..1, higher = worse)
                          └── TSV head        → tsv_margin    (~ -0.3..0.3, higher = worse)
                                   │
                                   ▼
                          FUSION (tiny logistic regression)
                                   │
                                   ▼
                       fused = P(hallucination)  ∈ [0, 1]
                                   │
                                   ▼
                   thresholds → tier:  Reliable / Uncertain / Likely Hallucinated
```

Three independent "heads" each look at the frozen LLM while it answers and emit **one number**.
A **fusion** model combines those numbers into a single calibrated probability. Two thresholds
turn that probability into a coloured risk label. For a long answer this is done **per sentence**,
then the sentence scores are aggregated into one headline (see §6).

The LLM itself is never fine-tuned — only the small heads + fusion are trained. They are retrained
**per model** because they read that model's internal activations (different layer count / hidden
size), so the 8B's heads can't score the 1B and vice-versa.

---

## 2. The three detector scores (the fusion's inputs)

All three are oriented the same way: **higher = more likely hallucinated.**

| Score | What it measures | How it's produced | Typical range |
|-------|------------------|-------------------|---------------|
| `sep_entropy` | Does the hidden state look "uncertain/made-up"? | A logistic **probe** reads the hidden states of **every layer** at the second-to-last token, flattened into one big vector, and outputs P(hallucinated). | 0..1 |
| `hallushift` | Do the activations **shift** abnormally across layers while decoding? | Wasserstein distance + cosine similarity between consecutive layer pairs, plus token-probability statistics (entropy, low-prob counts, gradients, percentiles) → standardized → a small MLP (`CombinedNN`) → sigmoid. | 0..1 |
| `tsv_margin` | Is the answer's representation closer to a "hallucinated" or "truthful" direction? | A trained **steering vector** is injected at one mid layer; the last-token representation is compared (cosine) to two learned centroids. `tsv_margin = cos(·, halluc_centroid) − cos(·, truth_centroid)`. | ~ −0.3 .. 0.3 |

Code: `hallking/sep_adapter.py`, `hallking/hallushift_adapter.py`, `hallking/tsv_adapter.py`.
(`sep_accuracy`, the probe's P(truthful), is shown in the UI but is **not** a fusion input.)

---

## 3. The fusion math (logistic regression)

The deployed fusion is a **logistic regression** over the selected head scores. For each input
feature it does three things — **standardize, weight-and-sum, squash**:

```
For each feature i:   z_i = (x_i − mean_i) / scale_i        # StandardScaler (zero-mean, unit-variance)
Combine:              logit = Σ_i (coef_i · z_i) + intercept
Probability:          fused = sigmoid(logit) = 1 / (1 + e^(−logit))   # P(hallucinated)
```

- **`mean_i` / `scale_i`** come from the training data — they put every detector on a comparable
  scale before weighting (TSV's ±0.3 margin and SEP's 0..1 probability would otherwise be
  incomparable).
- **`coef_i`** is the learned weight = how much that detector moves the decision. A bigger
  magnitude ⇒ that head dominates the fused score.
- **`intercept`** is the bias (the baseline log-odds when every standardized input is 0).
- Trained with `C=0.5` (moderate L2 regularization) and `class_weight="balanced"`.

> Implementation note: HalluScan computes this sigmoid **by hand** from the stored
> `coef_`/`intercept_`/scaler stats rather than calling sklearn's `predict_proba`, so the saved
> fusion keeps working across sklearn versions. See `FusionModel._proba1` in
> [`hallking/fusion.py`](../hallking/fusion.py).

The `.pkl` stores `{kind, feature_cols, C, scaler, clf}`. `feature_cols` decides **which** of the
three heads actually drive the fused score — and this is where the two models differ.

---

## 4. The actual deployed weights

### 4a. The 8B model — `fusion_claim_s1.pkl` → **TSV only**

```
kind         : logreg
feature_cols : ['tsv_margin']          ← only ONE head feeds the fused score
scaler.mean_ : [-0.0899]
scaler.scale_: [ 0.1251]
coef_        : [ 1.8473]
intercept_   : [-0.3196]
thresholds   : t_med = 0.70,  t_high = 0.90
```

On the 8B, SEP and HalluShift were found to be weak/flat (≈0.62 / ≈0.65 AUROC, see
`docs/head_audit.md`), so the fusion was deliberately fit on **TSV alone**. The other two scores
are still shown per sentence in the UI, but they do **not** change the fused number.

**Formula:**  `fused = sigmoid( 1.8473 · (tsv_margin + 0.0899)/0.1251 − 0.3196 )`

**Worked examples (8B):**

| `tsv_margin` | standardized z | logit | fused | tier |
|---|---|---|---|---|
| −0.09 (≈ average, truthful-leaning) | ≈ 0.0 | −0.32 | **0.42** | Reliable |
| 0.00 | 0.72 | 1.01 | **0.73** | Uncertain |
| +0.10 (hallucinated-leaning) | 1.52 | 2.48 | **0.92** | Likely Hallucinated |

So for the 8B, the whole fused score is essentially a calibrated, thresholded version of the TSV
margin.

### 4b. The 1B model — `fusion_claim_l1b.pkl` → **all three heads (SEP-led)**

```
kind         : logreg
feature_cols : ['sep_entropy', 'hallushift', 'tsv_margin']   ← all THREE feed the score
scaler.mean_ : [0.4768, 0.4885,  0.0148]
scaler.scale_: [0.4671, 0.2461,  0.1026]
coef_        : [4.5894, 0.2780,  0.2890]      ← SEP dominates (4.59 ≫ 0.28, 0.29)
intercept_   : [-0.2442]
thresholds   : t_med = 0.155, t_high = 0.258
```

On the 1B all three heads separate truthful from hallucinated (held-out AUROC: SEP 0.741,
HalluShift 0.793, TSV 0.727), so the fusion keeps all three. The logistic fit then leaned heavily
on **SEP** (coefficient 4.59 vs ~0.28), so the 1B's fused score is mostly SEP with small nudges
from HalluShift and TSV.

**Formula:**
```
z_sep = (sep_entropy − 0.4768)/0.4671
z_hs  = (hallushift  − 0.4885)/0.2461
z_tsv = (tsv_margin  − 0.0148)/0.1026
fused = sigmoid( 4.5894·z_sep + 0.2780·z_hs + 0.2890·z_tsv − 0.2442 )
```

**Worked example (1B)** — the real "What is the capital of France?" answer from the notebook run
(`sep=0.000, hs=0.154, tsv=−0.167`):

```
z_sep = (0.000 − 0.4768)/0.4671 = −1.021   →  term  4.5894·(−1.021) = −4.69
z_hs  = (0.154 − 0.4885)/0.2461 = −1.359   →  term  0.2780·(−1.359) = −0.38
z_tsv = (−0.167 − 0.0148)/0.1026 = −1.772  →  term  0.2890·(−1.772) = −0.51
logit = −4.69 − 0.38 − 0.51 − 0.2442 = −5.82
fused = sigmoid(−5.82) = 0.003   →  far below t_med 0.155  →  Reliable ✅
```

That 0.003 matches the demo's reported `fused = 0.00297` — and you can see the SEP term (−4.69)
dwarfs the others, which is exactly what the coefficient said it would.

### 4c. Why the two are so different

| | 8B (`s1`) | 1B (`l1b`) |
|---|---|---|
| Heads in the fused score | TSV only | SEP + HalluShift + TSV |
| Dominant head | TSV | SEP (by far) |
| Thresholds (med / high) | 0.70 / 0.90 | 0.155 / 0.258 |

The heads read each model's **own** internals, so which detector carries the truth signal changes
with the model (TSV for the 8B, SEP for the 1B). The fusion is **re-fit per model**, and its
thresholds are **re-calibrated per model** (F1-optimal on that model's training split) — which is
why the 1B's cut-offs (0.155 / 0.258) sit on a totally different scale than the 8B's (0.70 / 0.90).
You can never reuse one model's fusion or thresholds for the other.

---

## 5. From probability to risk tier

Two thresholds split the fused probability into three bands (`hallking/risk.py`), using the
**per-model** `..._thresholds.json`:

| Fused probability | Tier | UI label | Colour |
|---|---|---|---|
| `fused < t_med` | `ok` | **Reliable** | green |
| `t_med ≤ fused < t_high` | `medium` | **Uncertain** | yellow |
| `fused ≥ t_high` | `high` | **Likely Hallucinated** | red |

Thresholds come from the calibrated JSON at load time, not the `risk.py` defaults — the served
sentence-level fusion lives on a small score scale, so the generic 0.50/0.74 defaults would never
flag anything.

---

## 6. Per-sentence scoring → one answer-level headline

For a multi-sentence answer the demo does **localization** (`hallking/localize.py` +
`HallKingPipeline.score_with_sentences`):

1. Generate the answer, split it into sentences.
2. A claim filter drops non-claims (greetings, meta-commentary, questions) — those get
   `fused = null`, tier `filler`, and are never flagged.
3. Score **each claim sentence** independently with the three heads + the same fusion.
4. Aggregate to one headline number (`_aggregate_from_sentences`): the answer-level `fused` is the
   **2nd-worst** claim sentence (not the single worst), so one stray flagged sentence makes the
   answer "Uncertain" rather than "Likely Hallucinated" — it takes two+ bad sentences to drive the
   headline red. The per-detector sub-scores shown are **averages** over the claim sentences, and
   `n_flagged / n_sentences` counts the high-risk ones.

---

## 7. The rank-mean fusion (benchmark only, not the demo)

The cross-dataset benchmark notebooks (`11_*`, `14_*`) use a second fusion variant,
`kind="rankmean"` (`models/fusion_triviaqa_crosseval.pkl`): each detector is converted to its
**within-batch percentile rank**, then the three ranks are averaged. It has **no weights, no
scaler, nothing trained** — so it can't overfit to one dataset and is model-independent (the same
file works for the 8B and 1B). It needs a whole batch to rank, so it is **batch-only** and never
serves the live demo; it exists to test "does fusing stay on top across datasets even when the best
single detector changes?" without a per-dataset-fit advantage.

---

## 8. Where this lives / how to change it

| Concern | File |
|---|---|
| Fusion math + save/load | [`hallking/fusion.py`](../hallking/fusion.py) (`FusionModel`, `_proba1`, `_rankmean`) |
| Risk tiers + labels | [`hallking/risk.py`](../hallking/risk.py) |
| Per-sentence + aggregate | [`hallking/pipeline.py`](../hallking/pipeline.py) (`score_with_sentences`, `_aggregate_from_sentences`) |
| Head scorers | `hallking/{sep,hallushift,tsv}_adapter.py` |
| Train + calibrate a fusion | [`tools/train_claim_fusion.py`](../tools/train_claim_fusion.py) (`feats=` picks which heads; thresholds = F1-optimal on train) |
| Deployed artifacts | `models/fusion_claim_<tag>.pkl` + `_thresholds.json` (`s1` = 8B, `l1b` = 1B) |

To change which heads drive a model's fused score, re-run the fusion cell of its training notebook
with a different `FUSION_FEATS` (e.g. `['hallushift','tsv_margin']` for the 1B) — no head retraining
needed, it just re-fits the logistic blend on the already-scored claims and re-writes the `.pkl` +
thresholds.
