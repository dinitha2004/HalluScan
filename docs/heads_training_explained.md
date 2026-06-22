# How the three detector heads were trained (beginner guide)

HalluScan's lie‑detector is built from **three heads** — SEP, HalluShift, TSV. This explains how each one was *taught* to spot hallucinations. (How they're then blended is in `fusion_model_weights.md`.)

The same recipe was used for both models (`hallking/retrain.py` + `tools/train_claim_heads.py`):
the **1B** (tag `l1b`, notebook 13) and the **8B** (tag `8bv2`, notebook 15).

> **Key fact:** each head reads the model's *own* internal activations, so a head trained on one model can't score another. That's why we retrain all three per model.

---

## 0. The ingredients

| | 1B (`l1b`) | 8B (`8bv2`) |
|---|---|---|
| Model | Llama‑3.2‑1B‑Instruct | Llama‑3.1‑8B‑Instruct |
| Dataset | **TriviaQA** | **TriviaQA** |
| Questions | **1,500** (1,379 kept) | **1,500** (1,496 kept) |
| Layers (affects feature sizes) | 16 | 32 |
| Graded true/false by | a stronger **8B judge** | the **8B itself** (self‑judge) |
| Hallucination rate found | 47.6% | 15.8% |

---

## 1. Step one — generate answers and record "evidence" (`gen_and_cache`)

We ask the model **1,500 TriviaQA questions**. For each, the model writes **one short factual sentence**. While it writes, we secretly record the raw material each head needs ([retrain.py](../hallking/retrain.py) `gen_and_cache`):

- **SEP evidence** — the model's hidden state at *every layer* at the second‑to‑last token, flattened into one long vector (135,168 numbers for the 8B; 34,816 for the 1B).
- **HalluShift evidence** — 71 summary numbers (8B) describing how the activations *shift* across layers + how confident each token was. (39 numbers for the 1B's 16 layers.)
- **The answer text** — kept for TSV, which re‑reads the (question, answer) pair.

"I don't know" refusals are dropped (they're neither claims nor hallucinations).

### Labelling — the ground truth
To teach the heads, every answer needs a true/false tag. We do it in two passes:
1. **Substring match** — if the correct answer text appears in the model's answer → truthful (easy cases).
2. **Judge rescue** — the unclear ones are shown to a judge model (question + correct answer + the model's answer) which decides correct/incorrect. The 1B is too weak to grade itself, so a **stronger 8B judged it**; the 8B **judged its own** answers.

---

## 2. Step two — one fair train/test split

All rows are split **75% train / 25% test**, stratified (same hallucination ratio on both sides), with a fixed seed (42). Crucially the **same split is shared by all three heads**, so when we compare them later, every head is graded on the *exact same* held‑out questions it never trained on.

---

## 3. Step three — train each head

### SEP — a probe on the hidden states (`retrain_sep`)
SEP fits **two simple logistic‑regression formulas** on the flattened all‑layer hidden vector:
- one trained to output **P(hallucinated)** → `sep_entropy`
- one trained to output **P(truthful)** → `sep_accuracy`

It's a linear "reader" of the model's internal snapshot. (CPU‑only, fast, regularization `C=0.1`.)

### HalluShift — a tiny neural net on the 71 features (`retrain_hallushift`)
1. **Standardize** the 71 features (so each is on a comparable scale).
2. Train a small neural net (`CombinedNN`) to output a hallucination probability, using a **class‑weighted loss** (so the rarer class isn't ignored) and **early stopping** on the best validation AUROC. (CPU‑only.)

### TSV — a steering vector + two anchors (`train_tsv`)
This one runs on the GPU, on the **Instruct** model (the same variant the demo serves):
1. Learn a small **steering vector** injected at one mid layer (the 1B uses layer 4 of 16; the 8B uses layer 9 of 32 — about 28% of the depth).
2. Learn two **centroids** (anchor points): one for *truthful* answers, one for *hallucinated*, updated as a running average during training.
3. The score is `tsv_margin = cos(answer, halluc_anchor) − cos(answer, truth_anchor)` — how much the answer leans toward the "lie" anchor.

Trained for 40 epochs; the checkpoint with the best held‑out AUROC is kept.

---

## 4. Step four — grade each head on the held‑out test

Each head scores the 25% test questions it never saw, and we measure **AUROC** (ranking quality; 0.5 = random, 1.0 = perfect):

| Head | 1B (`l1b`) held‑out AUROC | 8B (`8bv2`) held‑out AUROC |
|---|---|---|
| SEP | 0.741 | 0.777 |
| HalluShift | 0.793 | 0.800 |
| TSV | 0.727 | 0.759 |

All three "separate" (well above 0.65), so all three are worth keeping for the fusion. The 8B's heads are a bit stronger across the board — a bigger model carries a clearer internal truth signal.

---

## 5. What changes between the two models (and what doesn't)

| | 1B | 8B |
|---|---|---|
| HalluShift feature count | 39 (16 layers) | 71 (32 layers) |
| TSV steering layer | 4 / 16 | 9 / 32 |
| Judge for labels | decoupled 8B | self‑judge |
| **Recipe / split / code** | **identical** | **identical** |

The feature sizes and TSV layer **auto‑scale** to the model's depth — no code change needed; only the model id and tag differ.

---

## 6. After the heads — the fusion

Once the three heads are trained and have scored the test rows, a small **fusion** model learns how to blend their three scores into one final hallucination probability, and two thresholds turn it into the green/yellow/red bands. That part is covered in **`fusion_model_weights.md`**.

---

## Where this lives

| Concern | File |
|---|---|
| Generate + cache features + labels | `hallking/retrain.py` (`gen_and_cache`) |
| Train SEP / HalluShift / TSV | `hallking/retrain.py` (`retrain_sep`, `retrain_hallushift`, `train_tsv`) |
| Orchestration (8B, deployed) | `tools/train_claim_heads.py` |
| The notebooks | `notebooks/13_train_1b_heads.ipynb` (1B), `notebooks/15_train_8b_heads.ipynb` (8B) |
| Head scorers used at inference | `hallking/{sep,hallushift,tsv}_adapter.py` |
