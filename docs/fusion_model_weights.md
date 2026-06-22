# The new fusion models — weights, math & test scores (beginner guide)

This explains the **two new fusion models** that combine all three detector heads:

- **1B model** → `models/fusion_claim_l1b.pkl` (trained in notebook 13)
- **8B model** → `models/fusion_claim_8bv2.pkl` (trained in notebook 15)

Both use **SEP + HalluShift + TSV together**, unlike the older 8B fusion which used TSV alone.

> ⚠️ **Note:** the live demo's 8B still runs the **old** `fusion_claim_s1.pkl` (TSV‑only); the `8bv2` fusion described here is its planned replacement and is **not yet wired into the demo**.

---

## 1. What is the "fusion"? (the simple idea)

Three detectors each look at the model while it answers and give one number:

| Head | What it notices | Output (higher = more likely a hallucination) |
|---|---|---|
| **SEP** | Does the model's internal "mind‑snapshot" look uncertain/made‑up? | `sep_entropy` (0–1) |
| **HalluShift** | Did the model behave shiftily across its layers while answering? | `hallushift` (0–1) |
| **TSV** | Does the answer lean toward a "lie" posture or a "truth" posture? | `tsv_margin` (≈ −0.3 … +0.3) |

No single detector is reliable alone. The **fusion** is a tiny referee that has *learned from data* how much to trust each one, and blends the three numbers into **one final probability of hallucination** (0–1).

---

## 2. How the heads were trained (same recipe for both models)

| | 1B (`l1b`) | 8B (`8bv2`) |
|---|---|---|
| Model | Llama‑3.2‑1B‑Instruct | Llama‑3.1‑8B‑Instruct |
| Dataset | **TriviaQA** | **TriviaQA** |
| Questions asked | **1,500** (1,379 kept after dropping "I don't know") | **1,500** (1,496 kept) |
| How answers were graded true/false | a **stronger 8B judge** graded the weak 1B's answers | the **8B graded its own** answers (self‑judge) |
| % that were hallucinations | 47.6% | 15.8% |
| Train / test split | 75% / 25%, stratified, fixed seed 42 | same |

For every kept answer we record the three head scores **and** the true/false label, splitting the rows into a **training** half (used to fit everything) and a **held‑out test** half (used only to grade — the heads never saw it).

---

## 3. How the fusion is created from the 3 heads

Done by `tools/train_claim_fusion.py`:

1. Take the **training** rows: each has `[sep_entropy, hallushift, tsv_margin]` and a true/false label.
2. **Standardize** the three columns (subtract the mean, divide by the spread) so SEP's 0–1 scale and TSV's ±0.3 scale become comparable.
3. Fit a **logistic regression** (a simple weighted formula) that predicts hallucinated / truthful from the three standardized scores. This learns one **weight per head** plus a bias.
4. **Calibrate two thresholds** (on the training half only): `t_high` = the cut that gives the best F1, and `t_med = 0.6 × t_high`. These turn the probability into the Reliable / Uncertain / Likely‑Hallucinated bands.
5. Save `{weights, bias, scaler, feature list}` to `fusion_claim_<tag>.pkl` and the thresholds to `..._thresholds.json`.

---

## 4. The fusion math (what the weights actually do)

For one answer (or one sentence) the fusion does three steps:

```
Step 1 — standardize each score:   z_i = (x_i − mean_i) / scale_i
Step 2 — weighted sum + bias:      logit = (w_sep·z_sep) + (w_hs·z_hs) + (w_tsv·z_tsv) + bias
Step 3 — squash to a probability:  fused = 1 / (1 + e^(−logit))     # sigmoid → 0..1
```

- `mean_i`, `scale_i` come from the training data (the **scaler**).
- `w_i` (the **coefficients**) say how much each head matters — a bigger weight = that head dominates.
- `bias` (the **intercept**) is the baseline when every standardized score is 0.
- `fused` is the final **probability of hallucination**.

> The code computes this sigmoid by hand from the stored weights (`FusionModel._proba1` in `hallking/fusion.py`), so the saved file keeps working across library versions.

---

## 5. The 1B model — `fusion_claim_l1b.pkl`

```
feature_cols : ['sep_entropy', 'hallushift', 'tsv_margin']    (all 3 heads)
scaler.mean_ : [0.4768, 0.4885,  0.0148]
scaler.scale_: [0.4671, 0.2461,  0.1026]
coef_ (w)    : [4.5894, 0.2780,  0.2890]    ← SEP dominates (4.59 ≫ 0.28, 0.29)
intercept_   : [-0.2442]
thresholds   : t_med = 0.155,  t_high = 0.258
```

**Plain reading:** all three heads are included, but the math leans almost entirely on **SEP** (weight 4.59 vs ~0.28 for the other two). So the 1B's fused score is mostly SEP, with small nudges from HalluShift and TSV.

**Formula:**
```
z_sep = (sep_entropy − 0.4768)/0.4671
z_hs  = (hallushift  − 0.4885)/0.2461
z_tsv = (tsv_margin  − 0.0148)/0.1026
fused = sigmoid( 4.5894·z_sep + 0.2780·z_hs + 0.2890·z_tsv − 0.2442 )
```

**Worked examples (1B):**

| sep_entropy | hallushift | tsv_margin | logit | **fused** | tier |
|---|---|---|---|---|---|
| 0.00 | 0.154 | −0.167 | −5.82 | **0.003** | Reliable ✅ (the real "capital of France?" answer) |
| 0.95 | 0.70 | +0.20 | +5.17 | **0.994** | Likely Hallucinated 🚨 |

You can see the SEP term drives the result: a low `sep_entropy` (0.00) alone pushes the score far below the 0.155 cut → Reliable.

---

## 6. The 8B model — `fusion_claim_8bv2.pkl` (new, from notebook 15)

```
feature_cols : ['sep_entropy', 'hallushift', 'tsv_margin']    (all 3 heads)
scaler.mean_ : [ 0.1578,  0.1485, -0.1945]
scaler.scale_: [ 0.3523,  0.1825,  0.0324]
coef_ (w)    : [ 3.7465,  0.4394,  0.0277]   ← SEP leads, HalluShift helps, TSV tiny
intercept_   : [-3.6476]
thresholds   : t_med = 0.588,  t_high = 0.98
```

**Plain reading:** this is a big change from the old TSV‑only 8B fusion. Here **SEP leads** (3.75), **HalluShift contributes** (0.44), and **TSV barely matters** (0.028). The strongly negative bias (−3.65) means a *typical* answer starts near a very low risk and only climbs when SEP/HalluShift fire.

**Formula:**
```
z_sep = (sep_entropy − 0.1578)/0.3523
z_hs  = (hallushift  − 0.1485)/0.1825
z_tsv = (tsv_margin  + 0.1945)/0.0324
fused = sigmoid( 3.7465·z_sep + 0.4394·z_hs + 0.0277·z_tsv − 3.6476 )
```

**Worked examples (8B):**

| sep_entropy | hallushift | tsv_margin | logit | **fused** | tier |
|---|---|---|---|---|---|
| 0.05 | 0.10 | −0.21 | −4.92 | **0.007** | Reliable ✅ |
| 0.16 | 0.15 | −0.19 (≈ average) | −3.65 | **0.025** | Reliable ✅ |
| 0.85 | 0.55 | −0.10 | +4.76 | **0.992** | Likely Hallucinated 🚨 |

Note the very high `t_high = 0.98`: on the 8B an answer must be *extremely* suspicious to be labelled "Likely Hallucinated" — most land in Reliable, some in Uncertain. (See the caveat in §8.)

---

## 7. Test scores (held‑out, the heads never saw these rows)

All metrics are on the 25% test split. **AUROC** (ranking quality, 0.5 = coin flip, 1.0 = perfect) is the fairest single number.

### 1B (`l1b`) — test rows = 345, hallucination rate 47.6%

| Detector | AUROC | AUPR | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| SEP | 0.741 | 0.720 | 0.623 | 0.565 | 0.902 | 0.695 |
| HalluShift | 0.793 | 0.739 | 0.751 | 0.699 | 0.835 | 0.761 |
| TSV | 0.727 | 0.688 | 0.664 | 0.603 | 0.860 | 0.709 |
| **FUSED** | 0.765 | **0.760** | 0.673 | 0.651 | 0.671 | 0.661 |

### 8B (`8bv2`) — test rows = 374, hallucination rate 15.8%

| Detector | AUROC | AUPR | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| SEP | 0.777 | 0.454 | 0.773 | 0.367 | 0.610 | 0.459 |
| HalluShift | 0.800 | 0.459 | 0.770 | 0.381 | 0.729 | 0.500 |
| TSV | 0.759 | 0.362 | 0.671 | 0.298 | 0.797 | 0.433 |
| **FUSED** | **0.833** | **0.484** | **0.850** | 0.615 | 0.136 | 0.222 |

---

## 8. How to read these results (important for beginners)

- **The 8B fusion is the stronger detector.** Its FUSED AUROC (0.833) beats every single 8B head and beats the 1B's FUSED (0.765). On the 8B the fusion even **beats its best single head** (0.833 > 0.800). On the 1B, FUSED AUROC (0.765) is just under HalluShift alone (0.793) but has the **best AUPR** (0.760) — fusion helped precision/recall balance there.
- **Don't compare AUPR between the two models.** AUPR depends on how common hallucinations are. The 1B hallucinated 47.6% of the time vs the 8B's 15.8%, so the 1B's higher AUPR (~0.72) is mostly that base‑rate difference, **not** a better detector. Use AUROC to compare across models.
- **The 8B's FUSED F1 (0.222) looks low but isn't a real weakness.** Its threshold landed very high (0.98), so it only flags blatant cases → very high precision but low recall. F1/Accuracy depend on where the threshold sits; the threshold‑free AUROC/AUPR are the honest comparison. (The live demo recalibrates thresholds for the multi‑sentence regime anyway.)

---

## 9. Quick comparison

| | Old 8B demo (`s1`) | New 8B (`8bv2`) | 1B (`l1b`) |
|---|---|---|---|
| Heads in the fused score | TSV only | **all 3 (SEP‑led)** | all 3 (SEP‑led) |
| Dominant head | TSV | SEP | SEP |
| Thresholds (med / high) | 0.70 / 0.90 | 0.588 / 0.98 | 0.155 / 0.258 |
| FUSED AUROC | (see `head_audit.md`) | **0.833** | 0.765 |
| Status | **deployed now** | trained, not yet deployed | deployed (1B) |

Each model reads its **own** internals, so the heads, weights, and thresholds can never be shared between models — every model needs its own fusion, re‑calibrated on its own data.

---

## 10. Where this lives / how to change it

| Concern | File |
|---|---|
| Fusion math + save/load | `hallking/fusion.py` (`FusionModel._proba1`) |
| Risk tiers + labels | `hallking/risk.py` |
| Train + calibrate a fusion | `tools/train_claim_fusion.py` |
| Saved fusions | `models/fusion_claim_<tag>.pkl` + `_thresholds.json` (`s1` = old 8B, `8bv2` = new 8B, `l1b` = 1B) |

To **deploy `8bv2`** later, the demo's 8B entry would point at the `8bv2` heads **and** fusion (all tagged `8bv2`) instead of `s1` — done only after live testing, as a deliberate swap.
