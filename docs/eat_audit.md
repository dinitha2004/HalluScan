# EAT audit — does anchoring at the Exact Answer Token help?

Validation for the Exact Answer Token (EAT) idea, run by `notebooks/12_eat_experiment.ipynb`. This
file is the **verdict the demo highlight is gated on** — Phase 3 (the in-demo highlight) is only built
after the numbers below clear the bar.

**Background.** Orgad et al., *LLMs Know More Than They Show* (ICLR 2025), find that probing a model's
hidden state **at the exact answer token** ("Paris", "1969") detects errors much better than at the
last token of the sequence. HallKing currently anchors every head at an *end-of-sequence* token
(SEP=second-last, TSV=last, HalluShift=whole range) — the very behaviour `head_audit.md` blames for the
**length confound** (heads ride answer length, not truth). So re-anchoring at the EAT targets both the
weak heads and the confound.

> Status: **2a PASSED, 2b DONE (2026-06-20) — VERDICT: EAT anchoring does NOT improve detection.**
> Keep the current end-anchored scoring; the EAT survives only as an optional pure-UX pointer (see Verdict).

---

## Phase 2a — extraction quality (the first gate) — ✅ PASSED

From `tools/eat_eval.py` (single-sentence answers, LLM self-extraction validated as a verbatim
substring). The **gate metric** is "EAT↔gold among correct" = EAT contains/equals gold **among answers
that actually contain the gold**, which isolates extraction quality from model correctness (the
extractor cannot match a gold the model never said).

| dataset | n | extraction rate | model correct | **EAT↔gold among correct (gate)** | wrong-span | missed-span |
|---|--:|--:|--:|--:|--:|--:|
| triviaqa | 200 | 95.5% | 70.0% | **92.9%** (130/140) | 9 | 1 |
| squad | 200 | 88.0% | 42.5% | **92.9%** (79/85) | 6 | 0 |

Unconditional headline numbers (model-correctness-bound, lower by design): EAT==gold (EM) 49.5% / 33.5%;
EAT↔gold (contains) 65.0% / 40.0%. These track **model correctness**, not extractor quality — SQuAD is
closed-book here (context dropped) so the model is only right 42.5% of the time.

**Gate:** extraction rate ≳ 85% **and** EAT↔gold among correct ≳ 85%. **Met on both** (95.5%/88.0% and
92.9%/92.9%) → proceed to 2b.

Failure-mode notes:
- **`EAT=None` is the guard working, not a failure.** Almost all None cases are refusals ("I'm sorry,
  I cannot verify…", "I'm not aware of the specific game…") — no answer span exists, so None is correct.
  This is most of SQuAD's 12% non-extraction.
- **Wrong-span (~6–7%)** is mostly (a) genuine ambiguity on comparison/multi-entity questions ("Hard Rock
  Stadium, then known as Sun Life Stadium" → picked current name, gold wanted old) and (b) normalization
  noise where the EAT is actually right (`two` vs gold `2`; `8` titles where the EAT correctly flags the
  model's wrong number). **One genuine extractor bug**: grabbed just `"The"` from "The American Football
  Conference." (1 case) — tightenable in the extraction prompt; not a blocker.

---

## Phase 2b — anchor comparison (the real test) — ❌ EAT does not help

Re-anchor each head at the EAT token vs its current end anchor; retrain the probe heads; same held-out
split (n=400 triviaqa, halluc 19.2%, EAT found 94%).

### Held-out AUROC (higher is better)

| head | END anchor | EAT anchor | Δ |
|---|--:|--:|--:|
| SEP (logreg probe) | 0.863 | 0.767 | **−0.096** |
| TSV (frozen, read-position only) | 0.884 | 0.823 | **−0.061** |
| HalluShift (CombinedNN, from nb) | 0.576 | 0.731 | +0.155 *(unreliable — see note)* |
| HalluShift (logreg readout, CPU recheck) | 0.811 | 0.790 | −0.021 |

**The two strong heads (SEP, TSV) both get WORSE at the EAT anchor.** HalluShift's CombinedNN "EAT win"
is an artifact: the END-anchored CombinedNN collapsed on test (val 0.866 → test 0.576 — the known
CombinedNN instability, `head_audit.md`/nb10), not a real EAT benefit. A stable logistic readout on the
same features shows END (0.811) ≳ EAT (0.790). → HalluShift = **no reliable gain**. (SEP numbers
reproduce exactly between the notebook's `retrain_sep` and the CPU recheck — both 0.863/0.767.)

### Answer-length correlation (the confound — lower \|ρ\| is better)

| head | \|ρ\| END | \|ρ\| EAT |
|---|--:|--:|
| SEP | 0.364 | 0.434 *(worse)* |
| HalluShift (logreg) | 0.459 | 0.197 *(better — but no AUROC gain, so moot)* |
| TSV | n/a (not computed) | n/a |

The length-confound hypothesis is **not supported**: SEP's |ρ| rises, and HalluShift's drop doesn't buy
any AUROC. Re-anchoring at the answer token did not fix the confound `head_audit.md` documented.

---

## Verdict — EAT anchoring does NOT improve detection ❌

- [x] **2a passed** (extraction reliable — 92.9% span quality among correct, both datasets)
- [x] **2b run → EAT does NOT help.** Strong heads degrade at the EAT (SEP 0.863→0.767, TSV 0.884→0.823);
  HalluShift shows no reliable gain (apparent CombinedNN win is a readout artifact); length confound not
  reduced. The end-of-sequence anchor wins here.

**Why (brief):** this is an *in-distribution* test where the end anchor is already strong (SEP 0.86,
TSV 0.88). For these short single-sentence answers the end-of-sequence token has aggregated the whole
answer and carries more truthfulness signal than the lone answer token does for these heads/readouts.
The project's real weakness is cross-dataset transfer + length — which EAT did not fix.

**Decision for Phase 3 (demo):**
- **Do NOT re-anchor detection at the EAT.** Keep the current end-anchored scoring (it's better).
- **No per-EAT confidence number** — the measurement tier is not earned.
- The EAT idea survives *only* as an **optional pure-UX pointer**: since extraction is reliable (2a), we
  *could* highlight the exact answer span **within an already-flagged sentence**, labelled **"exact
  answer — verify this"** (never "the hallucinated token"); the score shown stays the existing
  sentence-level fused number. Ship this only if the UX value justifies one extra short generation per
  claim sentence — it adds **zero** detection accuracy.
