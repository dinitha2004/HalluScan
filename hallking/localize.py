"""Per-sentence hallucination localization for long answers.

Generates one (longer) answer, splits it into sentences, and scores EACH sentence with all
three detectors extracted from that single generation:
  * SEP  -> SLT probe at each sentence's end token,
  * HalluShift -> features over each sentence's generation-step range,
  * TSV  -> steered last-token rep at each sentence-end token (on the base model).
The trained fusion model then gives a per-sentence hallucination probability, so the demo can
highlight WHERE a long answer goes wrong.

NOTE: per-sentence scores are uncalibrated (there is no per-sentence ground truth) — they are
indicative highlights. The answer-level score (notebook 3) is the calibrated number.
"""
from sentence_segmenter import split_sentences_with_spans
from risk import tier as risk_tier


def char_ends_to_step_ranges(engine, gen_result, answer_char_ends):
    """Map answer-relative char offsets to generation-step ranges (one step per new token)."""
    tok = engine.tokenizer
    seq = gen_result["sequences"][0]
    plen = gen_result["prompt_len"]
    new_ids = seq[plen:]
    n_new = len(new_ids)
    decoded_len = [0] * (n_new + 1)
    for kk in range(1, n_new + 1):
        decoded_len[kk] = len(tok.decode(new_ids[:kk], skip_special_tokens=True))
    ranges, prev = [], 0
    for ce in answer_char_ends:
        k = prev
        while k < n_new and decoded_len[k] < ce:
            k += 1
        k = min(max(k, prev + 1), n_new) if n_new > 0 else 0
        ranges.append((prev, max(k, prev + 1)))
        prev = k
    return ranges


def per_sentence_features(pipe, question, gen=None, answer=None, max_new_tokens=200,
                          use_claim_filter=False):
    """The SHARED per-sentence feature extractor — used by BOTH the live demo (localize) and the
    Option-B dataset builder, so training features == inference features.

    Returns {'answer': str, 'gen': gen, 'sentences': [{sentence, is_claim, sep_entropy, sep_accuracy,
    hallushift, tsv_margin}, ...]} — RAW detector features only, no fusion/label.
    """
    if gen is None:
        # match the regime the loaded heads were trained on (Option B = sentence; Option A = short QA)
        gen = (pipe.engine.generate_sentence(question, max_new_tokens=max_new_tokens)
               if getattr(pipe, "sentence_tag", None)
               else pipe.engine.generate(question, max_new_tokens=max_new_tokens))
    answer = gen["answer_full"] if answer is None else answer
    spans = split_sentences_with_spans(answer)
    sentences = [s["sentence"] for s in spans]
    char_ends = [s["end"] for s in spans]
    claim_flags = [True] * len(sentences)
    if use_claim_filter:
        try:
            from claim_filter import claim_detector
            claim_detector.load_nli_model()
            claim_flags = claim_detector.classify_sentences(sentences)
        except Exception as e:
            print("[localize] claim filter unavailable, scoring all sentences:", e)

    sep_list = pipe.sep.score_sentences(gen, sentences, char_ends, claim_flags)
    hs_list = pipe.hs.score_sentences(gen, char_ends_to_step_ranges(pipe.engine, gen, char_ends))
    # TSV trained on the QA prompt with SHORT answers -> score each claim sentence as an INDEPENDENT
    # short (question, sentence) pair (avoids the length drift of reading the long generation).
    tsv_list = [pipe.tsv.score_qa(question, s)["tsv_margin"] if claim_flags[i] else 0.0
                for i, s in enumerate(sentences)]

    rows = []
    for i, s in enumerate(sentences):
        rows.append({"sentence": s, "is_claim": bool(claim_flags[i]),
                     "sep_entropy": float(sep_list[i]["sep_entropy"]),
                     "sep_accuracy": float(sep_list[i]["sep_accuracy"]),
                     "hallushift": float(hs_list[i]), "tsv_margin": float(tsv_list[i])})
    return {"answer": answer, "gen": gen, "sentences": rows}


def localize(pipe, question, max_new_tokens=200, use_claim_filter=False, gen=None, answer=None):
    """Returns {'answer': str, 'sentences': [{sentence, is_claim, fused, tier, sep_entropy,
    sep_accuracy, hallushift, tsv_margin}, ...]}.

    Thin wrapper over `per_sentence_features` that adds the fused probability + risk tier. Fillers
    (non-claim) are NOT flagged: fused=None, tier='filler'. Pass `gen`/`answer` to reuse a generation
    / segment a specific (e.g. concise) text."""
    feat = per_sentence_features(pipe, question, gen=gen, answer=answer,
                                 max_new_tokens=max_new_tokens, use_claim_filter=use_claim_filter)
    # Use the pipeline's calibrated thresholds (the backend sets these from the fusion's thresholds JSON for
    # Option B; falls back to risk.py's 0.50/0.74 Option-A defaults when a pipe has none).
    import risk
    t_med = getattr(pipe, "t_med", risk.T_MED)
    t_high = getattr(pipe, "t_high", risk.T_HIGH)
    out = []
    for r in feat["sentences"]:
        row = {k: r[k] for k in ("sep_entropy", "sep_accuracy", "hallushift", "tsv_margin")}
        if r["is_claim"] and pipe.fusion is not None:
            fused = float(pipe.fusion.predict_proba_row(row))
            tier = risk_tier(fused, t_med, t_high)
        else:
            fused, tier = None, "filler"
        out.append({"sentence": r["sentence"], "is_claim": r["is_claim"],
                    "fused": fused, "tier": tier, **row})
    return {"answer": feat["answer"], "sentences": out}


def render_highlight(result, threshold: float = 0.5) -> str:
    """Plain-text rendering: each sentence tagged with its fused hallucination probability."""
    lines = []
    for r in result["sentences"]:
        f = r["fused"]
        if f is None:
            tag = ""
        else:
            tag = f"   <<HALLUCINATION p={f:.2f}>>" if f >= threshold else f"   (ok p={f:.2f})"
        lines.append(r["sentence"].strip() + tag)
    return "\n".join(lines)
