"""Exact Answer Token (EAT) extraction for HallKing.

The "exact answer token(s)" are the verbatim span inside an answer sentence that *actually*
answers the question — "Paris", "1969", "yes" — as opposed to the surrounding filler. The
concept and its use as a hallucination-detection anchor come from Orgad et al., "LLMs Know More
Than They Show" (ICLR 2025): probing a model's hidden state AT the exact answer token detects
errors far better than at the last token of the sequence.

This module is intentionally STANDALONE — it does NOT touch the pipeline, backend, or frontend.
It is the building block for (Phase 2) an offline validation experiment and only later, once that
verifies, (Phase 3) the demo highlight. See plan + `docs/eat_audit.md`.

Three pieces:
  * `extract_eat_text`  -> LLM self-extraction of the verbatim answer phrase (reuses engine.chat),
                           validated to be a real substring of the sentence (else None).
  * `locate_eat_span`   -> char offsets of that phrase inside the sentence (pure Python, CPU).
  * `eat_token_range`   -> answer-relative char span -> absolute token indices in gen["sequences"],
                           reusing the same progressive-decode mapping as sep_adapter / localize.
"""
import re

# Tight prompt: force a VERBATIM copy (so the span is locatable in the sentence) and a clean
# NONE when there is no direct answer (refusals, hedges, pure filler).
EAT_SYSTEM = (
    "You extract the exact answer span from an answer sentence. "
    "Output ONLY the shortest phrase, copied VERBATIM (character-for-character) from the Answer, "
    "that directly answers the Question. Do not paraphrase, normalise, translate, add quotation "
    "marks, or explain. If the Answer contains no direct answer, output exactly: NONE"
)
EAT_USER = "Question: {question}\nAnswer: {sentence}\nExact answer span:"


def locate_eat_span(sentence: str, eat_text: str):
    """Return (start, end) char offsets of `eat_text` inside `sentence`, or None.

    Tries, in order: case-insensitive direct match, then a whitespace-flexible regex match
    (so " France " differing only by internal/edge spacing still maps back to ORIGINAL offsets).
    Pure Python — no model, CPU-testable."""
    if not sentence or not eat_text or not eat_text.strip():
        return None
    needle = eat_text.strip()
    # 1) direct, case-insensitive (case differences never change length)
    i = sentence.lower().find(needle.lower())
    if i != -1:
        return (i, i + len(needle))
    # 2) whitespace-flexible, case-insensitive; .span() is in ORIGINAL-string coordinates
    words = needle.split()
    if not words:
        return None
    pat = re.compile(r"\s+".join(re.escape(w) for w in words), re.IGNORECASE)
    m = pat.search(sentence)
    return m.span() if m else None


def validate_eat(raw: str, sentence: str):
    """Turn a raw model reply into a verbatim EAT phrase, or None.

    Strips quotes / a trailing period the model may add, rejects NONE, and requires the result to
    be a real (normalised) substring of `sentence` — the guard against paraphrasing. Split out from
    `extract_eat_text` so the eval harness can capture the raw reply AND its verdict from a single
    generation."""
    cand = (raw or "").strip().strip('"“”\'').strip()
    if not cand or cand.upper() == "NONE":
        return None
    for c in (cand, cand.rstrip(".").strip()):
        if c and locate_eat_span(sentence, c) is not None:
            return c
    return None  # not a verbatim substring -> model paraphrased; treat as no EAT


def extract_eat_text(engine, question: str, sentence: str, max_new_tokens: int = 24):
    """LLM self-extraction of the exact answer phrase from `sentence` for `question`.

    Returns the verbatim phrase (as it appears in `sentence`) or None when the model declines
    (NONE) or returns something that is not actually a substring of the sentence. Reuses the lean
    text-only `engine.chat`."""
    if not sentence or not sentence.strip():
        return None
    user = EAT_USER.format(question=question, sentence=sentence)
    raw = engine.chat(user, system=EAT_SYSTEM, max_new_tokens=max_new_tokens)
    return validate_eat(raw, sentence)


def _new_token_decoded_lengths(tokenizer, new_ids):
    """decoded_len[k] = char length of the first k NEW (answer) tokens decoded together.

    Decoding cumulatively (rather than per-token) matches how sep_adapter / localize walk the
    sequence, so token<->char alignment is consistent with the detectors."""
    n_new = len(new_ids)
    decoded_len = [0] * (n_new + 1)
    for k in range(1, n_new + 1):
        decoded_len[k] = len(tokenizer.decode(new_ids[:k], skip_special_tokens=True))
    return decoded_len


def _token_of_char(decoded_len, c, n_new):
    """0-based NEW-token index whose char span [decoded_len[idx], decoded_len[idx+1]) covers char c."""
    for idx in range(n_new):
        if decoded_len[idx] <= c < decoded_len[idx + 1]:
            return idx
    return max(n_new - 1, 0)


def eat_token_range(engine, gen, ans_char_start: int, ans_char_end: int):
    """Map an ANSWER-relative char span [start, end) to ABSOLUTE, end-exclusive token indices
    [tok_start, tok_end) in gen["sequences"][0].

    Mirrors localize.char_ends_to_step_ranges / sep_adapter.score_sentences so the EAT anchor in
    Phase 2 reads exactly the positions the existing detectors would. The last EAT token
    (tok_end - 1) is the natural single-token anchor (analogous to SEP's SLT / TSV's last token).
    """
    tok = engine.tokenizer
    seq = gen["sequences"][0]
    plen = gen["prompt_len"]
    new_ids = seq[plen:]
    n_new = len(new_ids)
    if n_new == 0:
        return (plen, plen)
    decoded_len = _new_token_decoded_lengths(tok, new_ids)
    answer_len = decoded_len[n_new]
    start = max(0, min(ans_char_start, answer_len - 1))
    end = max(start + 1, min(ans_char_end, answer_len))
    tok_start = _token_of_char(decoded_len, start, n_new)
    tok_last = _token_of_char(decoded_len, end - 1, n_new)
    return (plen + tok_start, plen + tok_last + 1)
