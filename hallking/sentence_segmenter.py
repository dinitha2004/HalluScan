"""
Robust sentence boundary detection with 3-tier fallback.

Fallback chain:
  1. spaCy (en_core_web_sm sentencizer) — best accuracy
  2. pysbd (rule-based, 97.92% Golden Rule accuracy) — lightweight, no NLP pipeline
  3. Regex — abbreviation-aware last-resort fallback

All tiers handle:
  - Abbreviations and titles (Dr., Mr., U.S., e.g., i.e.)
  - Decimals and numbers (3.14, $2.5M)
  - URLs and emails
  - Newlines and bullet lists (each bullet → separate sentence)
"""

import re
from typing import List, Dict

# ──────────── optional imports with graceful fallback ────────────
try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

try:
    import pysbd
    _PYSBD_AVAILABLE = True
except ImportError:
    _PYSBD_AVAILABLE = False

# ──────────── constants ────────────

# Regex for bullet / numbered list lines
_BULLET_RE = re.compile(
    r"^\s*(?:[-•*▪▸►]|\d{1,3}[.)]\s|[a-zA-Z][.)]\s)",
    re.MULTILINE,
)

# Known abbreviations that should NOT end a sentence
_ABBREVIATIONS = frozenset([
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st",
    "gen", "gov", "sgt", "cpl", "pvt", "capt", "lt", "col",
    "rev", "hon", "pres",
    "inc", "corp", "ltd", "co", "llc",
    "dept", "div", "est", "assn",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "sept", "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "vs", "etc", "approx", "appt", "apt",
    "ave", "blvd", "rd", "hwy",
    "fig", "figs", "eq", "eqs", "no", "nos", "vol", "vols",
    "al",   # et al.
    "e", "i",  # e.g., i.e.  (single-letter portions)
])

# Multi-letter abbreviation sequences to protect (e.g., U.S.A., D.C.)
_ABBREV_SEQUENCE_RE = re.compile(r"\b([A-Z]\.){2,}")

# URL pattern (simplified but sufficient)
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
)

# Decimal / number pattern  (e.g.  3.14,  $2.5M,  1,234.56)
_DECIMAL_RE = re.compile(r"\d+\.\d+")


# ──────────── spaCy backend ────────────

_spacy_nlp = None

def _get_spacy_nlp():
    """Lazy-load and cache spaCy model."""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded — try the basic sentencizer pipeline
            _spacy_nlp = spacy.blank("en")
            _spacy_nlp.add_pipe("sentencizer")
    return _spacy_nlp


def _split_spacy(text: str) -> List[Dict]:
    """Split using spaCy sentence segmentation."""
    nlp = _get_spacy_nlp()
    doc = nlp(text)
    results = []
    for sent in doc.sents:
        s = sent.text.strip()
        if s:
            results.append({
                "sentence": s,
                "start": sent.start_char,
                "end": sent.end_char,
                "source": "sentencizer",
            })
    return results


# ──────────── pysbd backend ────────────

def _split_pysbd(text: str) -> List[Dict]:
    """Split using pysbd rule-based segmenter."""
    segmenter = pysbd.Segmenter(language="en", clean=False)
    raw_sentences = segmenter.segment(text)

    results = []
    cursor = 0
    for sent_text in raw_sentences:
        s = sent_text.strip()
        if not s:
            continue
        # Find position in original text from cursor onwards
        idx = text.find(s, cursor)
        if idx == -1:
            # Fallback: pysbd may have altered whitespace
            idx = cursor
        start = idx
        end = start + len(s)
        results.append({
            "sentence": s,
            "start": start,
            "end": end,
            "source": "pysbd",
        })
        cursor = end
    return results


# ──────────── regex backend ────────────

def _split_regex(text: str) -> List[Dict]:
    """
    Abbreviation-aware regex sentence splitter (last resort).

    Strategy:
      1. Protect URLs, decimals, and abbreviation sequences with placeholders.
      2. Split on sentence-ending punctuation followed by space + uppercase.
      3. Restore placeholders.
    """
    # --- protect special patterns ---
    protected = {}
    counter = [0]

    def _protect(match):
        key = f"\x00PROT{counter[0]}\x00"
        protected[key] = match.group(0)
        counter[0] += 1
        return key

    work = _URL_RE.sub(_protect, text)
    work = _DECIMAL_RE.sub(_protect, work)
    work = _ABBREV_SEQUENCE_RE.sub(_protect, work)

    # --- protect known abbreviation dots ---
    def _protect_abbrev(m):
        word = m.group(1).lower().rstrip(".")
        if word in _ABBREVIATIONS:
            return _protect(m)
        return m.group(0)

    work = re.sub(
        r"\b([A-Za-z]{1,10})\.",
        _protect_abbrev,
        work,
    )

    # --- split on sentence-ending punctuation ---
    # Match .!? followed by whitespace and an uppercase letter or end-of-string
    raw_parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', work)

    # --- restore and build results ---
    def _restore(s):
        for key, val in protected.items():
            s = s.replace(key, val)
        return s

    results = []
    # We need to track offsets in original text
    cursor = 0
    for part in raw_parts:
        restored = _restore(part).strip()
        if not restored:
            continue
        idx = text.find(restored, cursor)
        if idx == -1:
            idx = cursor
        start = idx
        end = start + len(restored)
        results.append({
            "sentence": restored,
            "start": start,
            "end": end,
            "source": "regex",
        })
        cursor = end

    return results


# ──────────── bullet / newline pre-processing ────────────

def _split_blocks(text: str) -> List[Dict]:
    """
    Pre-process text into blocks split by blank lines and bullet items.

    Returns a list of dicts with:
      - "text": block content
      - "start": offset in original text
      - "source": "bullet" | "newline" | "paragraph"
    """
    blocks = []
    # Split on blank lines first (two or more newlines)
    paragraphs = re.split(r"\n\s*\n", text)

    cursor = 0
    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            # skip empty
            idx = text.find(para, cursor)
            if idx != -1:
                cursor = idx + len(para)
            continue

        idx = text.find(para_stripped, cursor)
        if idx == -1:
            idx = cursor

        # Check if this paragraph is a set of bullet lines
        lines = para_stripped.split("\n")
        if len(lines) > 1 and all(_BULLET_RE.match(line) for line in lines if line.strip()):
            # Each bullet line is its own block
            line_cursor = idx
            for line in lines:
                ls = line.strip()
                if not ls:
                    continue
                li = text.find(ls, line_cursor)
                if li == -1:
                    li = line_cursor
                blocks.append({
                    "text": ls,
                    "start": li,
                    "end": li + len(ls),
                    "source": "bullet",
                })
                line_cursor = li + len(ls)
        else:
            # Also check individual lines for bullet patterns
            if len(lines) > 1:
                has_bullets = False
                for line in lines:
                    ls = line.strip()
                    if ls and _BULLET_RE.match(ls):
                        has_bullets = True
                        break

                if has_bullets:
                    line_cursor = idx
                    for line in lines:
                        ls = line.strip()
                        if not ls:
                            continue
                        li = text.find(ls, line_cursor)
                        if li == -1:
                            li = line_cursor
                        src = "bullet" if _BULLET_RE.match(ls) else "newline"
                        blocks.append({
                            "text": ls,
                            "start": li,
                            "end": li + len(ls),
                            "source": src,
                        })
                        line_cursor = li + len(ls)
                else:
                    blocks.append({
                        "text": para_stripped,
                        "start": idx,
                        "end": idx + len(para_stripped),
                        "source": "paragraph",
                    })
            else:
                blocks.append({
                    "text": para_stripped,
                    "start": idx,
                    "end": idx + len(para_stripped),
                    "source": "paragraph",
                })

        cursor = idx + len(para_stripped)

    # If no blocks were produced, return the whole text
    if not blocks:
        stripped = text.strip()
        if stripped:
            idx = text.find(stripped)
            blocks.append({
                "text": stripped,
                "start": idx if idx != -1 else 0,
                "end": (idx if idx != -1 else 0) + len(stripped),
                "source": "paragraph",
            })

    return blocks


# ──────────── core sentencizer (dispatches to backend) ────────────

def _sentencize_text(text: str) -> List[Dict]:
    """Run the best available sentencizer on a plain text block."""
    if _SPACY_AVAILABLE:
        return _split_spacy(text)
    elif _PYSBD_AVAILABLE:
        return _split_pysbd(text)
    else:
        return _split_regex(text)


# ──────────── public API ────────────

def split_sentences_with_spans(text: str) -> List[Dict]:
    """
    Split text into sentences with character-level spans.

    Returns:
        List of dicts, each containing:
            - "sentence": str  — the sentence text
            - "start": int     — start character offset in original text
            - "end": int       — end character offset in original text
            - "source": str    — "bullet" | "newline" | "sentencizer" | "pysbd" | "regex"
    """
    if not text or not text.strip():
        return []

    # Step 1: pre-process into blocks (bullets, paragraphs)
    blocks = _split_blocks(text)

    results = []
    for block in blocks:
        if block["source"] == "bullet":
            # Bullet lines are already individual sentences
            results.append({
                "sentence": block["text"],
                "start": block["start"],
                "end": block["end"],
                "source": "bullet",
            })
        else:
            # Run sentence segmentation on this block
            sub_results = _sentencize_text(block["text"])
            # Adjust offsets relative to original text
            offset = block["start"]
            for sr in sub_results:
                results.append({
                    "sentence": sr["sentence"],
                    "start": sr["start"] + offset,
                    "end": sr["end"] + offset,
                    "source": sr["source"],
                })

    # Deduplicate / sanity check: remove empty sentences
    results = [r for r in results if r["sentence"].strip()]
    return results


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences (simple string list).

    Uses the same robust pipeline as split_sentences_with_spans.
    """
    return [r["sentence"] for r in split_sentences_with_spans(text)]


def get_backend_name() -> str:
    """Return the name of the active sentence splitting backend."""
    if _SPACY_AVAILABLE:
        return "spacy"
    elif _PYSBD_AVAILABLE:
        return "pysbd"
    else:
        return "regex"
