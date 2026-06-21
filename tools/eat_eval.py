"""Phase 2a — EAT EXTRACTION-QUALITY harness (the first gate; no retraining).

For each question we generate a single-sentence answer (the demo/sentence regime), run the LLM
self-extractor (hallking/eat.py), and measure how good the extracted Exact Answer span is:

  * extraction rate      — did the model return a usable verbatim span (not NONE / not paraphrased)?
  * gold match (EM)      — does the span normalise-equal a gold reference answer?
  * gold match (contains)— span <-> gold containment (either direction)
  * failure breakdown    — wrong span (right answer present but span missed it) vs missed span vs
                           model-wrong (answer doesn't contain gold at all -> not the extractor's fault).

This validates the EXTRACTOR itself before any GPU-heavy feature/anchor work (Phase 2b). Datasets:
triviaqa + squad only (web_questions gold is unusable — see docs/head_audit.md). One model load.

Run in se_probes_env (needs the GPU + the gated Llama weights):
    python tools/eat_eval.py --dataset triviaqa --n 200
    python tools/eat_eval.py --dataset squad --n 200
"""
import argparse
import json
import os
import re
import string
import sys

os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

from eat import EAT_SYSTEM, EAT_USER, validate_eat, locate_eat_span, eat_token_range
# engine / run_dataset (torch, transformers, datasets) are imported lazily inside evaluate() so the
# pure gold-matching helpers below stay importable + unit-testable on a CPU box without the GPU stack.

# ---------------------------------------------------------------- gold matching
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def _norm(s: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation + articles, collapse whitespace."""
    s = (s or "").lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _em_gold(text: str, refs) -> bool:
    t = _norm(text)
    return bool(t) and any(t == _norm(r) for r in refs)


def _contains_gold(text: str, refs) -> bool:
    t = _norm(text)
    if not t:
        return False
    for r in refs:
        rn = _norm(r)
        if rn and (rn in t or t in rn):
            return True
    return False


def _answer_contains_gold(answer: str, refs) -> bool:
    a = _norm(answer)
    return bool(a) and any(_norm(r) and _norm(r) in a for r in refs)


# ---------------------------------------------------------------- main eval
def evaluate(dataset="triviaqa", n=200, offset=0, max_new_tokens=64, eat_max_new_tokens=24,
             save=True, verbose=True):
    from engine import HallKingEngine
    from run_dataset import load_qa, INSTRUCT_MODEL
    questions, refs = load_qa(dataset, n=n, offset=offset)
    if verbose:
        print(f"[eat_eval] {dataset}: {len(questions)} questions (offset={offset})", flush=True)

    eng = HallKingEngine(model_name=INSTRUCT_MODEL).load()
    rows = []
    for i, (q, ref) in enumerate(zip(questions, refs)):
        gen = eng.generate_sentence(q, max_new_tokens=max_new_tokens)
        answer = gen["answer_full"].strip()
        # one generation: capture the raw reply AND its validated verdict
        raw = eng.chat(EAT_USER.format(question=q, sentence=answer), system=EAT_SYSTEM,
                       max_new_tokens=eat_max_new_tokens)
        eat = validate_eat(raw, answer)
        span = locate_eat_span(answer, eat) if eat else None
        tok_range = None
        if span is not None:
            tok_range = eat_token_range(eng, gen, span[0], span[1])  # 2b groundwork / sanity
        rows.append({
            "question": q, "refs": ref, "answer": answer,
            "raw_eat": (raw or "").strip(), "eat": eat,
            "extracted": eat is not None,
            "eat_char_span": list(span) if span else None,
            "eat_token_range": list(tok_range) if tok_range else None,
            "answer_contains_gold": _answer_contains_gold(answer, ref),
            "eat_em_gold": _em_gold(eat, ref) if eat else False,
            "eat_contains_gold": _contains_gold(eat, ref) if eat else False,
        })
        if verbose and i % 25 == 0:
            print(f"  [{i}/{len(questions)}] A={answer[:55]!r}  EAT={eat!r}", flush=True)
    eng.unload()

    _summarize(dataset, rows, verbose=verbose)
    if save:
        out = os.path.join(ROOT, "data", f"eat_eval_{dataset}_n{len(rows)}.jsonl")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if verbose:
            print(f"[eat_eval] saved {os.path.relpath(out, ROOT)}", flush=True)
    return rows


def _summarize(dataset, rows, verbose=True):
    n = len(rows)
    if n == 0:
        print("[eat_eval] no rows."); return {}
    extracted = sum(r["extracted"] for r in rows)
    ans_ok = sum(r["answer_contains_gold"] for r in rows)
    eat_em = sum(r["eat_em_gold"] for r in rows)
    eat_contains = sum(r["eat_contains_gold"] for r in rows)
    # failure breakdown over the answers that DO contain the gold (where a correct EAT is possible)
    wrong_span = sum(1 for r in rows if r["answer_contains_gold"]
                     and r["extracted"] and not r["eat_contains_gold"])
    missed_span = sum(1 for r in rows if r["answer_contains_gold"] and not r["extracted"])
    pct = lambda k: f"{100.0 * k / n:5.1f}%"
    summary = {
        "dataset": dataset, "n": n,
        "extraction_rate": extracted / n,
        "answer_contains_gold": ans_ok / n,
        "eat_em_gold": eat_em / n,
        "eat_contains_gold": eat_contains / n,
        "wrong_span_when_answer_correct": wrong_span,
        "missed_span_when_answer_correct": missed_span,
    }
    if verbose:
        print("\n" + "=" * 64)
        print(f"[eat_eval] {dataset}  (n={n})")
        print(f"  extraction rate (usable verbatim span) : {pct(extracted)}  ({extracted}/{n})")
        print(f"  answer contains gold (model correct)    : {pct(ans_ok)}  ({ans_ok}/{n})")
        print(f"  EAT == gold (exact match)               : {pct(eat_em)}  ({eat_em}/{n})")
        print(f"  EAT <-> gold (contains, either way)     : {pct(eat_contains)}  ({eat_contains}/{n})")
        print(f"  -- failure modes (of the {ans_ok} where the answer was correct) --")
        print(f"     wrong span (right answer, span off)  : {wrong_span}")
        print(f"     missed span (no span extracted)      : {missed_span}")
        print("=" * 64 + "\n")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="triviaqa", choices=["triviaqa", "squad"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--eat_max_new_tokens", type=int, default=24)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    evaluate(dataset=args.dataset, n=args.n, offset=args.offset, max_new_tokens=args.max_new_tokens,
             eat_max_new_tokens=args.eat_max_new_tokens, save=not args.no_save)
