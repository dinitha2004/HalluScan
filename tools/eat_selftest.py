"""CPU self-test for hallking/eat.py — no model, no GPU.

Exercises the pure-Python pieces (locate_eat_span, eat_token_range, and extract_eat_text's
parsing/validation) with a fake char-piece tokenizer + a stubbed engine.chat, so the
char<->token mapping and the verbatim-substring guard are verified before any GPU pass.

Run:
    python tools/eat_selftest.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hallking"))

from eat import locate_eat_span, eat_token_range, extract_eat_text


class FakeTok:
    """Token id -> string piece; decode = concatenation (mirrors cumulative decode)."""
    def __init__(self, id2piece):
        self.id2piece = id2piece

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.id2piece[int(i)] for i in ids)


class FakeEngine:
    def __init__(self, id2piece=None, reply=None):
        self.tokenizer = FakeTok(id2piece or {})
        self._reply = reply

    def chat(self, user, system=None, max_new_tokens=8):
        return self._reply


def _build_gen(prompt_pieces, answer_pieces):
    """Return (gen, answer_text, engine) for a fake prompt+answer token sequence."""
    pieces = list(prompt_pieces) + list(answer_pieces)
    id2piece = {i: p for i, p in enumerate(pieces)}
    ids = list(range(len(pieces)))
    plen = len(prompt_pieces)
    eng = FakeEngine(id2piece=id2piece)
    gen = {"sequences": [ids], "prompt_len": plen}
    answer = eng.tokenizer.decode(ids[plen:])
    return gen, answer, eng, plen


PASS, FAIL = 0, 0


def check(name, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok   {name}: {got!r}")
    else:
        FAIL += 1
        print(f"  FAIL {name}: got {got!r} want {want!r}")


def main():
    # ---- locate_eat_span ----
    s = "Paris is the capital of France."
    check("locate entity", locate_eat_span(s, "Paris"), (0, 5))
    check("locate trailing word", locate_eat_span(s, "France"), (24, 30))
    check("locate phrase", locate_eat_span(s, "the capital of France"), (9, 30))
    check("locate case-insensitive", locate_eat_span(s, "paris"), (0, 5))
    check("locate absent", locate_eat_span(s, "London"), None)
    check("locate ws-flexible", locate_eat_span("Paris  is here", "Paris is"), (0, 9))
    check("locate empty", locate_eat_span(s, ""), None)

    # ---- eat_token_range (entity / trailing / date) ----
    gen, ans, eng, plen = _build_gen(
        ["<p0>", "<p1>"], ["Paris", " is", " the", " capital", " of", " France", "."])
    assert ans == s, ans
    st, en = locate_eat_span(ans, "Paris")
    check("range entity", eat_token_range(eng, gen, st, en), (plen + 0, plen + 1))
    st, en = locate_eat_span(ans, "France")
    check("range trailing", eat_token_range(eng, gen, st, en), (plen + 5, plen + 6))
    st, en = locate_eat_span(ans, "the capital of France")
    check("range phrase", eat_token_range(eng, gen, st, en), (plen + 2, plen + 6))

    gen2, ans2, eng2, plen2 = _build_gen(["<p>"], ["It", " was", " 1969", "."])
    st, en = locate_eat_span(ans2, "1969")
    check("range date", eat_token_range(eng2, gen2, st, en), (plen2 + 2, plen2 + 3))

    # empty answer -> degenerate but safe
    gen3 = {"sequences": [[0, 1]], "prompt_len": 2}
    check("range empty-answer", eat_token_range(FakeEngine({0: "a", 1: "b"}), gen3, 0, 1), (2, 2))

    # ---- extract_eat_text (parsing + verbatim guard) ----
    def ext(reply):
        return extract_eat_text(FakeEngine(reply=reply), "What is the capital of France?", s)
    check("extract plain", ext("Paris"), "Paris")
    check("extract quoted", ext('"Paris"'), "Paris")
    check("extract trailing-period", ext("Paris."), "Paris")
    check("extract NONE", ext("NONE"), None)
    check("extract paraphrase-rejected", ext("the City of Light"), None)
    check("extract empty", ext(""), None)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
