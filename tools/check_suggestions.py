"""Validate the UI's starter-question suggestions against the LIVE backend.

The frontend (frontend/src/components/ChatInterface.jsx) shows 4 starter chips: 2 "clean" (the model
answers correctly -> expected LOW risk) and 2 "caught" (the model hallucinates a false premise ->
expected HIGH/MEDIUM risk). They must hold on BOTH served models (1b + 8b), so the audience can switch
the dropdown without a suggestion suddenly behaving wrong.

This script POSTs each suggestion to /infer for each model (the backend lazy-swaps + blocks until the
model is resident), reads aggregate.tier, and prints PASS/FAIL vs the expected band. Keep this list in
sync with SUGGESTIONS in ChatInterface.jsx.

    python tools/check_suggestions.py                      # uses the committed default backend
    python tools/check_suggestions.py --backend https://xxxx.ngrok-free.dev
    python tools/check_suggestions.py --models 1b          # one model only

Exit code is non-zero if any check fails, so it can gate a pre-viva check.
"""
import argparse
import json
import sys
import urllib.request
import urllib.error

# The committed default backend (the reserved ngrok STATIC domain) — same value the deployed frontend
# targets by default (see DEFAULT_BACKEND_URL in frontend/src/App.jsx).
DEFAULT_BACKEND = "https://declared-angular-matchbox.ngrok-free.dev"

# Keep in sync with SUGGESTIONS in ChatInterface.jsx. expect: "ok" (clean) or "flag" (caught: high|medium).
SUGGESTIONS = [
    {"text": "Who painted the Mona Lisa?", "expect": "ok", "kind": "Quick · clean"},
    {"text": "Who was the lead architect of the Eiffel Tower's 1955 expansion?",
     "expect": "flag", "kind": "Quick · caught"},
    {"text": "Give me 5 facts about India", "expect": "ok", "kind": "Detailed · clean"},
    {"text": "What is the psychological effect of eating cheese before 3 PM? Provide three short studies.",
     "expect": "flag", "kind": "Detailed · caught"},
]

# tier -> which expectations it satisfies. "caught" passes on high OR medium; "clean" passes on ok.
_OK_TIERS = {"ok"}
_FLAG_TIERS = {"high", "medium"}


def infer(backend: str, question: str, model: str, timeout: int) -> dict:
    body = json.dumps({"question": question, "model": model}).encode("utf-8")
    req = urllib.request.Request(
        backend.rstrip("/") + "/infer", data=body, method="POST",
        headers={"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def expected_ok(expect: str, tier: str) -> bool:
    return tier in (_OK_TIERS if expect == "ok" else _FLAG_TIERS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default=DEFAULT_BACKEND, help="backend base URL (Colab/ngrok)")
    ap.add_argument("--models", nargs="+", default=["1b", "8b"], help="model keys to check (default: 1b 8b)")
    ap.add_argument("--timeout", type=int, default=300, help="per-request timeout secs (model swap can be slow)")
    args = ap.parse_args()

    print(f"backend: {args.backend}")
    print(f"models : {args.models}\n")
    failures = 0
    for model in args.models:
        print(f"=== model: {model} " + "=" * 40)
        for s in SUGGESTIONS:
            try:
                res = infer(args.backend, s["text"], model, args.timeout)
            except urllib.error.HTTPError as e:
                print(f"  ERROR  [{s['kind']:16}] HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:120]}")
                failures += 1
                continue
            except Exception as e:
                print(f"  ERROR  [{s['kind']:16}] {type(e).__name__}: {e}")
                failures += 1
                continue
            agg = res.get("aggregate", {}) or {}
            tier = agg.get("tier")
            fused = agg.get("fused")
            ok = expected_ok(s["expect"], tier or "")
            failures += 0 if ok else 1
            fused_s = f"{fused:.3f}" if isinstance(fused, (int, float)) else str(fused)
            want = "low/ok" if s["expect"] == "ok" else "high/med"
            print(f"  {'PASS' if ok else 'FAIL'}  [{s['kind']:16}] tier={str(tier):7} fused={fused_s:7} "
                  f"(want {want})  | {s['text'][:50]}")
        print()

    total = len(args.models) * len(SUGGESTIONS)
    print(f"{'ALL PASS' if failures == 0 else str(failures) + ' FAILURE(S)'} — {total - failures}/{total} ok")
    if failures:
        print("\nSwap any failing suggestion for a fallback (see plan / docs) and re-run, OR tune the model's\n"
              "fusion thresholds (tools/train_claim_fusion.py --t_med --t_high).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
