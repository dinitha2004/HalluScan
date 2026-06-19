"""Build the per-sentence factuality dataset for Option B (the claim-level detector).

For each prompt the MODEL generates a free-form answer; we segment + claim-filter, extract per-sentence
SEP/HalluShift/TSV features via the SAME path the live demo uses (`localize.per_sentence_features`), and
label each CLAIM sentence by factuality:
  * Wikipedia rows  -> `method`: 'llm' (LLM-as-judge, grounded yes/no — default, robust) or 'nli'
  * QA rows         -> normalized reference-match                          (claim_label.label_by_reference_match)

We save TWO files: `data/claims_raw_<tag>.parquet` (sentence + features + EVIDENCE, before labeling) and
`data/claims_<tag>.parquet` (labeled, evidence dropped). The raw file lets you `relabel()` cheaply (re-run
ONLY the judge, no GPU re-generation) when tuning the labeling.

GPU pass (you run it). Start small to smoke-test, then scale `--n_wiki`.
Run in se_probes_env:
    python tools/build_claim_dataset.py --tag v1 --n_wiki 400 --method llm
    python tools/build_claim_dataset.py --relabel --tag v1 --method nli   # re-label only, from the raw file
"""
import argparse, os, sys
os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np, pandas as pd
from tqdm.auto import tqdm

from pipeline import HallKingPipeline
from localize import per_sentence_features
from run_dataset import load_qa
import claim_label

FEAT_COLS = ["sep_entropy", "sep_accuracy", "hallushift", "tsv_margin"]


def wiki_items(n, config, seed, min_chars, max_evidence_chars):
    """Stream Wikipedia -> [{source, prompt, entity, evidence}]. Streaming avoids the full download."""
    from datasets import load_dataset
    try:
        ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    except Exception as e:
        raise RuntimeError(f"Could not load wikimedia/wikipedia config '{config}': {e}\n"
                           f"Pass --wiki_config with a valid snapshot (e.g. 20231101.en).")
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    out = []
    for r in ds:
        if len(r.get("text", "")) < min_chars:
            continue
        out.append({"source": "wiki", "prompt": f"Write a short, factual paragraph about {r['title']}.",
                    "entity": r["title"], "evidence": r["text"][:max_evidence_chars],
                    "refs": None})
        if len(out) >= n:
            break
    return out


def qa_items(dataset, n, offset):
    qs, refs = load_qa(dataset, n=n, offset=offset)
    return [{"source": f"qa:{dataset}", "prompt": q, "entity": None, "evidence": None, "refs": list(rf)}
            for q, rf in zip(qs, refs)]


def _label_df(df, method="llm", engine=None, ent_thr=0.5, con_thr=0.5, mode="factscore",
              top_k=4, verbose=True):
    """Label a RAW claims df: wiki rows via `method` ('llm' needs a loaded `engine`; 'nli' loads DeBERTa),
    qa rows via reference-match. Returns a copy with a 'label' column (+ method-specific aux columns)."""
    df = df.copy()
    df["label"] = -1
    wiki = df["source"] == "wiki"
    if wiki.any():
        claims = df.loc[wiki, "sentence"].tolist()
        evs = df.loc[wiki, "evidence"].tolist()
        if method == "llm":
            if engine is None:
                raise ValueError("method='llm' needs a loaded engine (pass engine=pipe.engine)")
            labels, info = claim_label.label_by_llm_judge(claims, evs, engine, top_k=top_k, verbose=verbose)
            df["judge_verdict"] = ""
            df.loc[wiki, "judge_verdict"] = info["verdict"]
        elif method == "nli":
            nli = claim_label.load_nli()
            labels, info = claim_label.label_claims(claims, evs, nli, top_k=top_k,
                                                    ent_thr=ent_thr, con_thr=con_thr, mode=mode)
            df["max_entailment"] = np.nan; df["max_contradiction"] = np.nan
            df.loc[wiki, "max_entailment"] = info["max_entailment"]
            df.loc[wiki, "max_contradiction"] = info["max_contradiction"]
        else:
            raise ValueError(f"unknown method '{method}' (use 'llm' or 'nli')")
        df.loc[wiki, "label"] = labels
    qa_mask = ~wiki
    if qa_mask.any():
        df.loc[qa_mask, "label"] = claim_label.label_by_reference_match(
            df.loc[qa_mask, "sentence"].tolist(), df.loc[qa_mask, "refs"].tolist())
    return df


def _save_labeled(labeled, tag, verbose=True):
    kept = labeled[labeled["label"] >= 0].drop(columns=[c for c in ("evidence", "refs") if c in labeled],
                                               errors="ignore").reset_index(drop=True)
    out = os.path.join(ROOT, "data", f"claims_{tag}.parquet")
    kept.to_parquet(out)
    if verbose:
        print(f"[label] kept {len(kept)}/{len(labeled)} labeled claims | "
              f"halluc={int((kept.label == 1).sum())} ({kept.label.mean()*100:.1f}%) | saved {out}", flush=True)
    return kept


def build(tag="v1", n_wiki=400, wiki_config="20231101.en", min_chars=800, max_evidence_chars=8000,
          qa=None, n_qa=0, qa_offset=0, max_new_tokens=200, method="llm", ent_thr=0.5, con_thr=0.5,
          mode="factscore", top_k=4, seed=0, save=True, verbose=True):
    """Generate answers + per-sentence features (GPU) -> save RAW -> label -> save labeled.
    Callable from notebook 8 (tqdm shows live). Frees the LLM from VRAM when done."""
    items = []
    if n_wiki:
        items += wiki_items(n_wiki, wiki_config, seed, min_chars, max_evidence_chars)
    if qa and n_qa:
        items += qa_items(qa, n_qa, qa_offset)
    nw = sum(1 for it in items if it["source"] == "wiki")
    print(f"[build] {len(items)} prompts ({nw} wiki, {len(items)-nw} qa)", flush=True)
    if not items:
        raise SystemExit("no prompts — set n_wiki and/or qa/n_qa")

    # ---- GPU: generate + per-sentence features (shared with the live demo) ----
    pipe = HallKingPipeline(dataset="triviaqa", separate_tsv=False, retrained=True).load()
    pipe.fusion = None   # features only
    rows = []
    for it in tqdm(items, desc="generate + per-sentence features", unit="prompt"):
        gen = pipe.engine.generate_chat(it["prompt"], max_new_tokens=max_new_tokens)
        feat = per_sentence_features(pipe, it["prompt"], gen=gen, use_claim_filter=True)
        for r in feat["sentences"]:
            if not r["is_claim"]:
                continue
            rows.append({"source": it["source"], "prompt": it["prompt"], "entity": it["entity"],
                         "sentence": r["sentence"], **{k: r[k] for k in FEAT_COLS},
                         "evidence": it["evidence"], "refs": it["refs"]})
    df = pd.DataFrame(rows)
    print(f"[build] {len(df)} claim sentences from {len(items)} prompts", flush=True)
    if df.empty:
        pipe.engine.unload()
        raise SystemExit("no claim sentences extracted — check generation / claim filter")

    # ---- save RAW (sentence + features + evidence) so re-labeling needs no GPU re-generation ----
    if save:
        raw = os.path.join(ROOT, "data", f"claims_raw_{tag}.parquet")
        df.to_parquet(raw)
        print(f"[build] saved RAW {raw}  shape={df.shape}", flush=True)

    # ---- label (reuse the loaded engine for the LLM-judge), then free VRAM ----
    labeled = _label_df(df, method=method, engine=pipe.engine, ent_thr=ent_thr, con_thr=con_thr,
                        mode=mode, top_k=top_k, verbose=verbose)
    pipe.engine.unload()
    return _save_labeled(labeled, tag, verbose=verbose) if save else labeled


def relabel(tag="v1", method="llm", ent_thr=0.5, con_thr=0.5, mode="factscore", top_k=4,
            save=True, verbose=True):
    """Re-label data/claims_raw_<tag>.parquet WITHOUT re-generating (cheap iteration on the labeling).
    'llm' loads just the 8B engine for the judge; 'nli' loads DeBERTa (no 8B)."""
    raw = os.path.join(ROOT, "data", f"claims_raw_{tag}.parquet")
    if not os.path.exists(raw):
        raise FileNotFoundError(f"{raw} not found — run build() first to create the raw file.")
    df = pd.read_parquet(raw)
    print(f"[relabel] {len(df)} raw claims from {raw} | method={method}", flush=True)
    engine = None
    if method == "llm":
        from engine import HallKingEngine
        engine = HallKingEngine(fp16_nonquant=True).load()   # lean: engine only, no detector heads
    labeled = _label_df(df, method=method, engine=engine, ent_thr=ent_thr, con_thr=con_thr,
                        mode=mode, top_k=top_k, verbose=verbose)
    if engine is not None:
        engine.unload()
    return _save_labeled(labeled, tag, verbose=verbose) if save else labeled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--relabel", action="store_true", help="re-label from the raw file (no GPU generation)")
    ap.add_argument("--method", default="llm", choices=["llm", "nli"])
    ap.add_argument("--n_wiki", type=int, default=400)
    ap.add_argument("--wiki_config", default="20231101.en")
    ap.add_argument("--min_chars", type=int, default=800)        # skip stubs (need real evidence)
    ap.add_argument("--max_evidence_chars", type=int, default=8000)
    ap.add_argument("--qa", default=None, choices=[None, "triviaqa", "nq_open", "squad"])
    ap.add_argument("--n_qa", type=int, default=0)
    ap.add_argument("--qa_offset", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--ent_thr", type=float, default=0.5)
    ap.add_argument("--con_thr", type=float, default=0.5)
    ap.add_argument("--mode", default="factscore", choices=["factscore", "confident"])
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.relabel:
        relabel(tag=args.tag, method=args.method, ent_thr=args.ent_thr, con_thr=args.con_thr,
                mode=args.mode, top_k=args.top_k)
    else:
        d = vars(args); d.pop("relabel")
        build(**d)


if __name__ == "__main__":
    main()
