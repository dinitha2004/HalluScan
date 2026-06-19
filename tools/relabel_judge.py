"""Gate Step B/C — re-label the cached TriviaQA SENTENCE dataset with the comparative LLM-judge, WITHOUT
re-generating features, then recompute per-head AUROC against the OLD (substring) vs NEW (judge) labels.

Mirrors tools/truthfulqa_judge.py (label swap, no 8B re-generation). The head SCORES in
data/claims_<tag>.parquet don't depend on the label, so the decisive Step-C question is answered cheaply:
does each head's held-out AUROC HOLD under correct labels (genuine signal) or DROP (it had learned the
substring label noise)?

Also writes a retraining-ready corrected build (claims_sent_<newtag>.parquet + copied _sepfeats.npy) so
`train_claim_heads.train_heads(tag='<newtag>')` can refit the heads on the clean labels (no GPU re-gen).

Run in se_probes_env (needs the 8B for a short yes/no per row; ~6 new tokens each):
  python tools/relabel_judge.py --tag s1 --newtag s1j --datasets triviaqa
"""
import argparse
import os
import re
import shutil
import sys

os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "hallking"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA = os.path.join(ROOT, "data")
HEADS = [("sep_entropy", "SEP", True), ("hallushift", "HalluShift", True), ("tsv_margin", "TSV", True)]


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()


def _question_refs(datasets, n, offset):
    """{normalized question -> [gold answer strings]} from the same loader the build used."""
    from run_dataset import load_qa
    m = {}
    for ds in datasets:
        qs, rfs = load_qa(ds, n=n, offset=offset)
        for q, r in zip(qs, rfs):
            m[_norm(q)] = r
    return m


def _auroc(y, score, higher_halluc=True):
    s = np.asarray(score, dtype=float)
    if not higher_halluc:
        s = -s
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)


def relabel(tag="s1", newtag=None, datasets=("triviaqa",), n=4000, offset=0,
            instruct_model=None, mode="hybrid", save=True, verbose=True):
    newtag = newtag or (tag + "j")
    raw_path = os.path.join(DATA, f"claims_sent_{tag}.parquet")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} not found — build the dataset first.")
    raw = pd.read_parquet(raw_path).reset_index(drop=True)
    qcol = "question" if "question" in raw.columns else "prompt"

    qrefs = _question_refs(datasets, n, offset)
    refs = [qrefs.get(_norm(q), []) for q in raw[qcol]]
    cov = sum(1 for r in refs if r)
    print(f"[relabel] {len(raw)} rows | gold refs matched for {cov}/{len(raw)} questions "
          f"({'OK' if cov / max(len(raw), 1) > 0.9 else 'LOW -> judge guesses on the rest; results unreliable'})",
          flush=True)

    old_y = raw["hallucination"].to_numpy().astype(int)   # the existing substring labels
    if mode == "hybrid":
        judge_idx = np.where(old_y == 1)[0]   # only adjudicate substring-HALLUCINATED rows (rescue true ones)
        print(f"[relabel] mode=hybrid: keep {int((old_y == 0).sum())} substring-truthful rows as-is; "
              f"judge the {len(judge_idx)} substring-hallucinated rows", flush=True)
    else:
        judge_idx = np.arange(len(raw))
        print(f"[relabel] mode=full: judge all {len(raw)} rows (can introduce false positives)", flush=True)

    from engine import HallKingEngine
    from run_dataset import INSTRUCT_MODEL
    from claim_label import label_by_qa_judge
    eng = HallKingEngine(model_name=instruct_model or INSTRUCT_MODEL).load()
    jl, jinfo = label_by_qa_judge([raw[qcol].iloc[i] for i in judge_idx],
                                  [raw["answer"].iloc[i] for i in judge_idx],
                                  [refs[i] for i in judge_idx], eng, verbose=verbose)
    eng.unload()

    new_y = old_y.copy()
    verdicts = np.array(["substring-truthful"] * len(raw), dtype=object)
    for k, i in enumerate(judge_idx):
        new_y[i] = int(jl[k])
        verdicts[i] = jinfo["verdict"][k]
    flips = int((new_y != old_y).sum())
    print(f"\n[relabel] halluc rate: old(substring)={old_y.mean()*100:.1f}% -> new(judge)={new_y.mean()*100:.1f}%"
          f" | flips={flips} ({flips/len(new_y)*100:.1f}%) | agreement={(new_y == old_y).mean()*100:.1f}%")
    # direction of flips: substring's main error is true->halluc, so judge should flip many 1->0
    f_1to0 = int(((old_y == 1) & (new_y == 0)).sum())
    f_0to1 = int(((old_y == 0) & (new_y == 1)).sum())
    print(f"           flips 1->0 (was 'halluc', judge says truthful) = {f_1to0} | 0->1 = {f_0to1}")
    if verbose:
        idx = np.where((old_y == 1) & (new_y == 0))[0][:8]
        if len(idx):
            print("\n  sample 1->0 corrections (substring called these hallucinated; judge says truthful):")
            for i in idx:
                print(f"    Q: {str(raw[qcol].iloc[i])[:66]}")
                print(f"      A: {str(raw['answer'].iloc[i])[:94]}")

    if save:
        out = raw.copy()
        out["hallucination_refmatch"] = old_y
        out["hallucination"] = new_y
        out["judge_verdict"] = verdicts
        out.to_parquet(os.path.join(DATA, f"claims_sent_{newtag}.parquet"))
        src = os.path.join(DATA, f"claims_sent_{tag}_sepfeats.npy")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(DATA, f"claims_sent_{newtag}_sepfeats.npy"))
        print(f"  wrote claims_sent_{newtag}.parquet (+ copied sepfeats) "
              f"-> retrain with train_claim_heads.train_heads(tag='{newtag}')")

    # --- Step C verdict: per-head AUROC OLD vs NEW on the held-out scored table (no retrain) ---
    scored_path = os.path.join(DATA, f"claims_{tag}.parquet")
    if os.path.exists(scored_path):
        sc = pd.read_parquet(scored_path).reset_index(drop=True)
        if len(sc) == len(raw):
            mask = (sc["split"] == "test").to_numpy() if "split" in sc.columns else np.ones(len(sc), bool)
            yo, yn = old_y[mask], new_y[mask]
            print(f"\n=== per-head AUROC on TEST split (n={int(mask.sum())}): OLD(substring) vs NEW(judge) ===")
            print(f"  {'head':12} {'AUROC_old':>10} {'AUROC_new':>10}  delta")
            for col, name, hh in HEADS:
                if col not in sc.columns:
                    continue
                ao = _auroc(yo, sc[col].to_numpy()[mask], hh)
                an = _auroc(yn, sc[col].to_numpy()[mask], hh)
                print(f"  {name:12} {ao:10.3f} {an:10.3f}  {an - ao:+.3f}")
            print("  (HOLD/rise under NEW = genuine signal; large DROP = the head had learned the substring noise.)")
            if save:
                sc["label_refmatch"] = old_y
                sc["label"] = new_y
                sc["judge_verdict"] = verdicts
                sc.to_parquet(os.path.join(DATA, f"claims_{newtag}.parquet"))
        else:
            print(f"[relabel] {scored_path} not row-aligned ({len(sc)} vs {len(raw)}) — skipping AUROC recompute")
    else:
        print(f"[relabel] {os.path.relpath(scored_path, ROOT)} not found — skipping AUROC recompute")
    return new_y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s1", help="existing cached build to re-label")
    ap.add_argument("--newtag", default=None, help="output tag (default <tag>j)")
    ap.add_argument("--datasets", nargs="+", default=["triviaqa"])
    ap.add_argument("--n", type=int, default=4000, help="how many questions to load for the ref map")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "full"],
                    help="hybrid: judge only substring-hallucinated rows (safe, recommended); "
                         "full: judge every row (can introduce false positives)")
    args = ap.parse_args()
    relabel(tag=args.tag, newtag=args.newtag, datasets=tuple(args.datasets), n=args.n, offset=args.offset,
            mode=args.mode)


if __name__ == "__main__":
    main()
