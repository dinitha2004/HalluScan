"""Reusable two-pass dataset builder — the heavy lifting behind the notebooks.

For each question: generate ONE answer with the -Instruct model, score SEP + HalluShift
(Pass 1), then unload and score TSV on its native base model (Pass 2), then label with
BLEURT (the shared ground-truth). Returns a tidy DataFrame with one row per question:

    question, answer, sep_entropy, sep_accuracy, sep_hallucination, hallushift,
    tsv, tsv_margin, [hs_feat_00..70], bleurt, hallucination(0/1)

Only ONE 8B model is resident at a time (fits a 12 GB GPU / Colab T4). Runs in
se_probes_env; BLEURT runs in bleurt_env via gt_bleurt.
"""
import argparse
import os

import numpy as np
import pandas as pd
from datasets import load_dataset

from engine import HallKingEngine
from sep_adapter import SEPAdapter
from hallushift_adapter import HalluShiftAdapter
from tsv_adapter import TSVAdapter
from gt_bleurt import bleurt_labels, DEFAULT_BLEURT_PY

INSTRUCT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"  # TSV's native variant


def load_qa(dataset_name: str, n: int = None, offset: int = 0):
    """Returns (questions, references) where references[i] is a list of acceptable answers."""
    if dataset_name == "truthfulqa":
        ds = load_dataset("truthful_qa", "generation")["validation"]
        questions = [r["question"] for r in ds]
        refs = [list(r["correct_answers"]) + [r["best_answer"]] for r in ds]
    elif dataset_name == "triviaqa":
        # Only the 'rc' config is cached locally ('rc.nocontext' isn't); ignore the context and
        # prompt question-only, exactly like the HalluShift notebook (so de-dup order matches).
        ds = load_dataset("trivia_qa", "rc", split="validation")
        need = ((offset + (n or 0)) * 3) + 3000
        ds = ds.select(range(min(need, len(ds))))
        seen, questions, refs = set(), [], []
        for r in ds:
            qid = r["question_id"]
            if qid in seen:
                continue
            seen.add(qid)
            aliases = list(r["answer"].get("aliases", [])) + [r["answer"].get("value", "")]
            questions.append(r["question"])
            refs.append([a for a in aliases if a])
    elif dataset_name == "nq_open":
        # Natural Questions (open) — short factual answers, same regime as TriviaQA, so the
        # BLEURT>0.5 label stays VALID (unlike TruthfulQA's free-form adversarial answers).
        ds = load_dataset("nq_open", split="validation")
        questions = [r["question"] for r in ds]
        refs = [list(r["answer"]) for r in ds]
    elif dataset_name == "squad":
        # SQuAD v1.1 validation — short extractive answers (also BLEURT-valid). Question-only
        # prompt (context dropped) to match the shared generation format; ambiguous ones just
        # become harder examples.
        ds = load_dataset("squad", split="validation")
        questions = [r["question"] for r in ds]
        refs = [list(dict.fromkeys(r["answers"]["text"])) for r in ds]
    else:
        raise ValueError(f"unknown dataset {dataset_name}")
    end = len(questions) if n is None else min(offset + n, len(questions))
    return questions[offset:end], refs[offset:end]


def build_dataset(dataset_name="truthfulqa", n=500, offset=0, max_new_tokens=64,
                  sep_probe="llama3-triviaqa", with_hs_features=True,
                  instruct_model=INSTRUCT_MODEL, base_model=BASE_MODEL,
                  compute_gt=True, bleurt_python=DEFAULT_BLEURT_PY, save_path=None, verbose=True):
    questions, refs = load_qa(dataset_name, n=n, offset=offset)
    if verbose:
        print(f"[run_dataset] {dataset_name}: {len(questions)} questions (offset={offset})")

    # ---------------- Pass 1: Instruct (generation + SEP + HalluShift)
    eng = HallKingEngine(model_name=instruct_model).load()
    sep = SEPAdapter(eng, probe_name=sep_probe).load()
    hs = HalluShiftAdapter(eng, dataset=dataset_name).load()
    rows = []
    for i, q in enumerate(questions):
        gen = eng.generate(q, max_new_tokens=max_new_tokens)
        r = {"question": q, "answer": gen["answer_clean"]}
        r.update(sep.score(gen))
        r.update(hs.score(gen))
        if with_hs_features:
            for j, v in enumerate(hs.features(gen)):
                r[f"hs_feat_{j:02d}"] = float(v)
        rows.append(r)
        if verbose and i % 25 == 0:
            print(f"  [pass1 {i}/{len(questions)}] A={r['answer'][:60]!r}", flush=True)
    eng.unload(); del eng, sep, hs

    # ---------------- Pass 2: base (TSV on its native model; fp16 for the steering layer)
    beng = HallKingEngine(model_name=base_model, fp16_nonquant=True).load()
    tsv = TSVAdapter(beng).load()
    for i, r in enumerate(rows):
        r.update(tsv.score_qa(r["question"], r["answer"]))
        if verbose and i % 25 == 0:
            print(f"  [pass2/tsv {i}/{len(rows)}] tsv_margin={r['tsv_margin']:.4f}", flush=True)
    beng.unload(); del beng, tsv

    df = pd.DataFrame(rows)

    # ---------------- Ground truth (BLEURT in bleurt_env)
    if compute_gt:
        labels, scores = bleurt_labels(df["answer"].tolist(), refs, threshold=0.5,
                                       bleurt_python=bleurt_python)
        df["bleurt"] = scores
        df["hallucination"] = labels
        if verbose:
            print(f"[run_dataset] label balance: truthful={int((labels==0).sum())} "
                  f"hallucinated={int(labels.sum())}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path)
        if verbose:
            print(f"[run_dataset] saved {save_path}  shape={df.shape}")
    return df


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", r"D:/LLAMA CACHE/huggingface")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="truthfulqa", choices=["truthfulqa", "triviaqa"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "data", f"{args.dataset}_fusion_n{args.n}.parquet")
    build_dataset(dataset_name=args.dataset, n=args.n, offset=args.offset,
                  max_new_tokens=args.max_new_tokens, save_path=out)
