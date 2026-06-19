"""Evidence-grounded factuality labels for generated claim sentences (FActScore-lite).

For each generated sentence we ask a strong NLI model whether the EVIDENCE (the entity's Wikipedia text,
or a QA reference) entails it:
  * supported   (max entailment >= ent_thr)               -> truthful   (label 0)
  * contradicted(max contradiction >= con_thr)            -> hallucinated(label 1)
  * neither (unsupported / neutral)                       -> mode-dependent (see `mode`)

Cheap retrieval: rank evidence chunks by word-overlap with the claim and NLI only the top-k (so we don't
score the whole article for every sentence). Reuses the same DeBERTa NLI as claim_filter / nb6.

`mode`:
  "factscore" (default) : unsupported -> hallucinated (1)  [full coverage, noisier on true-but-absent facts]
  "confident"           : unsupported -> None (DROP from training)  [cleaner labels, fewer rows]

NOTE (label-noise trade-off — a tuning knob): if evidence is incomplete, a TRUE claim absent from it can be
mislabelled hallucinated. Use the FULL article as evidence + "confident" mode to reduce this. The notebook
exposes ent_thr/con_thr/mode so you can re-run.
"""
import re
import numpy as np

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
_WORD = re.compile(r"[a-z0-9]+")


def load_nli(model_name: str = NLI_MODEL, device=None):
    import torch
    from transformers import pipeline as hf_pipeline
    dev = (0 if torch.cuda.is_available() else -1) if device is None else device
    print(f"[claim_label] loading NLI {model_name} (device={'cuda' if dev == 0 else 'cpu'}) ...", flush=True)
    return hf_pipeline("text-classification", model=model_name, device=dev, top_k=None,
                       truncation=True, max_length=512)


def chunk_text(text: str, words_per_chunk: int = 90, stride: int = 70):
    """Split evidence into overlapping word windows (robust, dependency-light)."""
    words = (text or "").split()
    if not words:
        return []
    chunks = []
    for s in range(0, len(words), stride):
        chunks.append(" ".join(words[s:s + words_per_chunk]))
        if s + words_per_chunk >= len(words):
            break
    return chunks


def _overlap(a_words, chunk):
    cw = set(_WORD.findall(chunk.lower()))
    return len(a_words & cw)


def label_by_reference_match(claims, references):
    """Cheap label for SHORT-answer QA rows (evidence = short reference string, too thin for NLI):
    truthful (0) iff any normalized reference is a substring of the normalized claim. references[i] is a
    list of acceptable strings for claim i."""
    def norm(s):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(s).lower())).strip()
    labels = []
    for claim, refs in zip(claims, references):
        nc = norm(claim)
        hit = any(norm(r) and norm(r) in nc for r in (refs or []))
        labels.append(0 if hit else 1)
    return np.array(labels, dtype=int)


_JUDGE_SYS = ("You are a strict fact-checker. Using ONLY the reference text, decide whether the statement "
              "is supported by it. If the reference does not clearly support the statement, answer no. "
              "Answer with a single word: yes or no.")


def label_by_llm_judge(claims, evidences, engine, top_k: int = 4, max_evidence_chars: int = 3000,
                       verbose: bool = True):
    """LLM-as-judge factuality labels (robust to paraphrase, unlike strict NLI entailment).

    For each claim we retrieve the top-k most-overlapping evidence chunks and ask the (grounded) model
    "is this statement supported by the reference text? yes/no". `engine` is a loaded HallKingEngine
    (reuses the 8B already in VRAM during the build). Returns (labels[int], info dict with the raw verdicts).
    truthful (0) iff the judge says yes; otherwise hallucinated (1).
    """
    from tqdm.auto import tqdm
    labels, verdicts = [], []
    for claim, ev in tqdm(list(zip(claims, evidences)), desc="LLM-judge", unit="claim", disable=not verbose):
        chunks = chunk_text(ev)
        if chunks:
            aw = set(_WORD.findall(claim.lower()))
            chunks = sorted(chunks, key=lambda c: _overlap(aw, c), reverse=True)[:top_k]
        evidence_text = " ".join(chunks)[:max_evidence_chars]
        q = (f"Reference text:\n{evidence_text}\n\nStatement: \"{claim}\"\n\n"
             "Is the statement supported by the reference text? Answer yes or no.")
        ans = engine.chat(q, system=_JUDGE_SYS, max_new_tokens=6).lower()
        head = ans[:12]
        yes = "yes" in head
        no = "no" in head
        labels.append(0 if (yes and not no) else 1)   # supported -> 0; else hallucinated
        verdicts.append(ans[:20])
    return np.array(labels, dtype=int), {"verdict": verdicts}


def label_claims(claims, evidences, nli_pipe, top_k: int = 4, ent_thr: float = 0.5,
                 con_thr: float = 0.5, mode: str = "factscore", batch_size: int = 32, verbose: bool = True):
    """claims: list[str]; evidences: list[str] (parallel — each claim's evidence text).
    Returns (labels[int, -1=drop], info dict with max entailment/contradiction per claim)."""
    from tqdm.auto import tqdm
    # 1. build the (premise=evidence-chunk, hypothesis=claim) pairs via cheap top-k overlap retrieval
    pairs, owners = [], []   # owners[i] = claim index for pairs[i]
    per_claim_chunks = []
    for ci, (claim, ev) in enumerate(zip(claims, evidences)):
        chunks = chunk_text(ev)
        if chunks:
            aw = set(_WORD.findall(claim.lower()))
            chunks = sorted(chunks, key=lambda c: _overlap(aw, c), reverse=True)[:top_k]
        per_claim_chunks.append(len(chunks))
        for c in chunks:
            pairs.append({"text": c, "text_pair": claim})   # premise=evidence, hypothesis=claim
            owners.append(ci)

    # 2. run NLI in batches
    ent = np.zeros(len(pairs)); con = np.zeros(len(pairs))
    for s in tqdm(range(0, len(pairs), batch_size), desc="evidence-NLI", unit="batch", disable=not verbose):
        for j, res in enumerate(nli_pipe(pairs[s:s + batch_size], batch_size=batch_size)):
            d = {r["label"].lower(): r["score"] for r in res}
            ent[s + j] = d.get("entailment", 0.0)
            con[s + j] = d.get("contradiction", 0.0)

    # 3. aggregate per claim (max entailment / contradiction over its chunks) -> label
    labels = np.full(len(claims), -1, dtype=int)
    max_ent = np.zeros(len(claims)); max_con = np.zeros(len(claims))
    k = 0
    for ci, nch in enumerate(per_claim_chunks):
        if nch == 0:
            labels[ci] = (-1 if mode == "confident" else 1)   # no evidence -> drop or call halluc
            continue
        e = ent[k:k + nch]; c = con[k:k + nch]; k += nch
        max_ent[ci] = e.max(); max_con[ci] = c.max()
        if max_ent[ci] >= ent_thr:
            labels[ci] = 0                       # supported -> truthful
        elif max_con[ci] >= con_thr:
            labels[ci] = 1                       # contradicted -> hallucinated
        else:
            labels[ci] = (-1 if mode == "confident" else 1)   # unsupported
    return labels, {"max_entailment": max_ent, "max_contradiction": max_con}
