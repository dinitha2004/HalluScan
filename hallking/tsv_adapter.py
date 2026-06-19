"""TSV (Truthfulness Separator Vector) adapter.

Wraps the trained TSV checkpoint (`best_checkpoint.pt`) as a FROZEN scorer. Replicates
the TSV detector EXACTLY, except the steering layer is installed by the engine's
`tsv_steering(...)` context manager ONLY around the forward pass (so SEP / HalluShift
never see a steered model):

  prompt = "Answer the question concisely. Q: {q} A:{answer}"
  rep    = normalize( steered_model(prompt).hidden_states[-1][0, -1, :] )
  P(truthful) = softmax( rep @ centroids^T / cos_temp )[truthful]
  hallucination = 1 - P(truthful)        # higher = more hallucinated

NOTE on model variant: the checkpoint was trained on the BASE Meta-Llama-3.1-8B; here
it is applied to the shared -Instruct model (cross-variant). The fusion meta-classifier
reweights it; a retrain-on-Instruct is a documented enhancement.
"""
import os

import torch
import torch.nn.functional as F

_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "artifacts", "tsv", "best_checkpoint.pt")
TSV_PROMPT = "Answer the question concisely. Q: {question} A:{answer}"


class TSVAdapter:
    def __init__(self, engine, ckpt_path: str = _CKPT):
        self.engine = engine
        self.ckpt_path = ckpt_path
        self.tsv_vec = None        # steering vector at str_layer (4096,)
        self.centroids = None      # (2, 4096) normalized; row0=hallucinated, row1=truthful
        self.str_layer = None
        self.lam = None
        self.cos_temp = None

    def load(self):
        ck = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.str_layer = int(ck["str_layer"])
        self.lam = float(ck["lam"])
        self.cos_temp = float(ck.get("cos_temp", 0.1))
        self.tsv_vec = ck["tsv"][self.str_layer].clone()                       # (4096,)
        dev = self.engine.model.device
        self.centroids = F.normalize(ck["centroids"].float().to(dev), p=2, dim=-1)
        print(f"[TSV] loaded checkpoint | layer={self.str_layer} lam={self.lam} "
              f"cos_temp={self.cos_temp} | trained AUROC={ck.get('best_test_auroc', float('nan')):.4f}")
        return self

    @torch.no_grad()
    def _last_token_rep(self, prompt: str):
        tok = self.engine.tokenizer
        ids = tok(prompt, return_tensors="pt").input_ids.to(self.engine.model.device)
        with self.engine.tsv_steering(self.tsv_vec, self.lam, self.str_layer):
            out = self.engine.model(ids, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :]

    def _centroid_scores(self, rep):
        """Returns (p_truthful, margin) where margin = cos(hallucinated) - cos(truthful).

        The probability uses cos_temp=0.1 and SATURATES (~0/1), destroying its usefulness for
        thresholding/fusion, but the cosine `margin` is the un-saturated, rank-equivalent signal
        (TSV's AUROC is a ranking metric). Higher margin => more hallucinated."""
        rep = F.normalize(rep.float(), p=2, dim=-1)
        cos = torch.matmul(rep, self.centroids.T)         # (2,) -> [cos_halluc, cos_truth]
        p_truth = torch.softmax(cos / self.cos_temp, dim=-1)[1].item()
        margin = float(cos[0].item() - cos[1].item())     # cos_halluc - cos_truth
        return p_truth, margin

    def _centroid_prob_truthful(self, rep):
        return self._centroid_scores(rep)[0]

    # ----------------------------------------------------------- answer level
    def score_qa(self, question: str, answer: str) -> dict:
        """Score a (question, answer) text pair. Used both from a shared gen_result and in a
        decoupled pass where TSV runs on its native base model.

        `tsv_margin` (cos_halluc - cos_truth) is the primary, un-saturated signal used for
        fusion + AUROC; `tsv` (1 - P_truthful) is the paper-style probability (saturated)."""
        if not answer.startswith(" "):
            answer = " " + answer
        prompt = TSV_PROMPT.format(question=question, answer=answer)
        p_truth, margin = self._centroid_scores(self._last_token_rep(prompt))
        return {"tsv": float(1.0 - p_truth), "tsv_margin": margin}  # higher => more hallucinated

    def score(self, gen_result) -> dict:
        return self.score_qa(gen_result["question"], gen_result["answer_clean"])

    # --------------------------------------------------------- per-sentence
    @torch.no_grad()
    def score_sentences(self, question: str, full_answer: str, sentence_char_ends) -> list:
        """Per-sentence TSV scores: one steered forward over Q + full answer, then read the
        steered last-layer hidden state at each sentence-end token position."""
        tok = self.engine.tokenizer
        ans = full_answer if full_answer.startswith(" ") else " " + full_answer
        prompt = TSV_PROMPT.format(question=question, answer=ans)
        ids = tok(prompt, return_tensors="pt").input_ids.to(self.engine.model.device)
        # char offset where the answer starts inside the prompt (to align with sentence ends)
        ans_start_char = prompt.find(ans)

        with self.engine.tsv_steering(self.tsv_vec, self.lam, self.str_layer):
            out = self.engine.model(ids, output_hidden_states=True)
        last_layer = out.hidden_states[-1][0]  # (seq, dim)

        results = [None] * len(sentence_char_ends)  # tsv_margin per sentence (higher = hallucinated)
        cur = 0
        for token_i in range(1, ids.shape[1] + 1):
            if cur >= len(sentence_char_ends):
                break
            partial = tok.decode(ids[0, :token_i], skip_special_tokens=True)
            target = ans_start_char + sentence_char_ends[cur]  # char offset in the prompt
            if len(partial) >= target:
                pos = min(token_i - 1, ids.shape[1] - 1)
                results[cur] = self._centroid_scores(last_layer[pos, :])[1]  # margin
                cur += 1
        fallback = self._centroid_scores(last_layer[-1, :])[1]
        return [r if r is not None else fallback for r in results]
