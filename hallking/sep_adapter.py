"""SEP (Semantic Entropy Probes) adapter.

Wraps the trained SEP logistic-regression probes (`Llama3-8b_inference.pkl`) as a
FROZEN scorer. Replicates the SEP backend's extraction EXACTLY:

  * take a clean forward pass over the full generated sequence,
  * stack ALL hidden-state layers (embedding + 32 = 33, dim 4096) at the
    second-last token (SLT) -> flatten to 135168,
  * `s_bmodel.predict_proba(...)[:,1]` = P(high semantic entropy)  -> higher = more hallucinated,
  * `s_amodel.predict_proba(...)[:,1]` = P(correct)                -> higher = truthful.

Also exposes a per-sentence scorer (for long-output localization) that mirrors the
SEP backend's token-to-char mapping + claim filter.
"""
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression

DEFAULT_PROBE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "artifacts", "sep", "Llama3-8b_inference.pkl")


class SEPAdapter:
    def __init__(self, engine, probe_path: str = DEFAULT_PROBE_PATH, probe_name: str = "llama3-triviaqa"):
        self.engine = engine
        self.probe_path = probe_path
        self.probe_name = probe_name
        self.probe = None

    def load(self):
        with open(self.probe_path, "rb") as f:
            probes = pickle.load(f)
        # sklearn version-compat patch (probe pickled under an older sklearn) — same as SEP backend.
        for p in probes:
            for key in ("s_bmodel", "s_amodel"):
                m = p.get(key)
                if isinstance(m, LogisticRegression) and not hasattr(m, "multi_class"):
                    m.multi_class = "auto"
        self.probe = next((p for p in probes if p.get("name") == self.probe_name), None) or probes[0]
        print(f"[SEP] loaded probe '{self.probe.get('name')}' from {os.path.basename(self.probe_path)}")
        return self

    # --------------------------------------------------------------- internals
    @staticmethod
    def _slt_stack(hidden_states, idx):
        """Stack all layers' hidden state at token `idx` -> np.array (n_layers, dim)."""
        return np.stack([h[0, idx, :].float().cpu().numpy() for h in hidden_states])

    def _run_probe(self, model, hidden_stack, layer_range):
        start, end = layer_range
        vec = hidden_stack[start:end].reshape(1, -1)
        return float(model.predict_proba(vec)[0, 1])

    # ----------------------------------------------------------- answer level
    def features(self, gen_result) -> np.ndarray:
        """The flattened all-layer SLT hidden-state vector the probe consumes (for re-fitting SEP)."""
        sequences = gen_result["sequences"]
        prompt_len = gen_result["prompt_len"]
        slt_idx = max(sequences.shape[1] - 2, prompt_len)
        hidden_states = self.engine.forward_hidden_states(sequences)
        return self._slt_stack(hidden_states, slt_idx).astype(np.float32).reshape(-1)

    def score(self, gen_result) -> dict:
        """Answer-level SEP scores from the second-last token of the full sequence."""
        sequences = gen_result["sequences"]
        prompt_len = gen_result["prompt_len"]
        seq_len = sequences.shape[1]
        slt_idx = max(seq_len - 2, prompt_len)

        hidden_states = self.engine.forward_hidden_states(sequences)  # steering OFF
        stack = self._slt_stack(hidden_states, slt_idx)

        entropy = self._run_probe(self.probe["s_bmodel"], stack, self.probe["sep_layer_range"])
        accuracy = self._run_probe(self.probe["s_amodel"], stack, self.probe["ap_layer_range"])
        return {
            "sep_entropy": entropy,    # higher => more hallucinated
            "sep_accuracy": accuracy,  # higher => truthful
            "sep_hallucination": float((entropy + (1.0 - accuracy)) / 2.0),  # convenience, higher=hallucinated
        }

    # --------------------------------------------------------- per-sentence
    def score_sentences(self, gen_result, sentences, sentence_char_ends, claim_flags):
        """Per-sentence SEP scores for long-output localization.

        `sentence_char_ends` are character offsets relative to the ANSWER text where each
        sentence ends. We add the prompt's decoded length, map to token indices by progressive
        decode (mirrors the SEP backend), then probe the SLT of each sentence. Non-claim
        sentences are skipped (entropy=0, accuracy=1).
        """
        tok = self.engine.tokenizer
        sequences = gen_result["sequences"]
        prompt_len = gen_result["prompt_len"]
        seq_len = sequences.shape[1]
        hidden_states = self.engine.forward_hidden_states(sequences)
        prompt_decode_len = len(tok.decode(sequences[0, :prompt_len], skip_special_tokens=True))
        full_ends = [prompt_decode_len + e for e in sentence_char_ends]

        results = [None] * len(sentences)
        cur = 0
        for token_i in range(prompt_len, seq_len + 1):
            if cur >= len(full_ends):
                break
            partial = tok.decode(sequences[0, :token_i], skip_special_tokens=True)
            # full prompt+answer decoded length up to this token
            if len(partial) >= full_ends[cur]:
                slt_idx = max(token_i - 2, prompt_len)
                if not claim_flags[cur]:
                    results[cur] = {"sep_entropy": 0.0, "sep_accuracy": 1.0}
                else:
                    stack = self._slt_stack(hidden_states, slt_idx)
                    results[cur] = {
                        "sep_entropy": self._run_probe(self.probe["s_bmodel"], stack, self.probe["sep_layer_range"]),
                        "sep_accuracy": self._run_probe(self.probe["s_amodel"], stack, self.probe["ap_layer_range"]),
                    }
                cur += 1
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {"sep_entropy": 0.0, "sep_accuracy": 1.0}
        return results
