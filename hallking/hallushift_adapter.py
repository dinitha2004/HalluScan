"""HalluShift adapter.

Wraps the trained HalluShift membership classifier (`CombinedNN`, 71-dim input) as a
FROZEN scorer. Reuses the copied `functions.py` / `classifier.py` VERBATIM so the
feature order/engineering exactly matches what the saved `.pth` + `StandardScaler`
expect:

  feature vector (71) =
     [15 Wasserstein_hidden, 15 cosine_hidden, 15 Wasserstein_attn, 15 cosine_attn,
      mtp, Mps, norm_entropy_max, norm_entropy_min, low_prob_count_max, low_prob_count_min,
      mean_grad_max, mean_grad_min, p25_max, p50_max, p75_max]

Output = sigmoid(CombinedNN(scaler.transform(vec)))  ->  P(hallucination), higher = more hallucinated.
Features come from the FULL `model.generate(...)` output (hidden_states/attentions/logits),
exactly as the HalluShift pipeline produced them.
"""
import os
import pickle

import numpy as np
import torch

from classifier import CombinedNN
from functions import (plot_internal_state_2, probability_function,
                       normalized_entropy, count_low_probs, mean_gradient, percentile)

_ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "hallushift")


class HalluShiftAdapter:
    def __init__(self, engine, dataset: str = "truthfulqa"):
        self.engine = engine
        self.dataset = dataset
        self.model = None
        self.scaler = None
        self.num_layers = None

    def load(self, dataset: str = None, model_path: str = None, scaler_path: str = None):
        if dataset:
            self.dataset = dataset
        self.num_layers = self.engine.num_layers  # 32
        model_path = model_path or os.path.join(_ART, f"hal_det_llama3_8B_{self.dataset}_model.pth")
        scaler_path = scaler_path or os.path.join(_ART, f"hal_det_llama3_8B_{self.dataset}_scaler.pkl")
        sd = torch.load(model_path, map_location=self.engine.model.device, weights_only=True)
        self.model = CombinedNN(self.num_layers).to(self.engine.model.device)
        self.model.load_state_dict(sd)
        self.model.eval()
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        print(f"[HalluShift] loaded {os.path.basename(model_path)} (num_layers={self.num_layers})")
        return self

    # --------------------------------------------------------- feature engineering
    @staticmethod
    def _engineer(hidden, attn, probs):
        """Replicates functions.data_preparation for a single generation -> 71-dim vector."""
        max_list, min_list = probs[0], probs[1]
        mtp = min(max_list)
        mps = max(a - b for a, b in zip(max_list, min_list))
        engineered = [
            mtp, mps,
            normalized_entropy(max_list), normalized_entropy(min_list),
            count_low_probs(max_list, 0.1), count_low_probs(min_list, 0.1),
            mean_gradient(max_list), mean_gradient(min_list),
            percentile(max_list, 25), percentile(max_list, 50), percentile(max_list, 75),
        ]
        return list(hidden) + list(attn) + engineered  # 30 + 30 + 11 = 71

    def _features_from_gen(self, gen):
        nl = self.num_layers
        hidden = plot_internal_state_2(gen, nl)                  # 30 (W_hidden + cos_hidden)
        attn = plot_internal_state_2(gen, nl, state="attention")  # 30 (W_attn + cos_attn)
        probs = probability_function(gen)                         # [max_prob_list, min_prob_list]
        return self._engineer(hidden, attn, probs)

    def _predict(self, vec):
        x = self.scaler.transform(np.asarray(vec, dtype=np.float64).reshape(1, -1))
        xt = torch.tensor(x, dtype=torch.float32, device=self.engine.model.device)
        with torch.no_grad():
            logit = self.model(xt)
            p = torch.sigmoid(logit).item()
        return p

    # ----------------------------------------------------------- answer level
    def features(self, gen_result) -> np.ndarray:
        """The raw 71-dim feature vector (for feature-level fusion)."""
        return np.asarray(self._features_from_gen(gen_result["gen_output"]), dtype=np.float64)

    def score(self, gen_result) -> dict:
        p = self._predict(self._features_from_gen(gen_result["gen_output"]))
        return {"hallushift": float(p)}  # higher => more hallucinated

    # --------------------------------------------------------- per-sentence
    def score_sentences(self, gen_result, step_ranges) -> list:
        """Per-sentence HalluShift scores for long-output localization.

        `step_ranges[i] = (start_step, end_step)` are generation-step indices (relative
        to the first generated token) covered by sentence i. We build a lightweight view
        of the generate output restricted to those steps and re-run the same extraction.
        Sentences with <2 steps fall back to the answer-level score.
        """
        from types import SimpleNamespace
        gen = gen_result["gen_output"]
        answer_level = self.score(gen_result)["hallushift"]
        out = []
        for (a, b) in step_ranges:
            if gen.hidden_states is None or b - a < 2:
                out.append(answer_level)
                continue
            view = SimpleNamespace(
                hidden_states=gen.hidden_states[a:b],
                attentions=gen.attentions[a:b],
                logits=gen.logits[a:b],
            )
            try:
                out.append(float(self._predict(self._features_from_gen(view))))
            except Exception:
                out.append(answer_level)
        return out
