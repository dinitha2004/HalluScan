"""Load a trained TSV detector (best_checkpoint.pt) and score QA pairs for hallucination.

The trained "model" is tiny: a per-layer steering vector (only `str_layer` is non-zero) plus
two centroids in the LLM's last-layer representation space. At inference we inject the steering
vector into the frozen LLM, take the last-token representation, and compare (cosine) to the two
centroids -> P(truthful). Hallucination score = 1 - P(truthful).

Usage (in the MAIN env, transformers 5.x):
    from tsv_detector import TSVDetector
    det = TSVDetector("TSV_llama3.1-8B_tqa/exemplar_num_32_num_selected_data_128/res/9/5/best_checkpoint.pt")
    print(det.hallucination_score("What is the capital of France?", " Berlin."))   # high
    print(det.hallucination_score("What is the capital of France?", " Paris."))    # low

Or from the command line:
    python tsv_detector.py --ckpt <path> --question "..." --answer "..."
"""
import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_layers import add_tsv_layers

HF_NAMES = {
    "llama3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    "qwen2.5-7B": "Qwen/Qwen2.5-7B",
}


class TSVDetector:
    def __init__(self, ckpt_path, load_in_4bit=True, device="cuda"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.cfg = ckpt
        self.device = device
        self.cos_temp = ckpt.get("cos_temp", 0.1)

        name = HF_NAMES[ckpt["model_name"]]
        kwargs = {}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        else:
            kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            name, low_cpu_mem_usage=True, device_map="auto", token=True, **kwargs)
        self.tok = AutoTokenizer.from_pretrained(name, token=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.config.pad_token_id = self.tok.pad_token_id
        for p in self.model.parameters():
            p.requires_grad = False

        # Rebuild and inject the trained steering vector at the same layer/component.
        tsv = nn.ParameterList([nn.Parameter(t.clone()) for t in ckpt["tsv"]])
        tsv.to(device).half()
        args = SimpleNamespace(component=ckpt["component"], str_layer=ckpt["str_layer"],
                               model_name=ckpt["model_name"], lam=ckpt["lam"])
        add_tsv_layers(self.model, tsv, [ckpt["lam"]], args)

        # Centroids: row 0 = hallucinated, row 1 = truthful.
        self.centroids = F.normalize(ckpt["centroids"].float().to(device), p=2, dim=-1)
        self.model.eval()
        print(f"Loaded TSV detector | model={ckpt['model_name']} layer={ckpt['str_layer']} "
              f"lam={ckpt['lam']} | trained AUROC={ckpt.get('best_test_auroc', float('nan')):.4f}")

    @torch.no_grad()
    def _last_token_rep(self, prompt):
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.model(ids, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :]  # [hidden_size], last (non-padded) token

    @torch.no_grad()
    def prob_truthful(self, question, answer):
        prompt = f"Answer the question concisely. Q: {question} A:{answer}"
        rep = F.normalize(self._last_token_rep(prompt).float(), p=2, dim=-1)
        sims = torch.matmul(rep, self.centroids.T) / self.cos_temp  # [2]
        return torch.softmax(sims, dim=-1)[1].item()                # P(class 1 = truthful)

    def hallucination_score(self, question, answer):
        """Higher => more likely hallucinated. In [0, 1]."""
        return 1.0 - self.prob_truthful(question, answer)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best_checkpoint.pt")
    ap.add_argument("--question", required=True)
    ap.add_argument("--answer", required=True)
    ap.add_argument("--load_in_8bit", action="store_true", help="load base model in 8-bit instead of 4-bit")
    args = ap.parse_args()

    det = TSVDetector(args.ckpt, load_in_4bit=not args.load_in_8bit)
    h = det.hallucination_score(args.question, args.answer)
    print(f"\nQ: {args.question}\nA: {args.answer}")
    print(f"P(truthful)         = {1 - h:.4f}")
    print(f"hallucination_score = {h:.4f}  ->  {'HALLUCINATION' if h > 0.5 else 'truthful'}")
