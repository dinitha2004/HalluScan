"""HallKing shared LLM engine.

Loads Llama-3.1-8B ONCE in 4-bit NF4 (eager attention so HalluShift can read
attentions) and exposes everything the three detector adapters need from a SINGLE
shared generation:

  * `generate(question)`  -> one greedy "most-likely" answer + the full
    `generate(...)` output (hidden_states / attentions / logits) that HalluShift
    consumes, plus the token sequence SEP re-reads for its SLT probe.
  * `forward_hidden_states(ids)` -> a clean forward pass (steering OFF) used by SEP.
  * `tsv_steering(...)` -> a context manager that installs the TSV steering layer at
    layer 9 ONLY for the duration of the TSV forward pass, then removes it, so SEP /
    HalluShift always see the unmodified model (zero contamination).

Design choices (see plan + the three papers):
  * Prompt = `"Answer the question concisely. Q: {question} A:"` — the exact format
    HalluShift and TSV trained on; SEP's probe reads the answer's hidden states and
    generalises across prompts.
  * Greedy decoding, `max_new_tokens=64` for the QA benchmarks (matches HalluShift /
    TSV); larger for the long-form demo.
  * Shared model variant = `-Instruct` (SEP backend + HalluShift use it). TSV's
    checkpoint was trained on the base model; it is applied cross-variant here and the
    fusion meta-classifier reweights it (a retrain-on-Instruct is a documented option).

Runs in the existing `se_probes_env` (transformers 5.x, torch cu118, bitsandbytes).
"""
import os
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# llm_layers.py is copied verbatim from the TSV repo (steering injection mechanics).
from llm_layers import get_layers, LlamaDecoderLayerWrapper, TSVLayer

DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
GEN_PROMPT = "Answer the question concisely. Q: {question} A:"

# Boundaries used to isolate the concise answer from any "Q: ... A: ..." rambling the
# model may append (mirrors the HalluShift `clean_answer` fix used to build the labels).
_STOP_MARKERS = ["\nQ:", "\nQuestion:", " Q:", "Q:", "Answer the question", "\n\n"]


def clean_answer(text: str) -> str:
    """Truncate a raw generation at the first rambling boundary; return the concise answer."""
    cut = len(text)
    for m in _STOP_MARKERS:
        i = text.find(m)
        if i != -1:
            cut = min(cut, i)
    return text[:cut].strip()


class HallKingEngine:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, load_in_4bit: bool = True,
                 device: str = "cuda", hf_token: bool | str = True, fp16_nonquant: bool = False):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device = device if torch.cuda.is_available() else "cpu"
        self.hf_token = hf_token
        # fp16_nonquant: if False (default) the non-quantized modules stay in the model's native
        # dtype (bfloat16 for Llama-3.1) — this is what the SEP/HalluShift artifacts were trained
        # with (verified: bit-identical features). Set True ONLY for the TSV base pass, whose
        # steering layer casts to .half() and whose training used float16.
        self.fp16_nonquant = fp16_nonquant
        self.model = None
        self.tokenizer = None
        self.num_layers = None

    # ------------------------------------------------------------------ loading
    def load(self):
        print(f"[HallKing] loading {self.model_name} (4bit={self.load_in_4bit}) ...")
        kwargs = {}
        if not torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float32
        elif self.fp16_nonquant:
            # float16 non-quant modules — required for the TSV base pass (TSVLayer casts to .half()).
            kwargs["torch_dtype"] = torch.float16
        # else: leave torch_dtype unset -> native bfloat16 residual stream, which exactly matches
        # how the SEP/HalluShift artifacts were produced (verified: bit-identical features).
        if self.load_in_4bit and torch.cuda.is_available():
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        # use_fast=False matches the SEP backend and the HalluShift run (tokenization affects the
        # hidden states the frozen detectors read).
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token,
                                                       use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            token=self.hf_token,
            attn_implementation="eager",   # required for output_attentions (HalluShift)
            **kwargs,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.num_layers = len(self.model.model.layers)  # 32 for Llama-3.1-8B
        print(f"[HallKing] model ready | num_layers={self.num_layers} | device={self.model.device}")
        return self

    def unload(self):
        """Free the model from VRAM (used to swap between the Instruct and base passes)."""
        import gc
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[HallKing] model unloaded.")

    # -------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, question: str, max_new_tokens: int = 64) -> dict:
        """One shared greedy generation. Returns everything the 3 adapters need.

        The returned `gen_output` is the raw `model.generate(...)` ModelOutput with
        per-step hidden_states / attentions / logits (consumed by HalluShift as-is).
        """
        prompt = GEN_PROMPT.format(question=question)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = enc["input_ids"].shape[1]

        gen = self.model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions=True,
            output_logits=True,
        )
        sequences = gen.sequences                      # (1, prompt_len + n_new)
        new_ids = sequences[0, prompt_len:]
        answer_full = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        answer_clean = clean_answer(answer_full)

        return {
            "question": question,
            "prompt": prompt,
            "prompt_len": prompt_len,
            "sequences": sequences,        # SEP re-reads this for its SLT probe
            "answer_full": answer_full,    # HalluShift features come from the full generation
            "answer_clean": answer_clean,  # display + BLEURT GT + the "A" TSV scores
            "gen_output": gen,             # HalluShift consumes .hidden_states/.attentions/.logits
        }

    DEMO_SYSTEM = "You are a helpful assistant. Answer the user's question accurately and concisely."

    # Option-B TRAINING regime: force a SINGLE factual sentence so each cached example is exactly one fact
    # with a clean reference-match label. train_claim_heads / gen_and_cache generate with THIS prompt.
    SENTENCE_SYSTEM = ("Answer the question directly in a single complete sentence. "
                       "State the fact only — no preamble, no lists, no extra sentences.")

    # Option-B DEMO regime: answer at NATURAL length (a one-fact question gets one sentence; an open-ended
    # question gets several), with each distinct fact in its OWN separate sentence so the live demo can
    # defragment the answer (segment -> claim-filter -> score per sentence). The one-fact-per-sentence
    # structure (NOT a length target) is what keeps each segmented sentence close to the single-fact training
    # unit; without it the model packs many facts into one compound sentence that can't be split. "Only the
    # facts that answer the question / no padding" stops it elaborating on simple questions. TSV re-scores each
    # sentence independently (prompt-agnostic); SEP/HalluShift read this generation, so per-sentence scores
    # carry a mild prompt shift and are indicative.
    DEMO_FACTUAL_SYSTEM = ("Answer the question directly and accurately. Put each distinct fact in its own "
                           "separate sentence, and include only the facts that answer the question — "
                           "no preamble, no lists, no padding.")

    @torch.no_grad()
    def generate_sentence(self, question: str, max_new_tokens: int = 64,
                          system: str = SENTENCE_SYSTEM) -> dict:
        """Sentence-regime generation (Option B): one natural, single-sentence answer via the Instruct
        chat template. Same dict shape as generate(); the unit of training AND inference is this sentence."""
        return self.generate_chat(question, max_new_tokens=max_new_tokens, system=system)

    @torch.no_grad()
    def generate_demo(self, question: str, max_new_tokens: int = 256,
                      system: str = DEMO_FACTUAL_SYSTEM) -> dict:
        """Demo-regime generation (Option B live): a full, possibly multi-sentence answer with one fact per
        separate sentence so it can be defragmented + scored per sentence. Same dict shape as generate()."""
        return self.generate_chat(question, max_new_tokens=max_new_tokens, system=system)

    @torch.no_grad()
    def generate_chat(self, question: str, max_new_tokens: int = 256, system: str = DEMO_SYSTEM) -> dict:
        """CHAT generation for the live demo — uses the Instruct chat template so the model gives ONE
        natural answer and stops at the end-of-turn token (instead of continuing the base-style
        'Q:.. A:..' prompt and inventing a fake transcript). Returns the same dict shape as generate().

        The benchmark `generate()` keeps the QA prompt the detector heads were trained on (so the eval
        AUROCs stay valid); the demo uses this chat path and the detector scores are indicative.
        """
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": question}]
            templated = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        except Exception as e:
            print(f"[HallKing] chat template unavailable ({e}); falling back to QA prompt", flush=True)
            return self.generate(question, max_new_tokens=max_new_tokens)

        # apply_chat_template may return a plain tensor OR a BatchEncoding depending on tokenizer/version.
        if isinstance(templated, torch.Tensor):
            enc = {"input_ids": templated}
        else:
            enc = {k: v for k, v in templated.items() if k in ("input_ids", "attention_mask")}
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_ids = sorted({i for i in (self.tokenizer.eos_token_id, eot)
                          if isinstance(i, int) and i >= 0})

        gen = self.model.generate(
            **enc,
            do_sample=False, max_new_tokens=max_new_tokens,
            eos_token_id=eos_ids or self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True, output_hidden_states=True,
            output_attentions=True, output_logits=True,
        )
        sequences = gen.sequences
        answer = self.tokenizer.decode(sequences[0, prompt_len:], skip_special_tokens=True).strip()
        return {
            "question": question,
            "prompt": "(instruct chat template)",
            "prompt_len": prompt_len,
            "sequences": sequences,
            "answer_full": answer,
            "answer_clean": answer,
            "gen_output": gen,
        }

    @torch.no_grad()
    def chat(self, question: str, system: str = None, max_new_tokens: int = 8) -> str:
        """Lean chat completion that returns ONLY the decoded text (no hidden_states/attentions) —
        used by the LLM-as-judge labeler, where requesting per-token states over long evidence prompts
        would be wasteful/OOM-prone. Greedy, Instruct chat template."""
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": question}]
            templated = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        except Exception:
            return ""
        if isinstance(templated, torch.Tensor):
            enc = {"input_ids": templated}
        else:
            enc = {k: v for k, v in templated.items() if k in ("input_ids", "attention_mask")}
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
        prompt_len = enc["input_ids"].shape[1]
        eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_ids = sorted({i for i in (self.tokenizer.eos_token_id, eot) if isinstance(i, int) and i >= 0})
        out = self.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                  eos_token_id=eos_ids or self.tokenizer.eos_token_id,
                                  pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def forward_hidden_states(self, input_ids: torch.Tensor):
        """Clean forward (steering OFF) returning the hidden_states tuple. Used by SEP."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.model.device)
        out = self.model(input_ids=input_ids,
                         attention_mask=torch.ones_like(input_ids),
                         output_hidden_states=True)
        return out.hidden_states

    # ----------------------------------------------------------- TSV steering
    @contextmanager
    def tsv_steering(self, tsv_vec: torch.Tensor, lam: float, str_layer: int):
        """Install the TSV steering layer at `str_layer` for the duration of the block,
        then remove it. SEP / HalluShift therefore never see a steered model."""
        layers = get_layers(self.model)
        original = layers[str_layer]
        tsv_t = tsv_vec.to(self.model.device).half()
        layers[str_layer] = LlamaDecoderLayerWrapper(
            original, TSVLayer(tsv_t, [lam]), self.model_name)
        try:
            yield
        finally:
            layers[str_layer] = original
