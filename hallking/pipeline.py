"""HallKing pipeline — single entry point that orchestrates the shared generation and
the three frozen detectors, then (optionally) the fusion meta-classifier.

    pipe = HallKingPipeline(dataset="truthfulqa").load()
    out  = pipe.score("What is the capital of France?")
    # out -> {answer, sep_entropy, sep_accuracy, hallushift, tsv, tsv_margin, [fused]}

Model-variant handling (verified empirically):
  * SEP + HalluShift run on the -Instruct model in fp16 (the config the heads were re-fit on).
  * TSV is now ALSO re-trained on the Instruct model (notebook 1b), so the recommended demo path is
    `separate_tsv=False, retrained=True`: ONE Instruct model serves all three detectors (~6 GB, fits a
    12 GB GPU / Colab T4) and scores any question in seconds. (The legacy two-model path — base-trained TSV
    on a second 8B via `separate_tsv=True` — is kept only for the original frozen checkpoint.)

`score_row` returns a flat dict suitable for a pandas DataFrame.
`score_with_sentences` returns the live-demo payload: answer + aggregate risk + per-sentence detail.
"""
import numpy as np

from engine import HallKingEngine, DEFAULT_MODEL_NAME
from sep_adapter import SEPAdapter
from hallushift_adapter import HalluShiftAdapter
from tsv_adapter import TSVAdapter

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
# Order of the score-level fusion features (keep stable across train/eval).
SCORE_FEATURES = ["sep_entropy", "sep_accuracy", "hallushift", "tsv_margin"]


class HallKingPipeline:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, dataset: str = "truthfulqa",
                 sep_probe: str = "llama3-triviaqa", load_in_4bit: bool = True, hf_token=True,
                 separate_tsv: bool = True, tsv_model: str = BASE_MODEL, retrained: bool = False,
                 sentence_tag: str = None, hs_tag: str = None):
        # sentence_tag (Option B): load the per-sentence heads trained by tools/train_claim_heads.py
        # (artifacts suffixed `_sentence_<tag>`) and generate in the sentence regime. Heads are fp16,
        # like the Option-A retrained heads. Default None keeps the Option-A short-QA path untouched.
        # hs_tag: load ONLY the HalluShift head from a DIFFERENT sentence tag (SEP/TSV/fusion stay on
        # sentence_tag). Used to observe a retrained HalluShift head live without touching the served
        # TSV-only fusion. Defaults to sentence_tag (no change).
        self.sentence_tag = sentence_tag
        self.hs_tag = hs_tag
        self.retrained = retrained or (sentence_tag is not None)
        self.engine = HallKingEngine(model_name=model_name, load_in_4bit=load_in_4bit,
                                     hf_token=hf_token, fp16_nonquant=self.retrained)
        self.dataset = dataset
        self.sep_probe = "retrained" if retrained else sep_probe
        self.separate_tsv = separate_tsv
        self.tsv_model = tsv_model
        self.tsv_engine = None
        self.sep = None
        self.hs = None
        self.tsv = None
        self.fusion = None  # optional FusionModel, attached after training
        # Risk thresholds for tier()/risk_label(). Default to risk.py (the calibrated Option-A 0.50/0.74);
        # the backend overrides these from fusion_claim_<tag>_thresholds.json when serving Option B (the
        # sentence fusion lives on a much smaller score scale, e.g. 0.039/0.066 — see plan §2).
        import risk
        self.t_med = risk.T_MED
        self.t_high = risk.T_HIGH

    def load(self):
        import os
        self.engine.load()
        art = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
        if self.sentence_tag:
            tag = self.sentence_tag
            self.sep = SEPAdapter(self.engine, probe_path=os.path.join(art, "sep", f"probes_sentence_{tag}.pkl"),
                                  probe_name=f"sentence_{tag}").load()
            hs_tag = self.hs_tag or tag   # HalluShift can come from a different tag (display/eval only)
            if hs_tag != tag:
                print(f"[HallKing] HalluShift head overridden to tag '{hs_tag}' "
                      f"(SEP/TSV/fusion stay on '{tag}')", flush=True)
            self.hs = HalluShiftAdapter(self.engine, dataset=self.dataset).load(
                model_path=os.path.join(art, "hallushift", f"hal_det_sentence_{hs_tag}_model.pth"),
                scaler_path=os.path.join(art, "hallushift", f"hal_det_sentence_{hs_tag}_scaler.pkl"))
        elif self.retrained:
            self.sep = SEPAdapter(self.engine, probe_path=os.path.join(art, "sep", "probes_retrained.pkl"),
                                  probe_name="retrained").load()
            self.hs = HalluShiftAdapter(self.engine, dataset=self.dataset).load(
                model_path=os.path.join(art, "hallushift", f"hal_det_retrained_{self.dataset}_model.pth"),
                scaler_path=os.path.join(art, "hallushift", f"hal_det_retrained_{self.dataset}_scaler.pkl"))
        else:
            self.sep = SEPAdapter(self.engine, probe_name=self.sep_probe).load()
            self.hs = HalluShiftAdapter(self.engine, dataset=self.dataset).load()
        if self.sentence_tag:
            tsv_ckpt = os.path.join(art, "tsv", f"best_checkpoint_sentence_{self.sentence_tag}.pt")
        elif self.retrained:
            tsv_ckpt = os.path.join(art, "tsv", "best_checkpoint_retrained.pt")
        else:
            tsv_ckpt = os.path.join(art, "tsv", "best_checkpoint.pt")
        if self.separate_tsv:
            self.tsv_engine = HallKingEngine(model_name=self.tsv_model, fp16_nonquant=True).load()
            self.tsv = TSVAdapter(self.tsv_engine, ckpt_path=tsv_ckpt).load()
        else:
            self.tsv = TSVAdapter(self.engine, ckpt_path=tsv_ckpt).load()
        return self

    # ------------------------------------------------------------- scoring
    def score_row(self, question: str, max_new_tokens: int = 64, with_hs_features: bool = False) -> dict:
        gen = self.engine.generate(question, max_new_tokens=max_new_tokens)
        row = {"question": question, "answer": gen["answer_clean"], "answer_full": gen["answer_full"]}
        row.update(self.sep.score(gen))      # sep_entropy, sep_accuracy, sep_hallucination
        row.update(self.hs.score(gen))       # hallushift
        row.update(self.tsv.score_qa(question, gen["answer_clean"]))  # tsv, tsv_margin
        if with_hs_features:
            for j, v in enumerate(self.hs.features(gen)):
                row[f"hs_feat_{j:02d}"] = float(v)
        if self.fusion is not None:
            row["fused"] = float(self.fusion.predict_proba_row(row))
        return row

    def score(self, question: str, max_new_tokens: int = 64) -> dict:
        return self.score_row(question, max_new_tokens=max_new_tokens)

    # -------------------------------------------------- live demo (answer + per-sentence)
    def score_with_sentences(self, question: str, max_new_tokens: int = 64,
                             use_claim_filter: bool = True) -> dict:
        """ONE generation -> calibrated answer-level risk + per-sentence localization.

        Returns:
          {
            "answer": str,
            "aggregate": {fused, tier, label, sep_entropy, hallushift, tsv_margin, n_flagged, n_sentences},
            "sentences": [{sentence, is_claim, fused, tier, sep_entropy, sep_accuracy, hallushift, tsv_margin}],
          }
        Fillers (non-claim sentences) carry fused=None / tier='filler' and are never flagged.

        Regime depends on which heads are loaded:
          * Option B (sentence_tag set) — the live demo: generate a natural answer (DEMO_FACTUAL_SYSTEM,
            one fact per short sentence), segment -> claim-filter -> score EACH claim sentence with the
            per-sentence heads. The headline aggregate is the WORST claim sentence (see
            `_aggregate_from_sentences`), so long answers no longer over-flag as a single blob.
          * Option A (sentence_tag=None) — legacy short-QA: answer-level scoring over the whole short
            answer (held-out TriviaQA AUROC 0.83); long free-form answers are out-of-distribution.
            `clean_answer` strips any 'Q:..' continuation so the segmented answer is ramble-free.
        """
        from localize import localize
        from risk import tier as risk_tier, risk_label

        if self.sentence_tag:
            # Option B: generate a natural answer with one fact per short sentence (DEMO_FACTUAL_SYSTEM),
            # then defragment -> claim-filter -> score EACH claim sentence. The headline comes from the
            # sentences, not a single score over the whole blob (which is what over-flagged long answers).
            gen = self.engine.generate_demo(question, max_new_tokens=max_new_tokens)
            answer = gen["answer_full"]
            loc = localize(self, question, use_claim_filter=use_claim_filter, gen=gen, answer=answer)
            sentences = loc["sentences"]
            aggregate = self._aggregate_from_sentences(sentences)
            return {"answer": answer, "aggregate": aggregate, "sentences": sentences}

        # Option A (legacy short-QA): calibrated answer-level scoring over the whole short answer.
        gen = self.engine.generate(question, max_new_tokens=max_new_tokens)
        answer = gen["answer_clean"]
        agg = {}
        agg.update(self.sep.score(gen))                       # sep_entropy, sep_accuracy, sep_hallucination
        agg.update(self.hs.score(gen))                        # hallushift
        agg.update(self.tsv.score_qa(question, answer))       # tsv, tsv_margin (short answer -> in-distribution)
        fused = float(self.fusion.predict_proba_row(agg)) if self.fusion is not None else None

        # per-sentence on the CONCISE clean answer (segment -> claim-filter -> score claims only)
        loc = localize(self, question, use_claim_filter=use_claim_filter, gen=gen, answer=answer)
        sentences = loc["sentences"]
        n_flagged = sum(1 for s in sentences if s["tier"] == "high")

        aggregate = {
            "fused": fused,
            "tier": risk_tier(fused, self.t_med, self.t_high) if fused is not None else "filler",
            "label": risk_label(fused, self.t_med, self.t_high) if fused is not None else "Not a claim",
            "sep_entropy": float(agg["sep_entropy"]), "hallushift": float(agg["hallushift"]),
            "tsv_margin": float(agg["tsv_margin"]),
            "n_flagged": int(n_flagged), "n_sentences": len(sentences),
        }
        return {"answer": answer, "aggregate": aggregate, "sentences": sentences}

    def _aggregate_from_sentences(self, sentences: list) -> dict:
        """Option-B headline: the answer-level risk = the SECOND-worst claim sentence, NOT the single worst.
        This makes the headline robust to one stray flagged sentence (a single suspicious claim makes the
        answer 'Uncertain', not 'Likely Hallucinated' — it takes two+ to drive it 'high'), while every bad
        sentence still lights up individually. Single-claim answers use that one claim. Sub-scores are averaged
        over claim sentences. No scored claims -> filler (e.g. a pure refusal / non-claim answer)."""
        from risk import tier as risk_tier, risk_label
        claims = [s for s in sentences if s.get("fused") is not None]
        n_flagged = sum(1 for s in sentences if s["tier"] == "high")
        if not claims:
            return {"fused": None, "tier": "filler", "label": "Not a claim",
                    "sep_entropy": 0.0, "hallushift": 0.0, "tsv_margin": 0.0,
                    "n_flagged": 0, "n_sentences": len(sentences)}
        fused_desc = sorted((s["fused"] for s in claims), reverse=True)
        fused = fused_desc[1] if len(fused_desc) >= 2 else fused_desc[0]   # 2nd-worst (robust headline)
        mean = lambda k: float(sum(s[k] for s in claims) / len(claims))
        return {
            "fused": float(fused),
            "tier": risk_tier(fused, self.t_med, self.t_high),
            "label": risk_label(fused, self.t_med, self.t_high),
            "sep_entropy": mean("sep_entropy"), "hallushift": mean("hallushift"),
            "tsv_margin": mean("tsv_margin"),
            "n_flagged": int(n_flagged), "n_sentences": len(sentences),
        }

    @staticmethod
    def score_vector(row: dict) -> np.ndarray:
        return np.array([row[k] for k in SCORE_FEATURES], dtype=np.float64)
