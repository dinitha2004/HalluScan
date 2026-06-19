"""
Hybrid claim filter: regex fast-pass + Zero-Shot NLI (DeBERTa).

Architecture:
  1. Regex fast-pass catches obvious non-claims (fillers, greetings, meta)
  2. Ambiguous sentences go through DeBERTa NLI for intelligent classification
  3. Graceful fallback: if NLI model fails to load, regex-only mode is used

Design principles:
  - Conservative: defaults to is_claim=True (better to score a non-claim
    than to miss a real claim)
  - NLI model loads on CPU to preserve GPU for the main LLM
  - Fast and robust
"""

import re
import os
import torch
from typing import List, Optional

# ──────────── regex patterns (FAST-PASS for obvious non-claims) ────────────

# Very short filler phrases (case-insensitive exact matches after stripping)
_FILLER_PHRASES = frozenset([
    "sure", "sure!", "sure.", "sure thing",
    "of course", "of course!", "of course.",
    "absolutely", "absolutely!", "absolutely.",
    "certainly", "certainly!", "certainly.",
    "great question", "great question!",
    "good question", "good question!",
    "right", "right.", "right!",
    "okay", "okay.", "ok", "ok.",
    "yes", "yes.", "yes!", "no", "no.", "no!",
    "well", "well.", "well,",
    "thanks", "thanks!", "thank you", "thank you!",
    "you're welcome", "you're welcome!",
    "here you go", "here you go:", "here you go.",
    "i see", "i see.",
    "got it", "got it.", "got it!",
    "exactly", "exactly!", "exactly.",
    "interesting", "interesting!", "interesting.",
    "understood", "understood.",
])

# Expanded meta verbs for _META_PATTERNS
_META_VERBS = (
    r"explain|describe|elaborate|clarify|break|walk|go through|go over|"
    r"provide|give|outline|discuss|show|list|share|cover|present|"
    r"summarize|highlight|talk about|review|explore|examine|"
    r"look at|dive into|address|answer|help|start|begin"
)

# Meta / transitional sentence patterns (case-insensitive)
_META_PATTERNS = [
    re.compile(rf"^(let me|i'll|i will|i'm going to|i'd like to|allow me to|let's)\s+({_META_VERBS})\b", re.IGNORECASE),
    re.compile(r"^here'?s?\s+(what|how|a|an|the|my|some|are)", re.IGNORECASE),
    re.compile(r"^(in summary|to summarize|to sum up|in conclusion|overall|to conclude|in short|in brief)[,:\s]", re.IGNORECASE),
    re.compile(r"^(first|second|third|finally|next|additionally|furthermore|moreover|however|that said)[,:]?\s*$", re.IGNORECASE),
    re.compile(r"^(for example|for instance)[,:]?\s*$", re.IGNORECASE),
    re.compile(r"^(note|please note|keep in mind|remember|it'?s?\s+worth noting|it'?s?\s+important to note)\b", re.IGNORECASE),
    re.compile(r"^(hope this helps|i hope|does that|is there anything|if you have|let me know)\b", re.IGNORECASE),
    re.compile(r"^(as (i|we) (mentioned|discussed|noted|said|stated))\b", re.IGNORECASE),
    re.compile(r"^(feel free to|don'?t hesitate to)\b", re.IGNORECASE),
    re.compile(r"^i('ll| will| am going to)\s+\w+\s+(an overview|a (brief|quick|short|detailed|comprehensive))\b", re.IGNORECASE),
    re.compile(r"^(below (is|are)|the following (is|are|list|table|section))\b", re.IGNORECASE),
    re.compile(r"^(this|that)\s+(is a|means|shows|demonstrates|illustrates|suggests)\s+(good|great|common|typical|important)", re.IGNORECASE),
]

# Advisory / caveat / disclaimer patterns
_ADVISORY_PATTERNS = [
    re.compile(r"\b(may vary|might vary|can vary|varies|vary depending|depends on|it depends)\b", re.IGNORECASE),
    re.compile(r"\b(not necessarily|not always|not guaranteed|no guarantee|your mileage may vary)\b", re.IGNORECASE),
    re.compile(r"\b(consult|seek advice|professional advice|do your own research)\b", re.IGNORECASE),
    re.compile(r"\b(please note that|keep in mind that|bear in mind that|be aware that|be advised that)\b", re.IGNORECASE),
    re.compile(r"\b(this is not|this isn'?t)\s+(financial|medical|legal|professional)\s+(advice)\b", re.IGNORECASE),
    re.compile(r"\b(individual results|results may|experiences may|opinions may)\b", re.IGNORECASE),
]

# Hedging / opinion-framing sentences
_HEDGING_PATTERNS = [
    re.compile(r"^(in my opinion|personally|from my perspective|i think|i believe|i feel|i'd say)\b", re.IGNORECASE),
    re.compile(r"^(it'?s?\s+(hard|difficult|tricky|complicated|complex|subjective|debatable)\s+to)\b", re.IGNORECASE),
    re.compile(r"^(there'?s?\s+no\s+(one|single|simple|definitive|clear))\b", re.IGNORECASE),
]

# Refusal / "I don't know" / no-information — the model DECLINING to answer is not a factual claim,
# so we leave it alone (NOT scored, NOT flagged as a hallucination). We only judge FABRICATED facts.
_REFUSAL_PATTERNS = [
    re.compile(r"\bI\s+(could\s?n'?t|could\s+not|can'?t|can\s+not|cannot|was\s+unable\s+to|am\s+unable\s+to|was\s+not\s+able\s+to)\s+find\b", re.IGNORECASE),
    re.compile(r"\bI\s+(don'?t|do\s+not|did\s*n'?t)\s+(have|find|know|recognize)\b", re.IGNORECASE),
    re.compile(r"\bI('?m|\s+am)\s+not\s+(aware|familiar|sure|certain)\b", re.IGNORECASE),
    re.compile(r"\bI('?m|\s+am)\s+(sorry|afraid)\b.*\b(don'?t|do\s+not|cannot|can'?t|no|unable)\b", re.IGNORECASE),
    re.compile(r"\b(unfortunately,?\s+)?I\s+(don'?t|do\s+not)\s+have\s+(access|enough|sufficient|any)\b", re.IGNORECASE),
    re.compile(r"\b(there\s+(is|are)\s+no|no)\s+(information|record|records|data|details|mention)\b", re.IGNORECASE),
    re.compile(r"^(it'?s|it\s+is)\s+possible\s+that\b.*(misspelled|different\s+name|lesser-known|obscure|typo)", re.IGNORECASE),
    re.compile(r"^(if\s+you\s+(could|can)\s+provide|could\s+you\s+(provide|clarify)|please\s+provide\s+(more|additional))", re.IGNORECASE),
    re.compile(r"\b(may\s+be\s+a\s+)?(fictional|made-up|non-existent|hypothetical)\s+(character|person|place|entity|topic)\b", re.IGNORECASE),
]

def is_refusal(text: str) -> bool:
    """True if the sentence is the model DECLINING to answer ("I don't know", "I couldn't find ...").
    A refusal is not a factual claim and not a hallucination — Option-B training excludes these rows
    (we only judge fabricated facts), and the live demo leaves them unscored."""
    t = (text or "").strip()
    return any(pat.search(t) for pat in _REFUSAL_PATTERNS)


# Pure question pattern
_QUESTION_RE = re.compile(r"^(what|who|where|when|why|how|do|does|did|is|are|was|were|can|could|would|should|shall|will|have|has|had)\b.*\?$", re.IGNORECASE)

# Greeting / sign-off
_GREETING_RE = re.compile(r"^(hi|hello|hey|greetings|dear|sincerely|regards|best|cheers)\b", re.IGNORECASE)

# Label/heading pattern: short sentence ending with ":"
_LABEL_RE = re.compile(r"^[^:]{3,80}:\s*$")

# Named entity heuristic: capitalized multi-word names
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Contains numbers
_NUMBER_RE = re.compile(r"\d+")

# Factual content indicators (numbers with units, measurements, percentages, dates)
_FACTUAL_INDICATORS = [
    re.compile(r"\d+[\.,]?\d*\s*(km|m|cm|mm|miles?|feet|ft|inches?|mph|km/s|m/s|kg|g|lb|lbs|°[CF]|degrees?|percent|%|years?|months?|days?|hours?|minutes?|seconds?|GB|MB|TB|GHz|MHz|Hz)", re.IGNORECASE),
    re.compile(r"\d{2,}"),  # Numbers with 2+ digits (specific quantities)
    re.compile(r"\$[\d,.]+|\d+[\.,]\d+"),  # Currency or decimal numbers
    re.compile(r"approximately|roughly|about\s+\d", re.IGNORECASE),  # Approximate measurements
]


# ──────────── NLI Configuration ────────────

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Reduced to 2 labels for performance (2 forward passes instead of 4)
NLI_CLAIM_LABEL = "factual claim or verifiable statement"
NLI_NON_CLAIM_LABEL = "joke, opinion, creative writing, or filler"
NLI_LABELS = [NLI_CLAIM_LABEL, NLI_NON_CLAIM_LABEL]

# Threshold: sentence is a claim if "factual claim" is top AND score > this
NLI_CLAIM_THRESHOLD = 0.5


# ──────────── ClaimDetector Class ────────────

class ClaimDetector:
    """
    Hybrid claim detector: regex fast-pass + Zero-Shot NLI.
    
    Usage:
        detector = ClaimDetector()
        detector.load_nli_model()  # Optional, falls back to regex-only
        is_claim = detector.is_claim("The Earth orbits the Sun.")
    """
    
    def __init__(self):
        self.nli_pipeline = None
        self.nli_available = False
        self.nli_model_name = NLI_MODEL_NAME
        
    def load_nli_model(self):
        """Load the zero-shot NLI model on GPU (CUDA). Falls back to CPU if unavailable."""
        if self.nli_available:
            return True
            
        try:
            from transformers import pipeline as hf_pipeline
            
            # Use GPU if available, else CPU
            if torch.cuda.is_available():
                device = 0  # First CUDA device
                device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
            else:
                device = -1  # CPU
                device_name = "CPU"
            
            print(f"[ClaimDetector] Loading NLI model: {self.nli_model_name} ({device_name})...")
            
            self.nli_pipeline = hf_pipeline(
                "zero-shot-classification",
                model=self.nli_model_name,
                device=device,
            )
            
            self.nli_available = True
            print(f"[ClaimDetector] NLI model loaded successfully on {device_name}.")
            return True
            
        except Exception as e:
            print(f"[ClaimDetector] WARNING: NLI model failed to load: {e}")
            print(f"[ClaimDetector] Falling back to regex-only mode.")
            self.nli_available = False
            return False
    
    def _regex_fast_pass(self, sentence: str) -> Optional[bool]:
        """
        Regex fast-pass: returns False for obvious non-claims, None for ambiguous.
        
        Returns:
            False  → definitely not a claim (skip NLI)
            None   → ambiguous, needs NLI classification
        """
        text = sentence.strip()
        words = text.split()
        
        # Rule 1: very short fragments are not claims
        if len(words) < 3:
            if _NUMBER_RE.search(text):
                return None  # Short but has numbers — let NLI decide
            return False
        
        # Rule 2: known filler phrases
        normalized = text.lower().rstrip("!.,;:?")
        if normalized in _FILLER_PHRASES or normalized + "." in _FILLER_PHRASES:
            return False
        
        # Rule 3: meta / transitional sentences
        for pat in _META_PATTERNS:
            if pat.search(text):
                return False
        
        # Rule 4: advisory / caveat (only if no proper nouns alongside)
        for pat in _ADVISORY_PATTERNS:
            if pat.search(text) and not _PROPER_NOUN_RE.search(text):
                return False
        
        # Rule 5: hedging / opinion frames
        for pat in _HEDGING_PATTERNS:
            if pat.search(text):
                return False

        # Rule 5b: refusal / "I don't know" / no-information — declining to answer is not a claim
        for pat in _REFUSAL_PATTERNS:
            if pat.search(text):
                return False

        # Rule 6: label / heading sentences ending with ":"
        if _LABEL_RE.match(text) and len(words) <= 10:
            return False
        
        # Rule 7: pure questions
        if _QUESTION_RE.match(text):
            return False
        
        # Rule 8: greetings / sign-offs
        if _GREETING_RE.match(text) and len(words) < 6:
            return False
        
        # Ambiguous — needs NLI
        return None
    
    def _has_factual_content(self, sentence: str) -> bool:
        """
        Check if a sentence contains factual indicators (numbers, units, measurements)
        that suggest it should be scored even if NLI classifies it as non-claim.
        """
        for pat in _FACTUAL_INDICATORS:
            if pat.search(sentence):
                return True
        return False
    
    def _nli_classify_batch(self, sentences: List[str]) -> List[bool]:
        """
        Batch-classify sentences via NLI. Much faster than one-by-one.
        
        Returns a list of booleans (True = claim, False = non-claim).
        """
        if not self.nli_available or not self.nli_pipeline or not sentences:
            return [True] * len(sentences)  # Conservative fallback
        
        try:
            results = self.nli_pipeline(
                sentences,
                candidate_labels=NLI_LABELS,
                multi_label=False,
                batch_size=len(sentences),  # Process all at once
            )
            
            # Handle single sentence (pipeline returns dict instead of list)
            if isinstance(results, dict):
                results = [results]
            
            flags = []
            for result, sentence in zip(results, sentences):
                top_label = result["labels"][0]
                label_scores = dict(zip(result["labels"], result["scores"]))
                claim_score = label_scores.get(NLI_CLAIM_LABEL, 0.0)
                
                is_factual = (top_label == NLI_CLAIM_LABEL and claim_score > NLI_CLAIM_THRESHOLD)
                
                # FACT RESCUE: If NLI says non-claim but sentence has numbers/measurements,
                # override to claim — better to score it than miss a hallucination
                rescued = False
                if not is_factual and self._has_factual_content(sentence):
                    is_factual = True
                    rescued = True
                
                status = "RESCUED" if rescued else ("claim" if is_factual else "non-claim")
                print(f"[ClaimDetector/NLI] \"{sentence[:60]}\" -> "
                      f"claim={claim_score:.3f} | top='{top_label}' | "
                      f"is_claim={is_factual} [{status}]")
                
                flags.append(is_factual)
            
            return flags
            
        except Exception as e:
            print(f"[ClaimDetector/NLI] Batch error: {e}")
            return [True] * len(sentences)
    
    def is_claim(self, sentence: str) -> bool:
        """
        Classify a single sentence. For multiple sentences, use classify_sentences().
        """
        if not sentence:
            return False
        
        regex_result = self._regex_fast_pass(sentence)
        if regex_result is not None:
            return regex_result
        
        if self.nli_available:
            return self._nli_classify_batch([sentence])[0]
        
        return True  # Conservative default
    
    def classify_sentences(self, sentences: List[str]) -> List[bool]:
        """
        Batch-classify multiple sentences as claims/non-claims.
        
        Uses regex fast-pass first, then batches remaining ambiguous
        sentences into a single NLI call for maximum throughput.
        
        Returns:
            List of booleans aligned with input (True = claim).
        """
        results = [None] * len(sentences)
        nli_indices = []  # Indices of sentences that need NLI
        nli_sentences = []  # The actual sentences for NLI
        
        # Stage 1: Regex fast-pass
        for i, sent in enumerate(sentences):
            if not sent:
                results[i] = False
                continue
            regex_result = self._regex_fast_pass(sent)
            if regex_result is not None:
                results[i] = regex_result
            else:
                nli_indices.append(i)
                nli_sentences.append(sent)
        
        # Stage 2: Batch NLI for remaining ambiguous sentences
        if nli_sentences and self.nli_available:
            nli_flags = self._nli_classify_batch(nli_sentences)
            for idx, flag in zip(nli_indices, nli_flags):
                results[idx] = flag
        
        # Stage 3: Fill any remaining None with conservative True
        return [r if r is not None else True for r in results]


def filter_claims(sentences: List[str]) -> List[dict]:
    """
    Classify a list of sentences as claims or non-claims.
    Uses the module-level claim_detector singleton.

    Returns:
        List of dicts with "sentence", "is_claim" keys.
    """
    return [
        {"sentence": s, "is_claim": claim_detector.is_claim(s)}
        for s in sentences
    ]


# ──────────── Module-level singleton ────────────

claim_detector = ClaimDetector()
