

import os
import pickle
import torch
import numpy as np
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging
from sentence_segmenter import split_sentences_with_spans, get_backend_name
from claim_filter import claim_detector

# Configure logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

print(f"[Sentence Segmenter] Using backend: {get_backend_name()}")

# Constants
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "semantic_entropy_probes", "models", "Llama2-7b_inference.pkl")

# System prompt to keep answers focused without forcing a single sentence.
# We still want multiple sentences so the per-sentence probes have something to
# score, but we discourage unnecessary padding/rambling on simple questions.
SYSTEM_PROMPT = (
    "Answer the question directly and concisely. Use as many sentences as the "
    "question genuinely needs, but do not add unnecessary filler, repetition, or "
    "tangential detail."
)

class InferenceEngine:
    def __init__(self, model_name=DEFAULT_MODEL_NAME, load_in_4bit=True):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.probes = None
        self.selected_probe = None
        
    def check_memory(self):
        vm = psutil.virtual_memory()
        print(f"[DEBUG] System RAM - Total: {vm.total/1e9:.2f}GB, Available: {vm.available/1e9:.2f}GB")

    def load_probes(self, path=PROBE_PATH):
        print(f"Loading probes from {path}...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Probe file not found at {path}")
            
        with open(path, 'rb') as f:
            self.probes = pickle.load(f)
        
        # Patch LogisticRegression models for sklearn version compatibility
        # Probes may have been saved with a different sklearn version where
        # 'multi_class' was removed. We need to add it back for predict_proba.
        from sklearn.linear_model import LogisticRegression
        if self.probes:
            for probe in self.probes:
                for key in ['s_bmodel', 's_amodel']:
                    if key in probe and isinstance(probe[key], LogisticRegression):
                        if not hasattr(probe[key], 'multi_class'):
                            probe[key].multi_class = 'auto'
        
        if self.probes:
            # Default to TriviaQA as it's a good general benchmark, or SQuAD.
            # Try to find 'llama3-triviaqa' first
            selected = next((p for p in self.probes if p.get('name') == 'llama3-triviaqa'), None)
            
            # Fallback to squad
            if not selected:
                selected = next((p for p in self.probes if p.get('name') == 'llama3-squad'), None)
            
            # Fallback to whatever is first
            if not selected:
                selected = self.probes[0]
                
            self.selected_probe = selected
            print(f"Loaded probes. Using default probe trained on: {self.selected_probe['name']}")
        
    def unload_model(self):
        if self.model:
            print("Unloading model...")
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
            
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded and GPU memory cleared.")

    def load_model(self, model_name=None):
        if model_name:
            self.model_name = model_name
            
        print(f"Loading model {self.model_name}...")
        self.check_memory()
        
        # Dynamic Probe Loading based on Model Name
        if "Llama-3" in self.model_name:
             probe_file = "Llama3-8b_inference.pkl"
        else:
             # Default/Fallback to Llama 2
             probe_file = "Llama2-7b_inference.pkl"

        probe_path = os.path.join(os.path.dirname(SCRIPT_DIR), "semantic_entropy_probes", "models", probe_file)
        self.load_probes(probe_path)

        quantization_config = None
        if self.load_in_4bit:
            print("Using 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            
            # Standard Loading (Optimized for GPU)
            # Removed strict CPU memory limit to allow faster loading
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, 
                    device_map=device_map,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
                print("Model loaded successfully.")
                return True
            except Exception as e:
                print(f"Model load failed: {e}")
                return False
                
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False

    def get_hidden_states(self, inputs, token_index=-1):
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states
        extracted_states = []
        for layer_state in hidden_states:
            vec = layer_state[0, token_index, :].cpu().numpy()
            extracted_states.append(vec)
            
        return np.stack(extracted_states)

    def run_probe_prediction(self, probe_model, hidden_states, layer_range):
        start, end = layer_range
        selected_states = hidden_states[start:end] 
        input_vec = selected_states.flatten().reshape(1, -1)
        probs = probe_model.predict_proba(input_vec)
        return probs[0, 1]

    def generate_response(self, user_input):
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
            
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        # Check for chat template (Critical for Llama 3 Instruct)
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            chat_output = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )
            
            # Handle both tensor and BatchEncoding return types
            if isinstance(chat_output, torch.Tensor):
                input_ids = chat_output.to(self.model.device)
            else:
                input_ids = chat_output['input_ids'].to(self.model.device)
            
            inputs = {
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids)
            }
            prompt_len = input_ids.shape[1]
        else:
            # Fallback for Base models
            prompt = f"{SYSTEM_PROMPT}\nQuestion: {user_input}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            prompt_len = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=512, # Generous budget so the demo can give longer answers when needed
                do_sample=True,          
                temperature=0.7,         
                top_p=0.9,               
                repetition_penalty=1.2   
            )
        
        # Ensure generated_ids is 2D (batch, seq)
        if generated_ids.dim() == 1:
            generated_ids = generated_ids.unsqueeze(0)
        
        print(f"[DEBUG] generated_ids shape: {generated_ids.shape}, prompt_len: {prompt_len}")
        
        # Decode full text and answer portion
        full_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(generated_ids[0, :prompt_len], skip_special_tokens=True)
        answer_text = full_text[len(prompt_text):].strip()
        
        print(f"[DEBUG] full_text length: {len(full_text)}, prompt_text length: {len(prompt_text)}, answer_text length: {len(answer_text)}")

        if not answer_text:
             return {"answer": "", "confidence": 1.0, "slt_confidence": 1.0, "sentence_details": []}

        # 1. Robust Sentence Splitting (spaCy → pysbd → regex fallback)
        span_results = split_sentences_with_spans(answer_text)
        sentences = [s["sentence"] for s in span_results]
        sources = [s["source"] for s in span_results]
        claim_flags = claim_detector.classify_sentences(sentences)
        print(f"[DEBUG] Detected {len(sentences)} sentences ({sum(claim_flags)} claims, {len(sentences) - sum(claim_flags)} non-claims). Backend: {get_backend_name()}")

        # 2. Pre-compute hidden states for the FULL sequence
        full_inputs = {'input_ids': generated_ids, 'attention_mask': torch.ones_like(generated_ids)}
        
        with torch.no_grad():
             outputs = self.model(**full_inputs, output_hidden_states=True)
        
        all_hidden_states = outputs.hidden_states # Tuple of (batch, seq, dim)
        
        # 3. Extract OVERALL SLT (second-to-last token of full answer)
        # This is a SECONDARY reference score — kept for comparison
        overall_slt_idx = generated_ids.shape[1] - 2  # Second-to-last token
        
        # Bounds checking
        if overall_slt_idx < prompt_len:
            overall_slt_idx = prompt_len  # Fallback for very short answers
        
        # Extract hidden states at overall SLT
        overall_extracted_states = []
        for layer_state in all_hidden_states:
            vec = layer_state[0, overall_slt_idx, :].cpu().numpy()
            overall_extracted_states.append(vec)
        
        overall_slt_states = np.stack(overall_extracted_states)
        
        # Run probes on overall SLT for secondary scores
        overall_entropy = 0.0
        overall_accuracy = 0.0
        
        if self.selected_probe:
            if 's_bmodel' in self.selected_probe:
                overall_entropy = self.run_probe_prediction(
                    self.selected_probe['s_bmodel'], 
                    overall_slt_states, 
                    self.selected_probe['sep_layer_range']
                )
                
            if 's_amodel' in self.selected_probe:
                overall_accuracy = self.run_probe_prediction(
                    self.selected_probe['s_amodel'], 
                    overall_slt_states, 
                    self.selected_probe['ap_layer_range']
                )
        
        # 4. Sentence-level analysis (approximate, for highlighting)
        # Wrapped in try/except for robustness - if this fails, we still return overall scores
        sentence_details = []
        try:
            # Note: Probes were trained on short answers, so this is less reliable for long answers
            sentence_entropies = []
            sentence_accuracies = []
            
            # Sentence-level analysis (ROBUST MAPPING)
            sentence_end_indices = []
            
            # Determine character-level boundaries of sentences in the FULL text
            # We start searching after the prompt to avoid matching text in the prompt
            search_cursor = len(prompt_text) 
            for sent in sentences:
                # Find closest existence of this sentence after the cursor
                # Note: exact match required. nltk sent_tokenize usually preserves text.
                start_index = full_text.find(sent, search_cursor)
                if start_index == -1:
                    # Fallback: just add length (unlikely to happen unless normalization occurred)
                    start_index = search_cursor
                
                end_index = start_index + len(sent)
                sentence_end_indices.append(end_index)
                search_cursor = end_index

            print(f"[DEBUG] Sentence end chars: {sentence_end_indices}")

            # Now map these character indices to token indices
            # We scan through the generated tokens, decoding progressively
            sentence_entropies = []
            sentence_accuracies = []
            
            current_sent_idx = 0
            
            # Let's just scan from prompt_len to end
            scan_start_token = prompt_len
            scan_end_token = generated_ids.shape[1]
            
            # Token-to-Char mapping loop
            for token_i in range(scan_start_token, scan_end_token + 1):
                 # Decode up to this token
                 partial_text = self.tokenizer.decode(generated_ids[0, :token_i], skip_special_tokens=True)
                 
                 # Check if we covered the current sentence
                 if current_sent_idx < len(sentence_end_indices):
                     target_char_end = sentence_end_indices[current_sent_idx]
                     
                     if len(partial_text) >= target_char_end:
                         candidate_idx = token_i - 1
                         slt_idx = candidate_idx - 1
                         
                         # Bounds check
                         if slt_idx < prompt_len:
                             slt_idx = prompt_len
                         
                         # Skip probe computation for non-claim sentences
                         if not claim_flags[current_sent_idx]:
                             sentence_entropies.append(0.0)
                             sentence_accuracies.append(1.0)
                             current_sent_idx += 1
                             continue
                         
                         # Extract states (only for claims)
                         extracted_states = []
                         for layer_state in all_hidden_states:
                            vec = layer_state[0, slt_idx, :].cpu().numpy()
                            extracted_states.append(vec)
                         
                         sentence_slt_states = np.stack(extracted_states)
                         
                         # Run probes on claim-bearing sentences
                         sent_unc = 0.0
                         sent_acc = 0.0
                         
                         if self.selected_probe:
                            if 's_bmodel' in self.selected_probe:
                                sent_unc = self.run_probe_prediction(
                                    self.selected_probe['s_bmodel'], 
                                    sentence_slt_states, 
                                    self.selected_probe['sep_layer_range']
                                )
                            if 's_amodel' in self.selected_probe:
                                sent_acc = self.run_probe_prediction(
                                    self.selected_probe['s_amodel'], 
                                    sentence_slt_states, 
                                    self.selected_probe['ap_layer_range']
                                )
                         
                         sentence_entropies.append(sent_unc)
                         sentence_accuracies.append(sent_acc)
                         
                         current_sent_idx += 1

            # Fill remaining if any
            while len(sentence_entropies) < len(sentences):
                 sentence_entropies.append(0.0)
                 sentence_accuracies.append(1.0)

            print(f"[DEBUG] Sentence scores - matched {current_sent_idx}/{len(sentences)} sentences")
            for i in range(len(sentences)):
                print(f"[DEBUG]   [{i}] ent={sentence_entropies[i]:.4f} acc={sentence_accuracies[i]:.4f} claim={claim_flags[i]}")

            # CONSISTENCY FIX: When there is exactly one claim sentence,
            # use the overall SLT scores (more reliable, matches training)
            # so the per-sentence and aggregate badge show the same values.
            claim_indices = [i for i, flag in enumerate(claim_flags) if flag]
            if len(claim_indices) == 1:
                ci = claim_indices[0]
                sentence_entropies[ci] = overall_entropy
                sentence_accuracies[ci] = overall_accuracy

            # Prepare sentence details for frontend highlighting
            claim_confidences = []  # Collect claim sentence confidences for aggregate
            for i, sent in enumerate(sentences):
                ent = sentence_entropies[i] if i < len(sentence_entropies) else 0.0
                acc = sentence_accuracies[i] if i < len(sentence_accuracies) else 1.0
                # Unified confidence: average of (1-entropy) and accuracy_prob
                conf = (acc + (1.0 - ent)) / 2.0
                
                is_claim = claim_flags[i] if i < len(claim_flags) else True
                if is_claim:
                    claim_confidences.append(conf)
                
                sentence_details.append({
                    "text": sent,
                    "confidence": float(conf),
                    "entropy": float(ent),
                    "accuracy_prob": float(acc),
                    "is_claim": is_claim,
                    "source": sources[i] if i < len(sources) else "unknown",
                })
            
            print(f"[DEBUG] Successfully computed {len(sentence_details)} sentence details")
            
        except Exception as e:
            print(f"[WARNING] Sentence-level analysis failed: {e}")
            # Fallback: return sentences with overall SLT scores
            overall_conf = (overall_accuracy + (1.0 - overall_entropy)) / 2.0
            claim_confidences = []
            sentence_details = []
            for i, sent in enumerate(sentences):
                is_claim = claim_flags[i] if i < len(claim_flags) else True
                if is_claim:
                    claim_confidences.append(overall_conf)
                sentence_details.append({
                    "text": sent,
                    "confidence": float(overall_conf),
                    "entropy": float(overall_entropy),
                    "accuracy_prob": float(overall_accuracy),
                    "is_claim": is_claim,
                    "source": sources[i] if i < len(sources) else "unknown",
                })

        # Compute aggregate confidence (mean of claim sentences)
        if claim_confidences:
            aggregate_confidence = sum(claim_confidences) / len(claim_confidences)
        else:
            # No claim sentences — default to high confidence (nothing to hallucinate)
            aggregate_confidence = 1.0
        
        # Compute SLT confidence (secondary reference)
        slt_confidence = (overall_accuracy + (1.0 - overall_entropy)) / 2.0
        
        print(f"[DEBUG] Aggregate confidence: {aggregate_confidence:.3f} (from {len(claim_confidences)} claims), SLT confidence: {slt_confidence:.3f}")

        return {
            "answer": answer_text,
            "confidence": float(aggregate_confidence),      # PRIMARY: claim-average
            "slt_confidence": float(slt_confidence),        # SECONDARY: overall SLT
            "sentence_count": len(sentences),
            "sentence_details": sentence_details,
        }

# Singleton instance
engine = InferenceEngine()

