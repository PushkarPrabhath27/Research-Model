"""
GriceBench Human Evaluation - Sample Preparation - Part 2, Step 3
=================================================================

Prepares samples for human evaluation by:
1. Loading test data
2. Generating responses from your system
3. Generating responses from baseline models (optional, for comparison)
4. Shuffling and blinding system labels
5. Saving samples + secret key

Author: GriceBench
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PrepConfig:
    """Configuration for sample preparation."""
    test_data_path: str = "data_processed/repair_data/repair_test.json"
    output_path: str = "human_eval_samples.json"
    key_path: str = "human_eval_key_DO_NOT_SHARE.json"
    num_samples: int = 200
    include_baselines: bool = False  # Set to True on Kaggle for baseline comparison
    random_seed: int = 42


# ============================================================================
# SAMPLE PREPARATION
# ============================================================================

class HumanEvalSamplePreparer:
    """
    Prepares samples for human evaluation.
    
    Creates blinded samples from multiple systems so annotators
    can't tell which system generated each response.
    """
    
    def __init__(self, config: PrepConfig = None):
        self.config = config or PrepConfig()
        random.seed(self.config.random_seed)
    
    def prepare_samples(self, include_baselines: bool = None) -> List[Dict]:
        """
        Prepare samples for human evaluation.
        
        Args:
            include_baselines: Whether to include baseline model generations
                              (requires GPU and more memory)
        """
        include_baselines = include_baselines if include_baselines is not None else self.config.include_baselines
        
        print("="*70)
        print("PREPARING HUMAN EVALUATION SAMPLES")
        print("="*70)
        
        # Load test data
        test_data = self._load_test_data()
        if not test_data:
            return []
        
        # Sample subset
        num_samples = min(self.config.num_samples, len(test_data))
        sampled = random.sample(test_data, num_samples)
        print(f"\nSampled {num_samples} examples from test data")
        
        all_samples = []
        
        # Add samples from test data (assumes repaired responses exist)
        for i, item in enumerate(sampled):
            context = self._extract_context(item)
            evidence = self._extract_evidence(item)
            
            # Add reference (target) response as one candidate
            if item.get("target_text"):
                all_samples.append({
                    "context": context,
                    "evidence": evidence,
                    "response": item["target_text"],
                    "system": "gricebench_repair"
                })
            
            # Add original violated response as control
            violated = self._extract_violated_response(item)
            if violated:
                all_samples.append({
                    "context": context,
                    "evidence": evidence,
                    "response": violated,
                    "system": "original_violated"
                })
        
        print(f"Created {len(all_samples)} samples from your system")
        
        # Optionally add baseline model responses
        if include_baselines:
            print("\nGenerating baseline responses (this may take a while)...")
            baseline_samples = self._generate_baseline_samples(sampled)
            all_samples.extend(baseline_samples)
            print(f"Added {len(baseline_samples)} baseline samples")
        
        # Shuffle
        random.shuffle(all_samples)
        
        # Create blinded samples and key
        blinded_samples, system_key = self._blind_samples(all_samples)
        
        # Save
        self._save_samples(blinded_samples, system_key)
        
        print("\n" + "="*70)
        print("SAMPLE PREPARATION COMPLETE!")
        print("="*70)
        print(f"\nTotal samples: {len(blinded_samples)}")
        print(f"Samples file: {self.config.output_path}")
        print(f"Key file: {self.config.key_path} (DO NOT SHARE)")
        
        return blinded_samples
    
    def _load_test_data(self) -> List[Dict]:
        """Load test data."""
        path = Path(self.config.test_data_path)
        
        if not path.exists():
            print(f"Warning: Test data not found at {path}")
            print("Creating synthetic samples for demo...")
            return self._create_demo_data()
        
        print(f"Loading test data from {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"  Loaded {len(data)} examples")
        return data
    
    def _create_demo_data(self) -> List[Dict]:
        """Create demo data for testing."""
        return [
            {
                "input_text": "[CONTEXT] What is the capital of France? [EVIDENCE] Paris is the capital of France. [RESPONSE] The capital is Paris.",
                "target_text": "The capital of France is Paris, which is also its largest city."
            },
            {
                "input_text": "[CONTEXT] How do I make coffee? [EVIDENCE] [RESPONSE] Use beans.",
                "target_text": "To make coffee, grind fresh beans, use hot water (195-205°F), and brew for 4-5 minutes."
            },
            {
                "input_text": "[CONTEXT] What's your favorite color? [EVIDENCE] [RESPONSE] The stock market is up.",
                "target_text": "I enjoy many colors, but blue is quite calming and versatile."
            }
        ] * 20  # Repeat for more samples
    
    def _extract_context(self, item: Dict) -> str:
        """Extract context from item."""
        import re
        input_text = item.get("input_text", "")
        match = re.search(r'\[CONTEXT\](.*?)\[EVIDENCE\]', input_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return input_text[:200]
    
    def _extract_evidence(self, item: Dict) -> str:
        """Extract evidence from item."""
        import re
        input_text = item.get("input_text", "")
        match = re.search(r'\[EVIDENCE\](.*?)\[RESPONSE\]', input_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_violated_response(self, item: Dict) -> str:
        """Extract violated response from item."""
        import re
        input_text = item.get("input_text", "")
        match = re.search(r'\[RESPONSE\](.*?)$', input_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _generate_baseline_samples(self, sampled_data: List[Dict]) -> List[Dict]:
        """Generate responses from baseline models."""
        baseline_samples = []
        
        # Define baselines (these run on Kaggle)
        baselines = {
            "smollm_base": "HuggingFaceTB/SmolLM2-360M-Instruct",
            # Add more baselines as needed:
            # "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        }
        
        for baseline_name, model_id in baselines.items():
            print(f"  Generating with {baseline_name}...")
            
            try:
                responses = self._generate_with_model(model_id, sampled_data[:50])  # Limit for speed
                
                for i, response in enumerate(responses):
                    if i < len(sampled_data):
                        context = self._extract_context(sampled_data[i])
                        evidence = self._extract_evidence(sampled_data[i])
                        
                        baseline_samples.append({
                            "context": context,
                            "evidence": evidence,
                            "response": response,
                            "system": baseline_name
                        })
            except Exception as e:
                print(f"    Error with {baseline_name}: {e}")
        
        return baseline_samples
    
    def _generate_with_model(self, model_id: str, data: List[Dict]) -> List[str]:
        """Generate responses with a model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            print("transformers not available, skipping baseline generation")
            return []
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        responses = []
        
        for item in data:
            context = self._extract_context(item)
            
            # Format prompt
            prompt = f"User: {context}\nAssistant:"
            
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()
            
            responses.append(response[:500])  # Limit length
        
        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return responses
    
    def _blind_samples(self, samples: List[Dict]) -> tuple:
        """Create blinded samples and separate key."""
        blinded = []
        key = {}
        
        for i, sample in enumerate(samples):
            key[str(i)] = sample.get("system", "unknown")
            
            blinded.append({
                "id": i,
                "context": sample["context"],
                "evidence": sample.get("evidence", ""),
                "response": sample["response"]
            })
        
        return blinded, key
    
    def _save_samples(self, samples: List[Dict], key: Dict):
        """Save samples and key."""
        # Save blinded samples (share with annotators)
        with open(self.config.output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        # Save key (DO NOT share with annotators)
        with open(self.config.key_path, "w", encoding="utf-8") as f:
            json.dump(key, f, indent=2)
        
        print(f"\nSaved {len(samples)} blinded samples to {self.config.output_path}")
        print(f"Saved system key to {self.config.key_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Prepare human evaluation samples."""
    config = PrepConfig()
    preparer = HumanEvalSamplePreparer(config)
    preparer.prepare_samples()


if __name__ == "__main__":
    main()
