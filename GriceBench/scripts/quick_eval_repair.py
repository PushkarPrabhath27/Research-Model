"""
Quick Repair Model Evaluation
==============================

Quick test to verify the repair model works correctly.
Tests on 10 examples and shows results.
"""

import json
import torch
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random


# ============================================================================
# CONFIGURATION
# ============================================================================

class QuickEvalConfig:
    """Quick evaluation configuration."""
    # Model paths
    repair_model_path = 'models/repair/repair_model'
    test_data_path = 'data_processed/repair_data/repair_test.json'
    
    # Generation settings
    max_input_length = 512
    max_output_length = 256
    num_beams = 4
    num_examples = 10  # Quick test on 10 examples
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# QUICK EVALUATOR
# ============================================================================

class QuickRepairEvaluator:
    """Quick evaluation of repair model."""
    
    def __init__(self, config: QuickEvalConfig = None):
        self.config = config or QuickEvalConfig()
        self.device = torch.device(self.config.device)
        
        print(f"Quick Repair Evaluator")
        print(f"   Device: {self.device}")
    
    def load_model(self):
        """Load trained repair model."""
        print(f"\n[*] Loading repair model from {self.config.repair_model_path}...")
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.repair_model_path)
            self.model = T5ForConditionalGeneration.from_pretrained(self.config.repair_model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[OK] Model loaded successfully")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            return False
    
    def load_test_data(self):
        """Load test data."""
        print(f"\n[*] Loading test data from {self.config.test_data_path}...")
        
        try:
            with open(self.config.test_data_path, 'r', encoding='utf-8') as f:
                all_test_data = json.load(f)
            
            # Randomly sample examples
            if len(all_test_data) > self.config.num_examples:
                self.test_examples = random.sample(all_test_data, self.config.num_examples)
            else:
                self.test_examples = all_test_data
            
            print(f"[OK] Loaded {len(self.test_examples)} test examples")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading test data: {e}")
            return False
    
    def generate_repair(self, input_text: str) -> str:
        """Generate a repair for violated text."""
        try:
            inputs = self.tokenizer(
                input_text,
                max_length=self.config.max_input_length,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=self.config.max_output_length,
                    num_beams=self.config.num_beams,
                    early_stopping=True
                )
            
            repair = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return repair
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def run_quick_eval(self):
        """Run quick evaluation and display results."""
        print(f"\n{'='*80}")
        print("QUICK EVALUATION RESULTS")
        print('='*80)
        
        for idx, example in enumerate(self.test_examples):
            print(f"\n{'-'*80}")
            print(f"Example {idx + 1}/{len(self.test_examples)}")
            print(f"{'-'*80}")
            
            # Show violation types
            v_types = example.get('violation_types', [])
            print(f"\n[!] Violation Types: {', '.join(v_types) if v_types else 'Unknown'}")
            
            # Show input (truncated)
            input_preview = example['input_text'][:200] + "..." if len(example['input_text']) > 200 else example['input_text']
            print(f"\n[INPUT] Violated text:")
            print(f"   {input_preview}")
            
            # Generate repair
            print(f"\n[*] Generating repair...")
            repair = self.generate_repair(example['input_text'])
            
            # Show repair
            print(f"\n[REPAIR] Generated:")
            print(f"   {repair}")
            
            # Show reference (target)
            print(f"\n[TARGET] Reference:")
            print(f"   {example['target_text']}")
            
            # Quick similarity check
            repair_lower = repair.lower()
            target_lower = example['target_text'].lower()
            
            if repair_lower == target_lower:
                print(f"\n[***] EXACT MATCH!")
            elif target_lower in repair_lower or repair_lower in target_lower:
                print(f"\n[+] Close match (contains target)")
            else:
                # Count word overlap
                repair_words = set(repair_lower.split())
                target_words = set(target_lower.split())
                overlap = len(repair_words & target_words)
                total = len(target_words)
                if total > 0:
                    similarity = overlap / total
                    print(f"\n[~] Word overlap: {overlap}/{total} ({similarity:.1%})")
        
        print(f"\n{'='*80}")
        print("[OK] QUICK EVALUATION COMPLETE")
        print('='*80)
        
        print(f"\n[NEXT] Next Steps:")
        print(f"   1. Review the repairs above - do they make sense?")
        print(f"   2. Check if violations are fixed")
        print(f"   3. If results look good → run full evaluation")
        print(f"   4. If results look bad → may need to retrain or debug")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run quick repair model evaluation."""
    print("="*80)
    print("QUICK REPAIR MODEL EVALUATION")
    print("="*80)
    
    # Initialize
    evaluator = QuickRepairEvaluator()
    
    # Load model
    if not evaluator.load_model():
        print("\n[ERROR] Could not load model. Check the model path!")
        return
    
    # Load test data
    if not evaluator.load_test_data():
        print("\n[ERROR] Could not load test data. Check the data path!")
        return
    
    # Run evaluation
    evaluator.run_quick_eval()


if __name__ == "__main__":
    main()
