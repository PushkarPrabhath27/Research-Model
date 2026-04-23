"""
Quick Repair Model Evaluation - File Output Version
====================================================
Saves results to JSON file instead of console output
"""

import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random


class QuickEvalConfig:
    repair_model_path = 'models/repair/repair_model'
    test_data_path = 'data_processed/repair_data/repair_test.json'
    max_input_length = 512
    max_output_length = 256
    num_beams = 4
    num_examples = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class QuickRepairEvaluator:
    
    def __init__(self):
        self.config = QuickEvalConfig()
        self.device = torch.device(self.config.device)
        self.results = {
            'model_info': {
                'device': str(self.device),
                'model_path': self.config.repair_model_path
            },
            'examples': [],
            'summary': {}
        }
    
    def load_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.repair_model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.repair_model_path)
        self.model.to(self.device)
        self.model.eval()
        return True
    
    def load_test_data(self):
        with open(self.config.test_data_path, 'r', encoding='utf-8') as f:
            all_test_data = json.load(f)
        
        if len(all_test_data) > self.config.num_examples:
            self.test_examples = random.sample(all_test_data, self.config.num_examples)
        else:
            self.test_examples = all_test_data
        
        return True
    
    def generate_repair(self, input_text):
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
    
    def run_evaluation(self):
        exact_matches = 0
        close_matches = 0
        
        for idx, example in enumerate(self.test_examples):
            repair = self.generate_repair(example['input_text'])
            reference = example['target_text']
            
            # Calculate similarity
            repair_lower = repair.lower()
            target_lower = reference.lower()
            
            exact_match = (repair_lower == target_lower)
            close_match = (target_lower in repair_lower or repair_lower in target_lower)
            
            repair_words = set(repair_lower.split())
            target_words = set(target_lower.split())
            overlap = len(repair_words & target_words)
            total = len(target_words)
            word_overlap = overlap / total if total > 0 else 0
            
            if exact_match:
                exact_matches += 1
            if close_match:
                close_matches += 1
            
            result = {
                'example_id': idx + 1,
                'violation_types': example.get('violation_types', []),
                'input_preview': example['input_text'][:200],
                'generated_repair': repair,
                'reference_repair': reference,
                'exact_match': exact_match,
                'close_match': close_match,
                'word_overlap': word_overlap,
                'similarity_pct': f"{word_overlap:.1%}"
            }
            
            self.results['examples'].append(result)
        
        self.results['summary'] = {
            'total_examples': len(self.test_examples),
            'exact_matches': exact_matches,
            'close_matches': close_matches,
            'exact_match_rate': f"{exact_matches / len(self.test_examples):.1%}",
            'close_match_rate': f"{close_matches / len(self.test_examples):.1%}",
            'avg_word_overlap': f"{sum(r['word_overlap'] for r in self.results['examples']) / len(self.results['examples']):.1%}"
        }
    
    def save_results(self, output_file='quick_eval_results.json'):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)


def main():
    evaluator = QuickRepairEvaluator()
    evaluator.load_model()
    evaluator.load_test_data()
    evaluator.run_evaluation()
    evaluator.save_results()
    
    # Print summary only
    print("Evaluation complete. Results saved to quick_eval_results.json")
    print(f"Exact matches: {evaluator.results['summary']['exact_matches']}/{evaluator.results['summary']['total_examples']}")
    print(f"Close matches: {evaluator.results['summary']['close_matches']}/{evaluator.results['summary']['total_examples']}")


if __name__ == "__main__":
    main()
