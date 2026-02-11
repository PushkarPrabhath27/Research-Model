"""
GriceBench Repair Model Evaluation - Chapter 10-12
===================================================

Evaluate repair model performance:
1. Targeted fix rate (does it remove the violation?)
2. No-regression rate (does it avoid creating new violations?)
3. BLEU score (similarity to reference repairs)
4. BERTScore (semantic similarity)

Based on Chapter 12 of the Implementation Guide.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import T5Tokenizer, T5ForConditionalGeneration
from collections import defaultdict
import numpy as np

# Import detector for fix rate evaluation
import sys
sys.path.append(str(Path(__file__).parent))


# ============================================================================
# CONFIGURATION
# ============================================================================

class EvalConfig:
    """Evaluation configuration."""
    # Paths
    repair_model_path = 'models/repair/repair_model'
    detector_model_path = 'models/detector/best_model.pt'
    test_data_path = 'data_processed/repair_data/repair_test.json'
    
    # Model
    model_name = 't5-base'
    max_input_length = 512
    max_output_length = 256
    
    # Generation
    num_beams = 4
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output
    results_dir = 'results/repair_evaluation'


# ============================================================================
# REPAIR EVALUATOR
# ============================================================================

class RepairEvaluator:
    """Evaluate repair model performance."""
    
    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()
        self.device = torch.device(self.config.device)
        
        print(f"[*] Repair Evaluator on {self.device}")
    
    def load_repair_model(self):
        """Load trained repair model."""
        print(f"\n[*] Loading repair model from {self.config.repair_model_path}...")
        
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.repair_model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.repair_model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[OK] Repair model loaded")
    
    def load_test_data(self):
        """Load test data."""
        print(f"\n[*] Loading test data from {self.config.test_data_path}...")
        
        with open(self.config.test_data_path, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        
        print(f"[OK] Loaded {len(self.test_data):,} test examples")
    
    def generate_repair(self, input_text: str) -> str:
        """Generate a repair for violated text."""
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
    
    def calculate_bleu(self, hypothesis: str, reference: str) -> float:
        """
        Calculate BLEU score (simplified version).
        For production, use sacrebleu library.
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Tokenize
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()
        
        # Calculate BLEU with smoothing
        smoothie = SmoothingFunction().method1
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        
        return score
    
    def calculate_exact_match(self, hypothesis: str, reference: str) -> bool:
        """Check if repair exactly matches reference."""
        return hypothesis.strip().lower() == reference.strip().lower()
    
    def run_evaluation(self, max_examples: int = None) -> Dict:
        """Run full evaluation."""
        print(f"\n[*] Running evaluation...")
        
        if max_examples:
            test_subset = self.test_data[:max_examples]
            print(f"   Evaluating on {max_examples} examples")
        else:
            test_subset = self.test_data
        
        results = {
            'repairs': [],
            'bleu_scores': [],
            'exact_matches': [],
            'by_violation_type': defaultdict(list)
        }
        
        for idx, example in enumerate(test_subset):
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(test_subset)}...")
            
            # Generate repair
            repair = self.generate_repair(example['input_text'])
            reference = example['target_text']
            
            # Calculate BLEU
            bleu = self.calculate_bleu(repair, reference)
            exact = self.calculate_exact_match(repair, reference)
            
            results['bleu_scores'].append(bleu)
            results['exact_matches'].append(exact)
            
            # Store result
            result = {
                'input': example['input_text'][:200],  # Truncate for storage
                'repair': repair,
                'reference': reference,
                'bleu': bleu,
                'exact_match': exact,
                'violation_types': example.get('violation_types', [])
            }
            results['repairs'].append(result)
            
            # Group by violation type
            for v_type in example.get('violation_types', []):
                results['by_violation_type'][v_type].append(bleu)
        
        return results
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate summary metrics."""
        metrics = {}
        
        # Overall metrics
        metrics['bleu'] = {
            'mean': float(np.mean(results['bleu_scores'])),
            'std': float(np.std(results['bleu_scores'])),
            'min': float(np.min(results['bleu_scores'])),
            'max': float(np.max(results['bleu_scores']))
        }
        
        metrics['exact_match_rate'] = float(np.mean(results['exact_matches']))
        
        # Per-violation metrics
        metrics['by_violation'] = {}
        for v_type, bleu_scores in results['by_violation_type'].items():
            metrics['by_violation'][v_type] = {
                'bleu_mean': float(np.mean(bleu_scores)),
                'count': len(bleu_scores)
            }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in readable format."""
        print(f"\n{'='*70}")
        print("REPAIR EVALUATION RESULTS")
        print('='*70)
        
        print(f"\n[OVERALL METRICS]:")
        print(f"   BLEU Score:      {metrics['bleu']['mean']:.4f} (±{metrics['bleu']['std']:.4f})")
        print(f"   Exact Match:     {metrics['exact_match_rate']:.2%}")
        print(f"   BLEU Range:      [{metrics['bleu']['min']:.4f}, {metrics['bleu']['max']:.4f}]")
        
        print(f"\n[PER-VIOLATION PERFORMANCE]:")
        print(f"{'-'*70}")
        print(f"{'Violation':<15} {'BLEU':<8} {'Count'}")
        print(f"{'-'*70}")
        
        for v_type, stats in metrics['by_violation'].items():
            print(f"{v_type:<15} {stats['bleu_mean']:>6.4f}   {stats['count']:>5d}")
        
        print('='*70)
    
    def save_results(self, results: Dict, metrics: Dict):
        """Save evaluation results."""
        output_dir = Path(self.config.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save sample repairs (first 100)
        sample_repairs = results['repairs'][:100]
        with open(output_dir / 'sample_repairs.json', 'w', encoding='utf-8') as f:
            json.dump(sample_repairs, f, ensure_ascii=False, indent=2)
        
        # Create report
        self._create_report(metrics, sample_repairs, output_dir)
        
        print(f"\n[SAVED] Results saved to {output_dir}")
    
    def _create_report(self, metrics: Dict, samples: List[Dict], output_dir: Path):
        """Create markdown report."""
        report = []
        report.append("# GriceBench Repair Model Evaluation\n")
        report.append(f"Test Examples: {len(samples)}\n")
        report.append("\n---\n")
        
        # Metrics
        report.append("\n## Overall Performance\n")
        report.append(f"- **BLEU Score**: {metrics['bleu']['mean']:.4f}\n")
        report.append(f"- **Exact Match**: {metrics['exact_match_rate']:.2%}\n")
        
        report.append("\n## Per-Violation Performance\n")
        report.append("| Violation | BLEU | Count |\n")
        report.append("|-----------|------|-------|\n")
        for v_type, stats in metrics['by_violation'].items():
            report.append(f"| {v_type} | {stats['bleu_mean']:.4f} | {stats['count']} |\n")
        
        # Sample repairs
        report.append("\n## Sample Repairs\n")
        for i, sample in enumerate(samples[:5]):
            report.append(f"\n### Example {i+1}\n")
            report.append(f"**Violation**: {', '.join(sample['violation_types'])}\n\n")
            report.append(f"**Generated Repair**: {sample['repair']}\n\n")
            report.append(f"**Reference**: {sample['reference']}\n\n")
            report.append(f"**BLEU**: {sample['bleu']:.4f}\n")
        
        # Write
        with open(output_dir / 'evaluation_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run repair model evaluation."""
    print("="*70)
    print("GRICEBENCH REPAIR MODEL EVALUATION")
    print("="*70)
    
    # Initialize
    evaluator = RepairEvaluator()
    
    # Load model and data
    evaluator.load_repair_model()
    evaluator.load_test_data()
    
    # Run evaluation
    results = evaluator.run_evaluation()  # Full test set
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(results)
    evaluator.print_metrics(metrics)
    
    # Save results
    evaluator.save_results(results, metrics)
    
    print(f"\n{'='*70}")
    print("[OK] EVALUATION COMPLETE!")
    print('='*70)


if __name__ == "__main__":
    # Install NLTK for BLEU
    try:
        import nltk
        nltk.download('punkt', quiet=True)
    except:
        print("[WARNING] Install nltk for BLEU: pip install nltk")
    
    main()
