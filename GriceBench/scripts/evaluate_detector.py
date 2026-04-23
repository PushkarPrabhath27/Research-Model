"""
GriceBench Detector Evaluation - Chapter 9
===========================================

Comprehensive evaluation of the trained DeBERTa detector including:
- Primary metrics (F1, precision, recall, exact match)
- Error analysis (confusion matrices, difficult examples)
- Per-maxim performance breakdown
- Evaluation report generation

Based on Chapter 9 of the Implementation Guide.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from collections import defaultdict
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

class EvalConfig:
    """Evaluation configuration."""
    # Model
    model_name = 'microsoft/deberta-v3-base'
    model_path = 'models/detector/best_model.pt'
    
    # Data
    test_data_path = 'data_processed/detector_data/detector_val.json'  # Using val as test
    
    # Evaluation
    batch_size = 8
    max_length = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Output
    results_dir = 'results/detector_evaluation'
    
    # Maxims
    maxims = ['quantity', 'quality', 'relation', 'manner']
    
    # Thresholds
    confidence_threshold = 0.5


# ============================================================================
# MODEL DEFINITION (Must match training)
# ============================================================================

class ViolationDetector(nn.Module):
    """Violation detector model (same as training)."""
    
    def __init__(self, model_name: str, num_labels: int = 4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled))
        probs = torch.sigmoid(logits)
        return {'logits': logits, 'probs': probs}


# ============================================================================
# DATASET
# ============================================================================

class EvalDataset(Dataset):
    """Dataset for evaluation."""
    
    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.maxims = EvalConfig.maxims
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            ex['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Labels
        labels = torch.tensor(
            [ex['labels'].get(m, 0) for m in self.maxims],
            dtype=torch.float
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'violation_type': ex.get('violation_type', 'unknown'),
            'input_text': ex['input_text']
        }


# ============================================================================
# EVALUATOR
# ============================================================================

class DetectorEvaluator:
    """Comprehensive detector evaluation."""
    
    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()
        self.device = torch.device(self.config.device)
        
        # Results storage
        self.results = {
            'predictions': [],
            'labels': [],
            'probabilities': [],
            'examples': []
        }
        
        print(f"Evaluator initialized on {self.device}")
    
    def load_model(self):
        """Load trained model."""
        print(f"\n📥 Loading model from {self.config.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Load model
        self.model = ViolationDetector(self.config.model_name).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(
            self.config.model_path,
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   Checkpoint F1: {checkpoint.get('f1', 'N/A')}")
    
    def load_test_data(self):
        """Load test data."""
        print(f"\n📂 Loading test data from {self.config.test_data_path}...")
        
        with open(self.config.test_data_path, 'r', encoding='utf-8') as f:
            test_examples = json.load(f)
        
        self.test_dataset = EvalDataset(test_examples, self.tokenizer, self.config.max_length)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        print(f"✅ Loaded {len(self.test_dataset)} test examples")
    
    def run_evaluation(self):
        """Run evaluation on test set."""
        print(f"\n🔍 Running evaluation...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_examples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                probs = outputs['probs'].cpu()
                preds = (probs > self.config.confidence_threshold).float()
                
                # Store results
                all_preds.append(preds)
                all_labels.append(labels)
                all_probs.append(probs)
                
                # Store example metadata
                for i in range(len(labels)):
                    all_examples.append({
                        'violation_type': batch['violation_type'][i],
                        'input_text': batch['input_text'][i][:200]  # Truncate for storage
                    })
                
                if (batch_idx + 1) % 20 == 0:
                    print(f"   Processed {(batch_idx + 1) * self.config.batch_size} examples...")
        
        # Concatenate results
        self.results['predictions'] = torch.cat(all_preds).numpy()
        self.results['labels'] = torch.cat(all_labels).numpy()
        self.results['probabilities'] = torch.cat(all_probs).numpy()
        self.results['examples'] = all_examples
        
        print(f"✅ Evaluation complete!")
    
    def calculate_metrics(self) -> Dict:
        """Calculate all evaluation metrics (Chapter 9.1)."""
        print(f"\n📊 Calculating metrics...")
        
        preds = self.results['predictions']
        labels = self.results['labels']
        
        metrics = {}
        
        # Per-maxim metrics
        per_maxim = {}
        for i, maxim in enumerate(self.config.maxims):
            per_maxim[maxim] = {
                'f1': f1_score(labels[:, i], preds[:, i], zero_division=0),
                'precision': precision_score(labels[:, i], preds[:, i], zero_division=0),
                'recall': recall_score(labels[:, i], preds[:, i], zero_division=0),
                'support': int(labels[:, i].sum())
            }
        
        metrics['per_maxim'] = per_maxim
        
        # Macro-averaged F1
        metrics['macro_f1'] = np.mean([m['f1'] for m in per_maxim.values()])
        metrics['macro_precision'] = np.mean([m['precision'] for m in per_maxim.values()])
        metrics['macro_recall'] = np.mean([m['recall'] for m in per_maxim.values()])
        
        # Exact match (all 4 maxims correct)
        exact_matches = (preds == labels).all(axis=1)
        metrics['exact_match'] = exact_matches.mean()
        
        # Partial match (at least one maxim correct)
        partial_matches = (preds == labels).any(axis=1)
        metrics['partial_match'] = partial_matches.mean()
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in readable format."""
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print('='*70)
        
        print(f"\n📈 MACRO METRICS:")
        print(f"   Macro F1:        {metrics['macro_f1']:.4f}")
        print(f"   Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"   Macro Recall:    {metrics['macro_recall']:.4f}")
        print(f"   Exact Match:     {metrics['exact_match']:.4f}")
        print(f"   Partial Match:   {metrics['partial_match']:.4f}")
        
        print(f"\n📊 PER-MAXIM PERFORMANCE:")
        print(f"{'─'*70}")
        print(f"{'Maxim':<12} {'F1':<8} {'Precision':<11} {'Recall':<8} {'Support':<8}")
        print(f"{'─'*70}")
        
        for maxim, scores in metrics['per_maxim'].items():
            status = "✓" if scores['f1'] >= 0.7 else "○"
            print(f"{status} {maxim:<10} {scores['f1']:.4f}   {scores['precision']:.4f}      {scores['recall']:.4f}   {scores['support']:<8}")
        
        print('='*70)
    
    def generate_confusion_matrices(self):
        """Generate confusion matrices for error analysis (Chapter 9.2)."""
        print(f"\n🔍 Generating confusion matrices...")
        
        preds = self.results['predictions']
        labels = self.results['labels']
        
        # Create output directory
        output_dir = Path(self.config.results_dir) / 'confusion_matrices'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-maxim confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, maxim in enumerate(self.config.maxims):
            cm = confusion_matrix(labels[:, i], preds[:, i])
            
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['No Violation', 'Violation'],
                yticklabels=['No Violation', 'Violation'],
                ax=axes[i],
                cbar=False
            )
            axes[i].set_title(f'{maxim.capitalize()} Maxim')
            axes[i].set_ylabel('True Label')
            axes[i].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'per_maxim_confusion.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved confusion matrices to {output_dir}")
    
    def analyze_errors(self) -> Dict:
        """Analyze error patterns (Chapter 9.2)."""
        print(f"\n🔬 Analyzing errors...")
        
        preds = self.results['predictions']
        labels = self.results['labels']
        probs = self.results['probabilities']
        examples = self.results['examples']
        
        error_analysis = {}
        
        # Per-maxim error analysis
        for i, maxim in enumerate(self.config.maxims):
            # False positives: predicted violation but actually clean
            fp_mask = (preds[:, i] == 1) & (labels[:, i] == 0)
            fp_indices = np.where(fp_mask)[0]
            
            # False negatives: predicted clean but actually violation
            fn_mask = (preds[:, i] == 0) & (labels[:, i] == 1)
            fn_indices = np.where(fn_mask)[0]
            
            # Most confident errors
            fp_confident = []
            if len(fp_indices) > 0:
                fp_probs = probs[fp_indices, i]
                top_fp = fp_indices[np.argsort(-fp_probs)[:5]]
                fp_confident = [
                    {
                        'index': int(idx),
                        'confidence': float(probs[idx, i]),
                        'violation_type': examples[idx]['violation_type'],
                        'text_preview': examples[idx]['input_text']
                    }
                    for idx in top_fp
                ]
            
            fn_confident = []
            if len(fn_indices) > 0:
                fn_probs = 1 - probs[fn_indices, i]
                top_fn = fn_indices[np.argsort(-fn_probs)[:5]]
                fn_confident = [
                    {
                        'index': int(idx),
                        'confidence': float(1 - probs[idx, i]),
                        'violation_type': examples[idx]['violation_type'],
                        'text_preview': examples[idx]['input_text']
                    }
                    for idx in top_fn
                ]
            
            error_analysis[maxim] = {
                'false_positives': int(fp_mask.sum()),
                'false_negatives': int(fn_mask.sum()),
                'top_fp_examples': fp_confident,
                'top_fn_examples': fn_confident
            }
        
        return error_analysis
    
    def save_results(self, metrics: Dict, error_analysis: Dict):
        """Save all results to disk."""
        print(f"\n💾 Saving results...")
        
        output_dir = Path(self.config.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save error analysis
        with open(output_dir / 'error_analysis.json', 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        # Create evaluation report
        self._create_report(metrics, error_analysis, output_dir)
        
        print(f"✅ Results saved to {output_dir}")
    
    def _create_report(self, metrics: Dict, error_analysis: Dict, output_dir: Path):
        """Create markdown evaluation report."""
        report = []
        report.append("# GriceBench Detector Evaluation Report\n")
        report.append(f"Model: {self.config.model_name}\n")
        report.append(f"Test Examples: {len(self.results['labels'])}\n")
        report.append("\n---\n")
        
        # Metrics
        report.append("\n## Overall Performance\n")
        report.append(f"- **Macro F1**: {metrics['macro_f1']:.4f}\n")
        report.append(f"- **Exact Match**: {metrics['exact_match']:.4f}\n")
        report.append(f"- **Partial Match**: {metrics['partial_match']:.4f}\n")
        
        report.append("\n## Per-Maxim Performance\n")
        report.append("| Maxim | F1 | Precision | Recall | Support |\n")
        report.append("|-------|-----|-----------|--------|----------|\n")
        for maxim, scores in metrics['per_maxim'].items():
            report.append(f"| {maxim.capitalize()} | {scores['f1']:.4f} | {scores['precision']:.4f} | {scores['recall']:.4f} | {scores['support']} |\n")
        
        report.append("\n## Error Analysis\n")
        for maxim, errors in error_analysis.items():
            report.append(f"\n### {maxim.capitalize()}\n")
            report.append(f"- False Positives: {errors['false_positives']}\n")
            report.append(f"- False Negatives: {errors['false_negatives']}\n")
        
        # Write report
        with open(output_dir / 'evaluation_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report)


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def main():
    """Run complete detector evaluation (Chapter 9)."""
    print("="*70)
    print("GRICEBENCH DETECTOR EVALUATION - CHAPTER 9")
    print("="*70)
    
    # Initialize evaluator
    evaluator = DetectorEvaluator()
    
    # Load model and data
    evaluator.load_model()
    evaluator.load_test_data()
    
    # Run evaluation
    evaluator.run_evaluation()
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics()
    evaluator.print_metrics(metrics)
    
    # Generate visualizations
    evaluator.generate_confusion_matrices()
    
    # Error analysis
    error_analysis = evaluator.analyze_errors()
    
    # Save results
    evaluator.save_results(metrics, error_analysis)
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: results/detector_evaluation/")
    print(f"  - metrics.json")
    print(f"  - error_analysis.json")
    print(f"  - evaluation_report.md")
    print(f"  - confusion_matrices/")


if __name__ == "__main__":
    main()
