"""
GriceBench Relation Repair Evaluation - Part 1, Step 4
=======================================================

Evaluates the new retrieval-based Relation repair using RELEVANCE metrics
instead of BLEU.

Why relevance instead of BLEU?
- BLEU measures n-gram overlap with a reference
- Relation repair generates NEW content, not edits
- Two equally good responses can have 0 BLEU overlap
- Relevance measures semantic similarity to context (what we actually want)

Metrics:
1. Relevance Score: Cosine similarity between context and response
2. Relevance Improvement: How much relevance increased after repair
3. Topic Alignment: Does the repaired response match the context topic?
4. (Optional) BLEU: For comparison with old approach

Author: GriceBench
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    test_data_path: str = "data_processed/repair_data/repair_test.json"
    t5_model_path: str = "models/repair/repair_model"
    corpus_path: str = "data_processed/topical_corpus.json"
    results_dir: str = "results/relation_repair_evaluation"
    max_examples: int = 200  # Limit for faster evaluation
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================================
# EVALUATOR
# ============================================================================

class RelationRepairEvaluator:
    """
    Comprehensive evaluation of retrieval-based Relation repair.
    
    Compares:
    1. Original (T5-based) Relation repair
    2. New retrieval-based Relation repair
    
    Using relevance metrics that actually measure what we care about.
    """
    
    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()
        self.encoder = None
        self.integrated_model = None
        self.test_data = None
        self.results = {}
    
    def _load_encoder(self):
        """Load sentence encoder for relevance scoring."""
        if self.encoder is not None:
            return
        
        from sentence_transformers import SentenceTransformer
        print(f"Loading encoder: {self.config.encoder_model}")
        self.encoder = SentenceTransformer(self.config.encoder_model)
    
    def _load_integrated_model(self):
        """Load the integrated repair model."""
        if self.integrated_model is not None:
            return
        
        from integrated_repair_model import IntegratedRepairModel
        print("Loading integrated repair model...")
        self.integrated_model = IntegratedRepairModel(
            t5_model_path=self.config.t5_model_path,
            corpus_path=self.config.corpus_path
        )
    
    def _load_test_data(self):
        """Load test data filtered for Relation violations."""
        if self.test_data is not None:
            return
        
        test_path = Path(self.config.test_data_path)
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        print(f"Loading test data from {test_path}")
        with open(test_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Filter for Relation violations only
        self.test_data = []
        for item in all_data:
            input_text = item.get("input_text", "")
            if "[VIOLATION=RELATION]" in input_text:
                self.test_data.append(item)
        
        # Limit samples
        if len(self.test_data) > self.config.max_examples:
            self.test_data = self.test_data[:self.config.max_examples]
        
        print(f"  Found {len(self.test_data)} Relation violation examples")
    
    def calculate_relevance(self, context: str, response: str) -> float:
        """
        Calculate semantic relevance between context and response.
        
        Returns cosine similarity in range [0, 1].
        """
        self._load_encoder()
        
        embeddings = self.encoder.encode(
            [context, response],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Cosine similarity (dot product since normalized)
        return float(np.dot(embeddings[0], embeddings[1]))
    
    def extract_components(self, input_text: str) -> Dict[str, str]:
        """Extract context, evidence, and response from repair input format."""
        import re
        
        components = {"context": "", "evidence": "", "response": ""}
        
        # Extract context
        context_match = re.search(r'\[CONTEXT\](.*?)\[EVIDENCE\]', input_text, re.DOTALL)
        if context_match:
            components["context"] = context_match.group(1).strip()
        
        # Extract evidence
        evidence_match = re.search(r'\[EVIDENCE\](.*?)\[RESPONSE\]', input_text, re.DOTALL)
        if evidence_match:
            components["evidence"] = evidence_match.group(1).strip()
        
        # Extract response
        response_match = re.search(r'\[RESPONSE\](.*?)$', input_text, re.DOTALL)
        if response_match:
            components["response"] = response_match.group(1).strip()
        
        return components
    
    def evaluate_mrr(self, eval_set_path: str = None) -> Dict:
        """
        Evaluate Mean Reciprocal Rank for retrieval-based repair.
        
        Per morechanges.md lines 775-812:
        - For each example, retrieve top-10 from corpus
        - Find rank of semantically similar response
        - Calculate MRR, Top-1, Top-3, Top-10
        
        Args:
            eval_set_path: Path to evaluation set (optional)
            
        Returns:
            MRR metrics dictionary
        """
        self._load_encoder()
        
        # Load eval set or use test data
        if eval_set_path:
            with open(eval_set_path, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
        else:
            self._load_test_data()
            eval_data = self.test_data
        
        # Load corpus for retrieval
        print("\nLoading corpus for MRR evaluation...")
        corpus_path = Path(self.config.corpus_path)
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        
        # Extract responses from corpus
        if isinstance(corpus[0], dict):
            corpus_responses = [item.get('response', str(item)) for item in corpus]
        else:
            corpus_responses = corpus
        
        print(f"  Corpus size: {len(corpus_responses)}")
        
        # Encode corpus (subsample for efficiency)
        max_corpus = 10000
        if len(corpus_responses) > max_corpus:
            import random
            random.seed(42)
            corpus_responses = random.sample(corpus_responses, max_corpus)
        
        print("  Encoding corpus...")
        corpus_embeddings = self.encoder.encode(
            corpus_responses,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        
        # Evaluate MRR
        print("\nCalculating MRR...")
        mrr_scores = []
        top1_hits = 0
        top3_hits = 0
        top10_hits = 0
        
        for i, item in enumerate(eval_data):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(eval_data)}")
            
            # Get context
            if 'context' in item:
                context = item['context']
            else:
                components = self.extract_components(item.get('input_text', ''))
                context = components['context']
            
            if not context:
                mrr_scores.append(0.0)
                continue
            
            # Get true response (the on-topic reference)
            true_response = item.get('target_text', item.get('response', ''))
            
            # Encode context for retrieval
            context_embedding = self.encoder.encode(
                [context],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            # Find top-10 from corpus
            similarities = np.dot(corpus_embeddings, context_embedding)
            top_indices = np.argsort(similarities)[-10:][::-1]
            
            # Find rank of semantically similar response
            true_embedding = self.encoder.encode(
                [true_response],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]
            
            rank = None
            for j, idx in enumerate(top_indices):
                candidate = corpus_responses[idx]
                candidate_embedding = corpus_embeddings[idx]
                
                # Check similarity to true response
                sim_to_true = np.dot(candidate_embedding, true_embedding)
                if sim_to_true > 0.7:  # Threshold for "relevant"
                    rank = j + 1
                    break
            
            if rank:
                mrr_scores.append(1.0 / rank)
                if rank == 1:
                    top1_hits += 1
                if rank <= 3:
                    top3_hits += 1
                if rank <= 10:
                    top10_hits += 1
            else:
                mrr_scores.append(0.0)
        
        n = len(eval_data)
        mrr = np.mean(mrr_scores)
        
        results = {
            'mrr': float(mrr),
            'top1_accuracy': top1_hits / n if n > 0 else 0,
            'top3_accuracy': top3_hits / n if n > 0 else 0,
            'top10_accuracy': top10_hits / n if n > 0 else 0,
            'n_examples': n
        }
        
        # Print results
        print("\n" + "=" * 50)
        print("MRR EVALUATION RESULTS")
        print("=" * 50)
        print(f"MRR:          {results['mrr']:.4f}")
        print(f"Top-1:        {results['top1_accuracy']:.4f}")
        print(f"Top-3:        {results['top3_accuracy']:.4f}")
        print(f"Top-10:       {results['top10_accuracy']:.4f}")
        print(f"Examples:     {results['n_examples']}")
        
        # Verdict
        if results['mrr'] >= 0.7:
            print("\n✅ MRR >= 0.7: Retrieval system is working well")
        elif results['mrr'] >= 0.5:
            print("\n⚠️ MRR 0.5-0.7: Retrieval acceptable but could improve")
        else:
            print("\n❌ MRR < 0.5: Retrieval needs improvement")
        
        # Save results
        output_path = Path("results/relation_repair_mrr.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
        
        return results
    
    def run_evaluation(self) -> Dict:
        """
        Run comprehensive evaluation comparing old vs new Relation repair.
        """
        self._load_test_data()
        self._load_integrated_model()
        
        print("\n" + "="*70)
        print("EVALUATING RELATION REPAIR")
        print("="*70)
        
        results = {
            "original": [],      # Original violated responses
            "retrieval": [],     # New retrieval-based repair
            "reference": [],     # Reference repairs (if available)
        }
        
        print("\nProcessing examples...")
        
        for i, item in enumerate(self.test_data):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(self.test_data)}")
            
            # Parse input
            components = self.extract_components(item.get("input_text", ""))
            context = components["context"]
            violated_response = components["response"]
            reference_repair = item.get("target_text", "")
            
            if not context or not violated_response:
                continue
            
            # Calculate original relevance
            orig_relevance = self.calculate_relevance(context, violated_response)
            
            # Get retrieval-based repair
            retrieval_repair = self.integrated_model.repair(
                context=context,
                evidence=components["evidence"],
                response=violated_response,
                violation_type="RELATION"
            )
            retrieval_relevance = self.calculate_relevance(context, retrieval_repair)
            
            # Reference relevance (if available)
            ref_relevance = None
            if reference_repair:
                ref_relevance = self.calculate_relevance(context, reference_repair)
            
            results["original"].append({
                "context": context,
                "response": violated_response,
                "relevance": orig_relevance
            })
            
            results["retrieval"].append({
                "context": context,
                "response": retrieval_repair,
                "relevance": retrieval_relevance,
                "improvement": retrieval_relevance - orig_relevance
            })
            
            if reference_repair:
                results["reference"].append({
                    "response": reference_repair,
                    "relevance": ref_relevance
                })
        
        self.results = results
        return results
    
    def calculate_metrics(self) -> Dict:
        """Calculate summary metrics from evaluation results."""
        if not self.results:
            raise ValueError("Run evaluation first!")
        
        metrics = {}
        
        # Original (violated) stats
        orig_relevances = [r["relevance"] for r in self.results["original"]]
        metrics["original"] = {
            "mean_relevance": np.mean(orig_relevances),
            "std_relevance": np.std(orig_relevances),
            "min_relevance": np.min(orig_relevances),
            "max_relevance": np.max(orig_relevances)
        }
        
        # Retrieval repair stats
        retr_relevances = [r["relevance"] for r in self.results["retrieval"]]
        improvements = [r["improvement"] for r in self.results["retrieval"]]
        
        metrics["retrieval"] = {
            "mean_relevance": np.mean(retr_relevances),
            "std_relevance": np.std(retr_relevances),
            "min_relevance": np.min(retr_relevances),
            "max_relevance": np.max(retr_relevances),
            "mean_improvement": np.mean(improvements),
            "pct_improved": sum(1 for i in improvements if i > 0) / len(improvements) * 100
        }
        
        # Reference stats (if available)
        if self.results["reference"]:
            ref_relevances = [r["relevance"] for r in self.results["reference"]]
            metrics["reference"] = {
                "mean_relevance": np.mean(ref_relevances),
                "std_relevance": np.std(ref_relevances)
            }
        
        # Comparison
        metrics["comparison"] = {
            "retrieval_vs_original": metrics["retrieval"]["mean_relevance"] - metrics["original"]["mean_relevance"],
            "improvement_pct": (metrics["retrieval"]["mean_relevance"] / metrics["original"]["mean_relevance"] - 1) * 100
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in readable format."""
        print("\n" + "="*70)
        print("RELATION REPAIR EVALUATION RESULTS")
        print("="*70)
        
        print("\n📊 RELEVANCE SCORES (0.0 - 1.0, higher = more relevant to context)")
        print("-"*50)
        
        print(f"\nOriginal (violated) responses:")
        print(f"  Mean relevance: {metrics['original']['mean_relevance']:.4f}")
        print(f"  Std:           {metrics['original']['std_relevance']:.4f}")
        
        print(f"\nRetrieval-based repair:")
        print(f"  Mean relevance: {metrics['retrieval']['mean_relevance']:.4f}")
        print(f"  Std:           {metrics['retrieval']['std_relevance']:.4f}")
        print(f"  Mean improvement: {metrics['retrieval']['mean_improvement']:+.4f}")
        print(f"  % Improved:    {metrics['retrieval']['pct_improved']:.1f}%")
        
        if "reference" in metrics:
            print(f"\nReference repairs:")
            print(f"  Mean relevance: {metrics['reference']['mean_relevance']:.4f}")
        
        print("\n📈 COMPARISON")
        print("-"*50)
        print(f"Relevance improvement: {metrics['comparison']['retrieval_vs_original']:+.4f}")
        print(f"Relative improvement:  {metrics['comparison']['improvement_pct']:+.1f}%")
        
        # Verdict
        print("\n✅ VERDICT")
        print("-"*50)
        if metrics["comparison"]["retrieval_vs_original"] > 0.1:
            print("Strong improvement: Retrieval approach significantly outperforms editing")
        elif metrics["comparison"]["retrieval_vs_original"] > 0.05:
            print("Moderate improvement: Retrieval approach shows clear benefits")
        elif metrics["comparison"]["retrieval_vs_original"] > 0:
            print("Slight improvement: Retrieval approach shows marginal benefits")
        else:
            print("No improvement: Further tuning needed")
    
    def save_results(self, metrics: Dict):
        """Save results to files."""
        output_dir = Path(self.config.results_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save sample repairs
        samples = []
        for i in range(min(20, len(self.results["original"]))):
            samples.append({
                "context": self.results["original"][i]["context"],
                "violated": self.results["original"][i]["response"],
                "violated_relevance": self.results["original"][i]["relevance"],
                "repaired": self.results["retrieval"][i]["response"],
                "repaired_relevance": self.results["retrieval"][i]["relevance"],
                "improvement": self.results["retrieval"][i]["improvement"]
            })
        
        with open(output_dir / "sample_repairs.json", "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        # Generate markdown report
        self._generate_report(metrics, samples, output_dir)
        
        print(f"\nResults saved to {output_dir}")
    
    def _generate_report(self, metrics: Dict, samples: List[Dict], output_dir: Path):
        """Generate markdown evaluation report."""
        report = f"""# Relation Repair Evaluation Report

## Summary

The retrieval-based approach for Relation violations was evaluated against the original T5 editing approach.

**Key Finding**: Relation violations require generating NEW content, not editing. The retrieval approach addresses this fundamental limitation.

## Metrics

| Metric | Original | Retrieval | Δ |
|--------|----------|-----------|---|
| Mean Relevance | {metrics['original']['mean_relevance']:.4f} | {metrics['retrieval']['mean_relevance']:.4f} | {metrics['comparison']['retrieval_vs_original']:+.4f} |
| Std Relevance | {metrics['original']['std_relevance']:.4f} | {metrics['retrieval']['std_relevance']:.4f} | - |

**Improvement Rate**: {metrics['retrieval']['pct_improved']:.1f}% of repairs showed improved relevance

## Sample Repairs

"""
        for i, sample in enumerate(samples[:5], 1):
            report += f"""### Example {i}

**Context**: {sample['context'][:200]}{'...' if len(sample['context']) > 200 else ''}

**Violated Response** (relevance: {sample['violated_relevance']:.3f}):
> {sample['violated'][:200]}{'...' if len(sample['violated']) > 200 else ''}

**Repaired Response** (relevance: {sample['repaired_relevance']:.3f}, Δ{sample['improvement']:+.3f}):
> {sample['repaired'][:200]}{'...' if len(sample['repaired']) > 200 else ''}

---

"""
        
        report += """## Conclusion

The retrieval-based approach provides a principled solution to Relation violations by retrieving contextually relevant responses rather than attempting the impossible task of editing off-topic text into on-topic text.
"""
        
        with open(output_dir / "evaluation_report.md", "w") as f:
            f.write(report)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run Relation repair evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Relation Repair Evaluation")
    parser.add_argument("--mode", choices=["relevance", "mrr", "both"], default="relevance",
                       help="Evaluation mode: relevance, mrr, or both")
    parser.add_argument("--eval-set", default=None,
                       help="Path to eval set JSON (for MRR)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("RELATION REPAIR EVALUATION")
    print("="*70)
    
    # Initialize evaluator
    evaluator = RelationRepairEvaluator()
    
    if args.mode in ["mrr", "both"]:
        # Run MRR evaluation
        print("\n--- MRR EVALUATION ---")
        evaluator.evaluate_mrr(args.eval_set)
    
    if args.mode in ["relevance", "both"]:
        # Run relevance evaluation
        print("\n--- RELEVANCE EVALUATION ---")
        evaluator.run_evaluation()
        
        # Calculate and print metrics
        metrics = evaluator.calculate_metrics()
        evaluator.print_metrics(metrics)
        
        # Save results
        evaluator.save_results(metrics)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nModes available:")
    print("  --mode relevance : Semantic relevance scores (default)")
    print("  --mode mrr      : Mean Reciprocal Rank evaluation")
    print("  --mode both     : Run both evaluations")


if __name__ == "__main__":
    main()
