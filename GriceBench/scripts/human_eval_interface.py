"""
GriceBench Human Evaluation - CLI Interface - Part 2, Step 1
=============================================================

Free command-line human evaluation interface for GriceBench.
For use when Gradio is not available or preferred.

Evaluates responses on 5 dimensions:
1. Helpfulness - How helpful is the response?
2. Accuracy - How accurate/truthful is the information?
3. Relevance - How relevant is the response to the context?
4. Clarity - How clear and well-organized is the response?
5. Conciseness - Is the response an appropriate length?

Author: GriceBench
"""

import json
import random
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvalConfig:
    """Configuration for human evaluation."""
    test_samples_path: str = "human_eval_samples.json"
    output_dir: str = "human_eval_results"
    num_samples: int = 50
    save_frequency: int = 10  # Auto-save every N samples


# ============================================================================
# EVALUATION DIMENSIONS
# ============================================================================

EVALUATION_DIMENSIONS = {
    "helpfulness": {
        "question": "How helpful is this response in addressing the question/context?",
        "scale": "(1=Not helpful at all, 2=Slightly helpful, 3=Moderately helpful, 4=Very helpful, 5=Extremely helpful)"
    },
    "accuracy": {
        "question": "How accurate/truthful is the information in this response?",
        "scale": "(1=Completely incorrect, 2=Mostly incorrect, 3=Mixed, 4=Mostly accurate, 5=Completely accurate)"
    },
    "relevance": {
        "question": "How relevant is this response to the question/context?",
        "scale": "(1=Completely off-topic, 2=Slightly relevant, 3=Moderately relevant, 4=Mostly relevant, 5=Directly relevant)"
    },
    "clarity": {
        "question": "How clear and well-organized is this response?",
        "scale": "(1=Very confusing, 2=Somewhat confusing, 3=Neutral, 4=Clear, 5=Very clear)"
    },
    "conciseness": {
        "question": "Is the response an appropriate length?",
        "scale": "(1=Way too long/short, 2=Somewhat too long/short, 3=About right, 4=Good length, 5=Perfect length)"
    }
}


# ============================================================================
# HUMAN EVALUATION INTERFACE
# ============================================================================

class HumanEvaluationInterface:
    """
    Free human evaluation interface using command line.
    Can be adapted for web interface using Gradio (free).
    """
    
    def __init__(self, config: EvalConfig = None):
        self.config = config or EvalConfig()
        self.samples = []
        self.results = []
        self.current_idx = 0
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_samples(self, path: str = None):
        """Load samples to evaluate."""
        path = path or self.config.test_samples_path
        
        if not Path(path).exists():
            print(f"Error: Sample file not found: {path}")
            print("Run prepare_human_eval_samples.py first to create samples.")
            return False
        
        with open(path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        
        # Shuffle for randomization
        random.shuffle(self.samples)
        
        print(f"Loaded {len(self.samples)} samples for evaluation")
        return True
    
    def run_evaluation(self, annotator_id: str, num_samples: int = None):
        """Run evaluation for one annotator."""
        num_samples = num_samples or self.config.num_samples
        num_samples = min(num_samples, len(self.samples))
        
        self._print_header(annotator_id, num_samples)
        self._print_instructions()
        
        samples_to_eval = self.samples[:num_samples]
        
        for i, sample in enumerate(samples_to_eval):
            self.current_idx = i
            
            # Display sample
            self._display_sample(i, num_samples, sample)
            
            # Get ratings
            ratings = self._get_ratings()
            
            if ratings is None:  # User quit
                break
            
            if ratings == "skip":
                continue
            
            # Record result
            result = {
                "sample_id": sample.get("id", i),
                "context": sample.get("context", ""),
                "evidence": sample.get("evidence", ""),
                "response": sample.get("response", ""),
                "ratings": ratings,
                "annotator_id": annotator_id,
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(result)
            
            # Auto-save
            if len(self.results) % self.config.save_frequency == 0:
                self._save_results(annotator_id)
                print(f"\n[Auto-saved {len(self.results)} results]")
        
        # Final save
        self._save_results(annotator_id)
        self._print_summary()
        
        return self.results
    
    def _print_header(self, annotator_id: str, num_samples: int):
        """Print evaluation header."""
        print("\n" + "="*70)
        print("GRICEBENCH HUMAN EVALUATION SESSION")
        print("="*70)
        print(f"Annotator ID: {annotator_id}")
        print(f"Samples to evaluate: {num_samples}")
        print(f"Output directory: {self.config.output_dir}")
        print("="*70 + "\n")
    
    def _print_instructions(self):
        """Print evaluation instructions."""
        print("INSTRUCTIONS:")
        print("-" * 50)
        print("• You will see a context and a response")
        print("• Rate each dimension from 1 to 5")
        print("• Type 'skip' to skip a sample")
        print("• Type 'quit' to save and exit")
        print("• Results are auto-saved every 10 samples")
        print("-" * 50 + "\n")
    
    def _display_sample(self, idx: int, total: int, sample: Dict):
        """Display a sample for evaluation."""
        print("\n" + "="*70)
        print(f"SAMPLE {idx + 1} / {total}")
        print("="*70)
        
        print(f"\n📝 CONTEXT:")
        print(f"   {sample.get('context', 'N/A')[:500]}")
        
        if sample.get("evidence"):
            print(f"\n📚 EVIDENCE:")
            print(f"   {sample['evidence'][:300]}")
        
        print(f"\n💬 RESPONSE TO EVALUATE:")
        print(f"   {sample.get('response', 'N/A')[:500]}")
        print()
    
    def _get_ratings(self) -> Optional[Dict]:
        """Get ratings for all dimensions from user."""
        ratings = {}
        
        for dim_name, dim_info in EVALUATION_DIMENSIONS.items():
            while True:
                print(f"\n{dim_info['question']}")
                print(f"   {dim_info['scale']}")
                
                try:
                    user_input = input(f"   Rating for {dim_name} (1-5): ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\n\nExiting...")
                    return None
                
                if user_input == "skip":
                    return "skip"
                
                if user_input == "quit":
                    return None
                
                try:
                    rating = int(user_input)
                    if 1 <= rating <= 5:
                        ratings[dim_name] = rating
                        break
                    else:
                        print("   ⚠️ Please enter a number between 1 and 5")
                except ValueError:
                    print("   ⚠️ Please enter a valid number (1-5), 'skip', or 'quit'")
        
        return ratings
    
    def _save_results(self, annotator_id: str):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{annotator_id}_{timestamp}.json"
        filepath = Path(self.config.output_dir) / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to {filepath}")
    
    def _print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70)
        print(f"Total samples evaluated: {len(self.results)}")
        
        if self.results:
            # Calculate average ratings
            for dim in EVALUATION_DIMENSIONS:
                ratings = [r["ratings"].get(dim) for r in self.results if r["ratings"].get(dim)]
                if ratings:
                    avg = sum(ratings) / len(ratings)
                    print(f"  Average {dim}: {avg:.2f}")
        
        print("\nThank you for your participation!")
        print("="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run human evaluation CLI."""
    print("\n" + "="*70)
    print("GRICEBENCH HUMAN EVALUATION - CLI")
    print("="*70)
    
    # Get annotator ID
    annotator_id = input("\nEnter your Annotator ID: ").strip()
    if not annotator_id:
        annotator_id = f"anon_{datetime.now().strftime('%H%M%S')}"
        print(f"Using anonymous ID: {annotator_id}")
    
    # Get number of samples
    try:
        num = input("Number of samples to evaluate (default: 50): ").strip()
        num_samples = int(num) if num else 50
    except ValueError:
        num_samples = 50
    
    # Initialize interface
    config = EvalConfig(num_samples=num_samples)
    interface = HumanEvaluationInterface(config)
    
    # Load samples
    if not interface.load_samples():
        return
    
    # Run evaluation
    interface.run_evaluation(annotator_id, num_samples)


if __name__ == "__main__":
    main()
