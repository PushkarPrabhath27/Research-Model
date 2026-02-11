"""
Create Relation Evaluation Set
==============================

Creates a stratified sample of 200 examples with Relation violations
for MRR evaluation per morechanges.md lines 746-769.

Author: GriceBench
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def create_relation_eval_set(
    test_data_path: str = "data_processed/repair_data/repair_test.json",
    output_path: str = "data_processed/relation_eval_set.json",
    num_examples: int = 200,
    seed: int = 42
) -> List[Dict]:
    """
    Sample 200 examples with Relation violations for evaluation.
    
    Args:
        test_data_path: Path to repair test data
        output_path: Output path for eval set
        num_examples: Number of examples to sample
        seed: Random seed for reproducibility
    
    Returns:
        List of sampled examples
    """
    random.seed(seed)
    
    print("=" * 70)
    print("CREATE RELATION EVALUATION SET")
    print("=" * 70)
    
    # Load test data
    print(f"\nLoading test data from {test_data_path}...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"  Total examples: {len(test_data)}")
    
    # Filter for Relation violations
    relation_examples = []
    for i, item in enumerate(test_data):
        input_text = item.get("input_text", "")
        if "[VIOLATION=RELATION]" in input_text:
            # Extract components
            example = {
                "id": f"relation_eval_{i}",
                "input_text": input_text,
                "target_text": item.get("target_text", ""),
                "source_index": i
            }
            
            # Try to extract context and response
            if "[CONTEXT]" in input_text and "[RESPONSE]" in input_text:
                import re
                context_match = re.search(r'\[CONTEXT\](.*?)\[', input_text, re.DOTALL)
                response_match = re.search(r'\[RESPONSE\](.*?)$', input_text, re.DOTALL)
                
                if context_match:
                    example["context"] = context_match.group(1).strip()
                if response_match:
                    example["response"] = response_match.group(1).strip()
            
            relation_examples.append(example)
    
    print(f"  Relation violations found: {len(relation_examples)}")
    
    if len(relation_examples) < num_examples:
        print(f"  WARNING: Only {len(relation_examples)} examples available")
        num_examples = len(relation_examples)
    
    # Sample stratified if possible
    sampled = random.sample(relation_examples, num_examples)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(sampled)} examples to {output_path}")
    
    return sampled


if __name__ == "__main__":
    create_relation_eval_set()
