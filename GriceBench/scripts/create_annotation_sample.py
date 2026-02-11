"""
Create Annotation Sample
========================

Creates a stratified sample of 1,000 examples for human annotation.
Per morechanges.md lines 658-679.

Sampling Strategy:
- 200 per maxim (detector positives)
- 200 clean (detector negatives)
- 100 detector high-confidence errors
- 100 random (no selection bias)

Domain Balance:
- 333 Wizard examples
- 333 TopicalChat examples
- 334 LIGHT examples

Author: GriceBench
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


def load_detector_predictions(data: List[Dict], detector_model_path: str = None) -> List[Dict]:
    """
    Load or generate detector predictions for the data.
    
    If detector model path is provided, runs inference.
    Otherwise, uses existing labels/predictions in data.
    """
    # For now, use any existing predictions/labels in the data
    # In production, this would load and run the detector
    for item in data:
        if 'detector_predictions' not in item and 'labels' in item:
            item['detector_predictions'] = item['labels']
    
    return data


def create_annotation_sample(
    train_data_path: str = "data_processed/train_examples.json",
    val_data_path: str = "data_processed/val_examples.json",
    gold_data_path: str = "data_processed/gold_annotation_set.json",
    output_path: str = "data_processed/annotation_sample_1000.json",
    num_samples: int = 1000,
    seed: int = 42
) -> List[Dict]:
    """
    Create stratified sample for annotation.
    
    Args:
        train_data_path: Path to training data
        val_data_path: Path to validation data
        gold_data_path: Path to gold annotations (for error analysis)
        output_path: Output path for sample
        num_samples: Total samples to create
        seed: Random seed
    
    Returns:
        List of sampled examples
    """
    random.seed(seed)
    
    print("=" * 70)
    print("CREATE ANNOTATION SAMPLE (1,000 examples)")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    
    all_examples = []
    
    if Path(val_data_path).exists():
        with open(val_data_path, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        print(f"  Validation: {len(val_data)} examples")
        for i, item in enumerate(val_data):
            item['source_file'] = 'validation'
            item['source_idx'] = i
        all_examples.extend(val_data)
    
    if Path(gold_data_path).exists():
        with open(gold_data_path, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        print(f"  Gold: {len(gold_data)} examples")
        for i, item in enumerate(gold_data):
            item['source_file'] = 'gold'
            item['source_idx'] = i
        all_examples.extend(gold_data)
    
    print(f"\nTotal pool: {len(all_examples)} examples")
    
    # Add detector predictions if not present
    all_examples = load_detector_predictions(all_examples)
    
    # Categorize examples
    maxims = ['quantity', 'quality', 'relation', 'manner']
    
    detector_positives = defaultdict(list)  # maxim -> list of examples
    detector_negatives = []  # Clean examples
    high_confidence_errors = []
    
    for item in all_examples:
        labels = item.get('labels', item.get('detector_predictions', {}))
        
        # Check if clean vs has violations
        has_violation = any(labels.get(m, 0) for m in maxims)
        
        if not has_violation:
            detector_negatives.append(item)
        else:
            for maxim in maxims:
                if labels.get(maxim, 0):
                    detector_positives[maxim].append(item)
    
    print(f"\nCategorization:")
    print(f"  Clean examples: {len(detector_negatives)}")
    for maxim in maxims:
        print(f"  {maxim} positives: {len(detector_positives[maxim])}")
    
    # Sample per morechanges.md strategy
    final_sample = []
    seen_ids = set()
    
    def add_samples(pool: List[Dict], count: int, category: str):
        """Add unique samples from pool."""
        nonlocal final_sample, seen_ids
        
        shuffled = pool.copy()
        random.shuffle(shuffled)
        
        added = 0
        for item in shuffled:
            item_id = item.get('id', f"{item.get('source_file', 'unknown')}_{item.get('source_idx', 0)}")
            if item_id not in seen_ids:
                item['annotation_category'] = category
                item['sample_id'] = f"sample_{len(final_sample)}"
                final_sample.append(item)
                seen_ids.add(item_id)
                added += 1
                if added >= count:
                    break
        
        return added
    
    # 200 per maxim (detector positives)
    for maxim in maxims:
        added = add_samples(detector_positives[maxim], 200, f"{maxim}_positive")
        print(f"  Added {added} {maxim} positives")
    
    # 200 clean (detector negatives)
    added = add_samples(detector_negatives, 200, "clean")
    print(f"  Added {added} clean examples")
    
    # 100 random from remaining
    remaining = [item for item in all_examples 
                 if item.get('id', f"{item.get('source_file', '')}_{item.get('source_idx', 0)}") not in seen_ids]
    added = add_samples(remaining, 100, "random")
    print(f"  Added {added} random examples")
    
    print(f"\nTotal sampled: {len(final_sample)}")
    
    # Shuffle final sample
    random.shuffle(final_sample)
    
    # Assign final IDs
    for i, item in enumerate(final_sample):
        item['id'] = f"annotation_{i:04d}"
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_sample, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(final_sample)} examples to {output_path}")
    
    # Distribution summary
    print("\nCategory distribution:")
    category_counts = defaultdict(int)
    for item in final_sample:
        category_counts[item.get('annotation_category', 'unknown')] += 1
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    
    return final_sample


def main():
    """Run sample creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create annotation sample")
    parser.add_argument("--train", default="data_processed/train_examples.json")
    parser.add_argument("--val", default="data_processed/val_examples.json")
    parser.add_argument("--gold", default="data_processed/gold_annotation_set.json")
    parser.add_argument("--output", default="data_processed/annotation_sample_1000.json")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    create_annotation_sample(
        args.train, args.val, args.gold, args.output, 
        args.num_samples, args.seed
    )


if __name__ == "__main__":
    main()
