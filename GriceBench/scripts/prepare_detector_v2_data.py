"""
Dataset Preparation Script for Detector V2
==========================================

Prepares balanced training data by combining:
1. Phase 4 natural violations (4,000 examples)
2. New realistic synthetic violations (from realistic_injectors.py)
3. Clean examples from original datasets

Creates train/val/test splits with proper balance.

Author: GriceBench Team
Date: 2026-01-27
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import argparse

# Import realistic injectors
try:
    from realistic_injectors import RealisticViolationInjector
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from realistic_injectors import RealisticViolationInjector


def load_phase4_violations(results_dir: Path) -> List[Dict]:
    """Load natural violations from Phase 4 output"""
    phase4_file = results_dir / "phase4output" / "improved_violations.json"
    
    if not phase4_file.exists():
        print(f"Warning: Phase 4 output not found at {phase4_file}")
        return []
    
    with open(phase4_file) as f:
        data = json.load(f)
    
    violations = []
    
    # Handle different possible formats
    if isinstance(data, list):
        violations = data
    elif isinstance(data, dict):
        # Could be organized by maxim
        for maxim, examples in data.items():
            if isinstance(examples, list):
                for ex in examples:
                    ex['source_maxim'] = maxim
                violations.extend(examples)
    
    print(f"Loaded {len(violations)} examples from Phase 4")
    return violations


def load_clean_examples(data_dir: Path, target_count: int = 3000) -> List[Dict]:
    """Load clean examples from original datasets"""
    clean_examples = []
    
    # Try different possible locations
    possible_files = [
        data_dir / "processed" / "training_data.json",
        data_dir / "training_data.json",
        data_dir / "wizard_of_wikipedia" / "train.json",
        data_dir / "topical_chat" / "train.json",
        data_dir / "light" / "train.json",
    ]
    
    for filepath in possible_files:
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        # Only include clean examples (no violations)
                        if isinstance(item, dict):
                            labels = item.get('labels', {})
                            is_clean = (
                                labels.get('quantity', 0) == 0 and
                                labels.get('quality', 0) == 0 and
                                labels.get('relation', 0) == 0 and
                                labels.get('manner', 0) == 0
                            )
                            if is_clean or 'labels' not in item:
                                clean_examples.append({
                                    'context': item.get('context', ''),
                                    'response': item.get('response', item.get('violated_response', '')),
                                    'labels': {'quantity': 0, 'quality': 0, 'relation': 0, 'manner': 0},
                                    'source': 'clean_original'
                                })
                
                print(f"Loaded {len(clean_examples)} clean examples from {filepath}")
                
                if len(clean_examples) >= target_count:
                    break
                    
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    
    # Shuffle and limit
    random.shuffle(clean_examples)
    return clean_examples[:target_count]


def generate_realistic_violations(
    clean_examples: List[Dict],
    target_per_maxim: int = 1000
) -> List[Dict]:
    """Generate new violations using realistic injectors"""
    
    injector = RealisticViolationInjector()
    
    # Prepare input format
    input_examples = []
    for ex in clean_examples:
        response = ex.get('response', ex.get('violated_response', ''))
        context = ex.get('context', '')
        if response and len(response) > 20:
            input_examples.append({
                'context': context,
                'response': response
            })
    
    # Generate violations
    violations = injector.inject_batch(
        input_examples,
        target_per_maxim=target_per_maxim,
        include_clean=False  # We'll add clean separately
    )
    
    print(f"Generated {len(violations)} realistic violations")
    return violations


def balance_dataset(
    phase4_violations: List[Dict],
    realistic_violations: List[Dict],
    clean_examples: List[Dict],
    target_distribution: Dict[str, float] = None
) -> List[Dict]:
    """
    Balance dataset with target distribution:
    - 40% Natural violations (Phase 4)
    - 30% Realistic synthetic (new injectors)
    - 30% Clean examples
    """
    if target_distribution is None:
        target_distribution = {
            'phase4': 0.40,
            'realistic': 0.30,
            'clean': 0.30
        }
    
    # Calculate counts
    total_phase4 = len(phase4_violations)
    total_realistic = len(realistic_violations)
    total_clean = len(clean_examples)
    
    # Determine total size based on smallest balanced category
    min_category = min(
        total_phase4 / target_distribution['phase4'],
        total_realistic / target_distribution['realistic'],
        total_clean / target_distribution['clean']
    )
    
    total_size = int(min_category)
    
    # Sample from each category
    n_phase4 = int(total_size * target_distribution['phase4'])
    n_realistic = int(total_size * target_distribution['realistic'])
    n_clean = int(total_size * target_distribution['clean'])
    
    sampled_phase4 = random.sample(phase4_violations, min(n_phase4, len(phase4_violations)))
    sampled_realistic = random.sample(realistic_violations, min(n_realistic, len(realistic_violations)))
    sampled_clean = random.sample(clean_examples, min(n_clean, len(clean_examples)))
    
    # Combine and shuffle
    combined = sampled_phase4 + sampled_realistic + sampled_clean
    random.shuffle(combined)
    
    print(f"Balanced dataset: {len(sampled_phase4)} Phase4 + {len(sampled_realistic)} Realistic + {len(sampled_clean)} Clean = {len(combined)} total")
    
    return combined


def balance_by_maxim(dataset: List[Dict]) -> List[Dict]:
    """Ensure equal representation of each maxim"""
    by_maxim = defaultdict(list)
    
    for ex in dataset:
        labels = ex.get('labels', {})
        maxim = ex.get('maxim', 'unknown')
        
        # Determine primary maxim from labels
        if isinstance(labels, dict):
            for m in ['quantity', 'quality', 'relation', 'manner']:
                if labels.get(m, 0) == 1:
                    maxim = m
                    break
            if all(labels.get(m, 0) == 0 for m in ['quantity', 'quality', 'relation', 'manner']):
                maxim = 'clean'
        
        by_maxim[maxim].append(ex)
    
    # Print distribution
    print("\nDataset distribution by maxim:")
    for maxim, examples in sorted(by_maxim.items()):
        print(f"  {maxim}: {len(examples)}")
    
    # Balance to minimum count (excluding unknowns)
    min_count = min(len(exs) for m, exs in by_maxim.items() if m != 'unknown')
    
    balanced = []
    for maxim, examples in by_maxim.items():
        if maxim == 'unknown':
            continue
        balanced.extend(random.sample(examples, min(min_count, len(examples))))
    
    random.shuffle(balanced)
    return balanced


def split_dataset(
    dataset: List[Dict],
    splits: Tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split into train/val/test"""
    random.shuffle(dataset)
    
    total = len(dataset)
    train_end = int(total * splits[0])
    val_end = train_end + int(total * splits[1])
    
    train = dataset[:train_end]
    val = dataset[train_end:val_end]
    test = dataset[val_end:]
    
    print(f"\nSplit sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    
    return train, val, test


def normalize_example(example: Dict) -> Dict:
    """Normalize example format for detector training"""
    
    # Get response text
    response = example.get('violated_response', example.get('response', ''))
    context = example.get('context', '')
    
    # Normalize labels
    labels = example.get('labels', {})
    if not isinstance(labels, dict):
        labels = {}
    
    normalized_labels = {
        'quantity': int(labels.get('quantity', 0)),
        'quality': int(labels.get('quality', 0)),
        'relation': int(labels.get('relation', 0)),
        'manner': int(labels.get('manner', 0)),
    }
    
    return {
        'context': context,
        'response': response,
        'labels': normalized_labels,
        'source': example.get('source', 'unknown'),
        'strategy': example.get('strategy', 'unknown'),
    }


def prepare_detector_v2_data(
    base_dir: Path,
    output_dir: Path,
    target_per_maxim: int = 1000,
    target_clean: int = 2000,
):
    """
    Main function to prepare Detector V2 training data
    """
    print("=" * 80)
    print("PREPARING DETECTOR V2 TRAINING DATA")
    print("=" * 80)
    
    results_dir = base_dir / "results"
    data_dir = base_dir / "data"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load Phase 4 natural violations
    print("\n[1/6] Loading Phase 4 natural violations...")
    phase4_violations = load_phase4_violations(results_dir)
    
    # Step 2: Load clean examples
    print("\n[2/6] Loading clean examples...")
    clean_examples = load_clean_examples(data_dir, target_count=target_clean * 2)
    
    # Step 3: Generate realistic violations
    print("\n[3/6] Generating realistic synthetic violations...")
    realistic_violations = generate_realistic_violations(
        clean_examples,
        target_per_maxim=target_per_maxim
    )
    
    # Step 4: Balance dataset
    print("\n[4/6] Balancing dataset...")
    balanced = balance_dataset(
        phase4_violations,
        realistic_violations,
        clean_examples[:target_clean]
    )
    
    # Balance by maxim
    balanced = balance_by_maxim(balanced)
    
    # Step 5: Normalize all examples
    print("\n[5/6] Normalizing examples...")
    normalized = [normalize_example(ex) for ex in balanced]
    
    # Step 6: Split into train/val/test
    print("\n[6/6] Creating splits...")
    train, val, test = split_dataset(normalized)
    
    # Save datasets
    datasets = {
        'train': train,
        'val': val,
        'test': test,
    }
    
    for split_name, data in datasets.items():
        output_file = output_dir / f"detector_v2_{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {split_name}: {len(data)} examples to {output_file}")
    
    # Save combined for easy loading
    combined_file = output_dir / "detector_v2_combined.json"
    with open(combined_file, 'w') as f:
        json.dump({
            'train': train,
            'val': val,
            'test': test,
            'metadata': {
                'total_examples': len(normalized),
                'phase4_count': len(phase4_violations),
                'realistic_count': len(realistic_violations),
                'clean_count': target_clean,
            }
        }, f, indent=2)
    print(f"\nSaved combined dataset to {combined_file}")
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal examples: {len(normalized)}")
    print(f"  Train: {len(train)} ({100*len(train)/len(normalized):.1f}%)")
    print(f"  Val: {len(val)} ({100*len(val)/len(normalized):.1f}%)")
    print(f"  Test: {len(test)} ({100*len(test)/len(normalized):.1f}%)")
    
    # Count by source
    source_counts = defaultdict(int)
    for ex in normalized:
        source_counts[ex.get('source', 'unknown')] += 1
    print("\nBy source:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")
    
    return train, val, test


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for Detector V2"
    )
    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path(__file__).parent.parent,
        help="GriceBench base directory"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help="Output directory for prepared data"
    )
    parser.add_argument(
        '--target-per-maxim',
        type=int,
        default=1000,
        help="Target number of violations per maxim"
    )
    parser.add_argument(
        '--target-clean',
        type=int,
        default=2000,
        help="Target number of clean examples"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.base_dir / "data" / "detector_v2"
    
    prepare_detector_v2_data(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        target_per_maxim=args.target_per_maxim,
        target_clean=args.target_clean,
    )


if __name__ == '__main__':
    main()
