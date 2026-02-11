"""
GriceBench Repair Data Preparation - Chapter 10
================================================

Prepare training data for the repair model by:
1. Reversing violation injection data (violated → clean)
2. Adding control tokens for violation types
3. Creating input-output pairs for T5 training
4. Splitting into train/val/test sets

Based on Chapter 10 of the Implementation Guide.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


# ============================================================================
# CONTROL TOKENS
# ============================================================================

CONTROL_TOKENS = {
    'task': '[REPAIR]',
    'context': '[CONTEXT]',
    'evidence': '[EVIDENCE]',
    'response': '[RESPONSE]',
}

VIOLATION_TOKENS = {
    'quantity': '[VIOLATION=QUANTITY]',
    'quality': '[VIOLATION=QUALITY]',
    'relation': '[VIOLATION=RELATION]',
    'manner': '[VIOLATION=MANNER]',
}


# ============================================================================
# DATA PREPARATION
# ============================================================================

def format_repair_input(
    context: str,
    evidence: str,
    violated_response: str,
    violation_types: List[str]
) -> str:
    """
    Format input for repair model with control tokens.
    
    Format: [REPAIR] [VIOLATION=X] [CONTEXT] ... [EVIDENCE] ... [RESPONSE] ...
    """
    # Task token
    parts = [CONTROL_TOKENS['task']]
    
    # Violation tokens (can be multiple)
    for v_type in violation_types:
        if v_type in VIOLATION_TOKENS:
            parts.append(VIOLATION_TOKENS[v_type])
    
    # Context
    parts.append(CONTROL_TOKENS['context'])
    parts.append(context if isinstance(context, str) else str(context))
    
    # Evidence
    parts.append(CONTROL_TOKENS['evidence'])
    if isinstance(evidence, dict):
        evidence = ' '.join(str(v) for v in evidence.values())
    elif isinstance(evidence, list):
        evidence = ' '.join(str(e) for e in evidence)
    parts.append(str(evidence))
    
    # Response to repair
    parts.append(CONTROL_TOKENS['response'])
    parts.append(violated_response)
    
    return ' '.join(parts)


def prepare_repair_example(example: Dict) -> Dict:
    """
    Convert violation injection example to repair training example.
    
    Input example from violation injection:
    - original_response (clean)
    - violated_response
    - violation_type
    - maxim
    - context, evidence, etc.
    
    Output for repair training:
    - input_text: violated response + control tokens
    - target_text: original response (clean)
    """
    # Extract fields
    context = example.get('context_text', example.get('context', ''))
    if isinstance(context, list):
        context = ' '.join(str(c) for c in context)
    
    evidence = example.get('evidence', '')
    violated_response = example.get('violated_response', '')
    original_response = example.get('original_response', '')
    
    # Determine violation types for this example
    maxim = example.get('maxim', 'unknown')
    violation_type = example.get('violation_type', 'none')
    
    # Handle multi-maxim violations
    if maxim == 'multiple':
        # Extract all violated maxims from the example
        violation_types = []
        for m in ['quantity', 'quality', 'relation', 'manner']:
            if m in violation_type.lower():
                violation_types.append(m)
    else:
        violation_types = [maxim] if maxim != 'none' else []
    
    # Create input with control tokens
    input_text = format_repair_input(
        context=context,
        evidence=evidence,
        violated_response=violated_response,
        violation_types=violation_types
    )
    
    # Target is the clean response
    target_text = original_response
    
    return {
        'input_text': input_text,
        'target_text': target_text,
        'violation_types': violation_types,
        'maxim': maxim,
        'source_id': example.get('source_example_id', 'unknown')
    }


def create_repair_dataset(
    injection_data: List[Dict],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create train/val/test splits for repair training.
    
    Returns:
        (train, val, test)
    """
    random.seed(random_seed)
    
    # Prepare all examples
    print("\n📊 Preparing repair examples...")
    repair_examples = []
    skipped = 0
    
    for ex in injection_data:
        try:
            # Skip examples without violations
            if ex.get('maxim') == 'none' or ex.get('violation_type') == 'none':
                skipped += 1
                continue
            
            # Skip if no original response
            if not ex.get('original_response'):
                skipped += 1
                continue
            
            repair_ex = prepare_repair_example(ex)
            repair_examples.append(repair_ex)
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"✅ Prepared {len(repair_examples):,} repair examples")
    print(f"   Skipped {skipped:,} examples (no violation or missing data)")
    
    # Shuffle
    random.shuffle(repair_examples)
    
    # Split
    n_total = len(repair_examples)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_test - n_val
    
    test = repair_examples[:n_test]
    val = repair_examples[n_test:n_test + n_val]
    train = repair_examples[n_test + n_val:]
    
    print(f"\n📦 Dataset splits:")
    print(f"   Train: {len(train):,} examples")
    print(f"   Val:   {len(val):,} examples")
    print(f"   Test:  {len(test):,} examples")
    
    return train, val, test


def analyze_dataset(examples: List[Dict], split_name: str):
    """Analyze and print dataset statistics."""
    print(f"\n📈 {split_name} Statistics:")
    
    # Count by violation type
    violation_counts = defaultdict(int)
    for ex in examples:
        for v_type in ex['violation_types']:
            violation_counts[v_type] += 1
    
    print(f"   Violation distribution:")
    for v_type, count in sorted(violation_counts.items()):
        pct = count / len(examples) * 100
        print(f"      {v_type:10s}: {count:5d} ({pct:5.1f}%)")
    
    # Multi-violation count
    multi = sum(1 for ex in examples if len(ex['violation_types']) > 1)
    print(f"   Multi-violation: {multi:5d} ({multi/len(examples)*100:5.1f}%)")
    
    # Average lengths
    avg_input_len = sum(len(ex['input_text'].split()) for ex in examples) / len(examples)
    avg_target_len = sum(len(ex['target_text'].split()) for ex in examples) / len(examples)
    print(f"   Avg input length:  {avg_input_len:.1f} words")
    print(f"   Avg target length: {avg_target_len:.1f} words")


def save_repair_data(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
    output_dir: Path
):
    """Save repair datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits = {'train': train, 'val': val, 'test': test}
    
    for split_name, data in splits.items():
        output_path = output_dir / f'repair_{split_name}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        file_size = output_path.stat().st_size / 1e6
        print(f"✅ Saved {split_name:5s}: {output_path} ({file_size:.1f} MB)")
    
    # Save control tokens config
    config = {
        'control_tokens': CONTROL_TOKENS,
        'violation_tokens': VIOLATION_TOKENS,
        'all_tokens': list(CONTROL_TOKENS.values()) + list(VIOLATION_TOKENS.values())
    }
    
    config_path = output_dir / 'control_tokens.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Saved config: {config_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Prepare repair training data from violation injection data."""
    print("="*70)
    print("GRICEBENCH REPAIR DATA PREPARATION - CHAPTER 10")
    print("="*70)
    
    # Paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / 'data_processed' / 'gricebench_weak_50k.json'
    output_dir = project_root / 'data_processed' / 'repair_data'
    
    # Load violation injection data
    print(f"\n📂 Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        injection_data = json.load(f)
    
    print(f"✅ Loaded {len(injection_data):,} violation examples")
    
    # Create repair dataset
    train, val, test = create_repair_dataset(injection_data)
    
    # Analyze
    for split_name, data in [('Train', train), ('Val', val), ('Test', test)]:
        analyze_dataset(data, split_name)
    
    # Save
    print(f"\n💾 Saving repair data to {output_dir}...")
    save_repair_data(train, val, test, output_dir)
    
    print(f"\n{'='*70}")
    print("✅ REPAIR DATA PREPARATION COMPLETE!")
    print('='*70)
    print(f"\nNext steps:")
    print(f"  1. Review: data_processed/repair_data/")
    print(f"  2. Implement: T5 repair model (scripts/repair_model.py)")
    print(f"  3. Train: on Kaggle GPU")


if __name__ == "__main__":
    main()
