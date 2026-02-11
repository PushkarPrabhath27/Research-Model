"""
GriceBench Data Preparation for Model Training
===============================================

This module prepares data for training the Gricean violation detector.
Creates input-output pairs in the format required by transformer models.

Based on Chapter 7 of the GriceBench Implementation Guide.
"""

import json
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DataConfig:
    """Configuration for data preparation."""
    max_context_length: int = 256
    max_evidence_length: int = 128
    max_response_length: int = 128
    max_total_length: int = 512
    
    # Special tokens
    context_token: str = "[CONTEXT]"
    evidence_token: str = "[EVIDENCE]"
    response_token: str = "[RESPONSE]"
    
    # Labels
    maxims: List[str] = None
    
    def __post_init__(self):
        if self.maxims is None:
            self.maxims = ['quantity', 'quality', 'relation', 'manner']


# ============================================================================
# INPUT FORMATTING
# ============================================================================

def format_input(
    context: str,
    evidence: str,
    response: str,
    config: DataConfig = None
) -> str:
    """
    Format input for the detector model.
    
    Format: [CONTEXT] {context} [EVIDENCE] {evidence} [RESPONSE] {response}
    
    Args:
        context: Conversation context
        evidence: Available knowledge/evidence
        response: Response to evaluate
        config: Data configuration
        
    Returns:
        Formatted input string
    """
    if config is None:
        config = DataConfig()
    
    # Truncate fields if needed
    context = context[:config.max_context_length * 4]  # Rough char estimate
    evidence = str(evidence)[:config.max_evidence_length * 4]
    response = response[:config.max_response_length * 4]
    
    formatted = f"{config.context_token} {context} {config.evidence_token} {evidence} {config.response_token} {response}"
    
    return formatted


def extract_labels(example: Dict, config: DataConfig = None) -> Dict[str, int]:
    """
    Extract binary labels for each maxim.
    
    Returns:
        Dict mapping maxim name to 0 (no violation) or 1 (violation)
    """
    if config is None:
        config = DataConfig()
    
    # Check if labels already exist
    if 'labels' in example:
        return example['labels']
    
    # Extract from violation_type
    vtype = example.get('violation_type', 'none')
    maxim = example.get('maxim', 'none')
    
    labels = {m: 0 for m in config.maxims}
    
    if maxim in config.maxims:
        labels[maxim] = 1
    elif maxim == 'multiple':
        # Multi-maxim violation
        for m in config.maxims:
            if m in vtype:
                labels[m] = 1
    
    return labels


def prepare_training_example(example: Dict, config: DataConfig = None) -> Dict:
    """
    Prepare a single example for training.
    
    Returns:
        Dict with 'input_text', 'labels', and metadata
    """
    if config is None:
        config = DataConfig()
    
    # Get text fields
    context = example.get('context_text', example.get('context', ''))
    if isinstance(context, list):
        context = ' '.join(str(c) for c in context)
    
    evidence = example.get('evidence', '')
    if isinstance(evidence, dict):
        evidence = ' '.join(str(v) for v in evidence.values())
    elif isinstance(evidence, list):
        evidence = ' '.join(str(e) for e in evidence)
    
    response = example.get('violated_response', example.get('response', ''))
    
    # Format input
    input_text = format_input(context, evidence, response, config)
    
    # Extract labels
    labels = extract_labels(example, config)
    
    return {
        'input_text': input_text,
        'labels': labels,
        'violation_type': example.get('violation_type', 'none'),
        'original_id': example.get('source_example_id', example.get('conversation_id', 'unknown'))
    }


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_dataset(
    examples: List[Dict],
    config: DataConfig = None
) -> List[Dict]:
    """Prepare all examples for training."""
    if config is None:
        config = DataConfig()
    
    prepared = []
    for ex in examples:
        try:
            prepared_ex = prepare_training_example(ex, config)
            prepared.append(prepared_ex)
        except Exception as e:
            continue
    
    return prepared


def create_train_val_split(
    examples: List[Dict],
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Split into train and validation sets."""
    random.seed(random_seed)
    
    # Shuffle
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    # Split
    val_size = int(len(shuffled) * val_ratio)
    val = shuffled[:val_size]
    train = shuffled[val_size:]
    
    return train, val


def calculate_class_weights(examples: List[Dict], config: DataConfig = None) -> Dict[str, float]:
    """
    Calculate class weights to handle imbalance.
    
    Uses inverse frequency weighting.
    """
    if config is None:
        config = DataConfig()
    
    # Count positives per maxim
    counts = {m: {'pos': 0, 'neg': 0} for m in config.maxims}
    
    for ex in examples:
        labels = ex.get('labels', {})
        for m in config.maxims:
            if labels.get(m, 0) == 1:
                counts[m]['pos'] += 1
            else:
                counts[m]['neg'] += 1
    
    # Calculate weights
    weights = {}
    for m in config.maxims:
        pos = counts[m]['pos']
        neg = counts[m]['neg']
        total = pos + neg
        
        if pos > 0 and neg > 0:
            # Weight inversely proportional to class frequency
            weights[m] = {
                'positive': total / (2 * pos),
                'negative': total / (2 * neg)
            }
        else:
            weights[m] = {'positive': 1.0, 'negative': 1.0}
    
    return weights


# ============================================================================
# SAVE/LOAD
# ============================================================================

def save_prepared_data(
    train: List[Dict],
    val: List[Dict],
    output_dir: Path,
    prefix: str = "detector"
) -> None:
    """Save prepared data to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / f"{prefix}_train.json"
    val_path = output_dir / f"{prefix}_val.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val, f, ensure_ascii=False)
    
    print(f"Saved {len(train)} train examples to {train_path}")
    print(f"Saved {len(val)} val examples to {val_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Prepare data for detector training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare data for detector training')
    parser.add_argument('--input', type=str, default='data_processed/gricebench_weak_50k.json')
    parser.add_argument('--output-dir', type=str, default='data_processed/detector_data')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    output_dir = project_root / args.output_dir
    
    # Load data
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")
    
    # Prepare
    config = DataConfig()
    print("\nPreparing examples...")
    prepared = prepare_dataset(examples, config)
    print(f"Prepared {len(prepared)} examples")
    
    # Split
    train, val = create_train_val_split(prepared, val_ratio=args.val_ratio)
    print(f"\nSplit: {len(train)} train, {len(val)} val")
    
    # Calculate class weights
    weights = calculate_class_weights(train, config)
    print("\nClass weights:")
    for m, w in weights.items():
        print(f"  {m}: pos={w['positive']:.2f}, neg={w['negative']:.2f}")
    
    # Save
    save_prepared_data(train, val, output_dir)
    
    # Save weights
    weights_path = output_dir / "class_weights.json"
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"Saved class weights to {weights_path}")
    
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
