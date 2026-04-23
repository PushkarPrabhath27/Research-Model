"""
GriceBench Data Processing Utilities
Helper functions to extract (context, evidence, response) tuples from datasets.
As described in Implementation Guide, Chapter 3.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
TOPICAL_CHAT_DIR = DATA_RAW_DIR / "topical_chat"


def load_json(filepath: Path) -> Optional[dict]:
    """Load JSON file safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_topical_chat(split: str = "train") -> Optional[dict]:
    """
    Load a Topical-Chat split.
    
    Args:
        split: One of "train", "valid_freq", "valid_rare", "test_freq", "test_rare"
        
    Returns:
        Dictionary of conversations keyed by conversation ID
    """
    filepath = TOPICAL_CHAT_DIR / f"{split}.json"
    return load_json(filepath)


def load_reading_sets(split: str = "train") -> Optional[dict]:
    """
    Load reading sets (knowledge/evidence) for a split.
    
    Args:
        split: One of "train", "valid_freq", "valid_rare", "test_freq", "test_rare"
        
    Returns:
        Dictionary of reading sets keyed by article/topic ID
    """
    filepath = TOPICAL_CHAT_DIR / "reading_sets" / f"{split}.json"
    return load_json(filepath)


def extract_training_examples(
    conversations: dict,
    reading_sets: Optional[dict] = None,
    max_context_turns: int = 3,
    min_response_words: int = 5
) -> List[Dict]:
    """
    Extract (context, evidence, response) tuples from conversations.
    
    This is the core function that creates training examples for GriceBench.
    Each example contains:
    - context: Recent conversation history (up to max_context_turns)
    - evidence: Knowledge snippet(s) available for the response
    - response: The actual response text
    - speaker: Which agent gave the response
    
    Args:
        conversations: Dictionary of conversations from Topical-Chat
        reading_sets: Optional reading sets for richer evidence
        max_context_turns: Maximum number of previous turns to include as context
        min_response_words: Minimum response length to include
        
    Returns:
        List of training examples as dictionaries
    """
    examples = []
    
    for conv_id, conv_data in conversations.items():
        content = conv_data.get('content', [])
        
        for i, turn in enumerate(content):
            message = turn.get('message', '')
            agent = turn.get('agent', '')
            knowledge_source = turn.get('knowledge_source')
            
            # Skip very short responses
            if len(message.split()) < min_response_words:
                continue
            
            # Build context from previous turns
            context_turns = content[max(0, i - max_context_turns):i]
            context = []
            for ct in context_turns:
                speaker = ct.get('agent', 'unknown')
                text = ct.get('message', '')
                context.append({
                    'speaker': speaker,
                    'text': text
                })
            
            # Extract evidence
            evidence = knowledge_source if knowledge_source and knowledge_source != 'NULL' else None
            
            example = {
                'conversation_id': conv_id,
                'turn_index': i,
                'context': context,
                'context_text': ' '.join([f"[{c['speaker']}]: {c['text']}" for c in context]),
                'evidence': evidence,
                'response': message,
                'speaker': agent
            }
            
            examples.append(example)
    
    return examples


def create_train_val_test_split(
    examples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split examples into train/validation/test sets.
    
    Ensures no conversation appears in multiple splits to prevent data leakage.
    
    Args:
        examples: List of training examples
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    import random
    random.seed(seed)
    
    # Group examples by conversation
    conv_groups = {}
    for ex in examples:
        conv_id = ex['conversation_id']
        if conv_id not in conv_groups:
            conv_groups[conv_id] = []
        conv_groups[conv_id].append(ex)
    
    # Shuffle conversation IDs
    conv_ids = list(conv_groups.keys())
    random.shuffle(conv_ids)
    
    # Split by conversation
    n_convs = len(conv_ids)
    n_train = int(n_convs * train_ratio)
    n_val = int(n_convs * val_ratio)
    
    train_ids = set(conv_ids[:n_train])
    val_ids = set(conv_ids[n_train:n_train + n_val])
    test_ids = set(conv_ids[n_train + n_val:])
    
    # Assign examples to splits
    train_examples = []
    val_examples = []
    test_examples = []
    
    for ex in examples:
        conv_id = ex['conversation_id']
        if conv_id in train_ids:
            train_examples.append(ex)
        elif conv_id in val_ids:
            val_examples.append(ex)
        else:
            test_examples.append(ex)
    
    return train_examples, val_examples, test_examples


def save_examples(examples: List[Dict], filepath: Path) -> None:
    """Save examples to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(examples)} examples to {filepath}")


def main():
    """Generate and save training examples."""
    print("Loading Topical-Chat training data...")
    conversations = load_topical_chat("train")
    
    if not conversations:
        print("Failed to load conversations!")
        return
    
    print(f"Loaded {len(conversations)} conversations")
    
    print("\nExtracting training examples...")
    examples = extract_training_examples(conversations)
    print(f"Extracted {len(examples)} examples")
    
    print("\nCreating train/val/test splits...")
    train, val, test = create_train_val_test_split(examples)
    print(f"  Train: {len(train)} examples")
    print(f"  Val: {len(val)} examples")
    print(f"  Test: {len(test)} examples")
    
    # Save to data_processed folder
    processed_dir = PROJECT_ROOT / "data_processed"
    save_examples(train, processed_dir / "train_examples.json")
    save_examples(val, processed_dir / "val_examples.json")
    save_examples(test, processed_dir / "test_examples.json")
    
    print("\nDone! Training examples ready for violation injection.")


if __name__ == "__main__":
    main()
