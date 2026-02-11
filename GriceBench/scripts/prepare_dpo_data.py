"""
Chapter 13: Prepare DPO Preference Data
========================================

Convert repair pairs to DPO format for generator training.

DPO (Direct Preference Optimization) requires:
- prompt: Input context + evidence
- chosen: Preferred response (repaired/cooperative)
- rejected: Dispreferred response (violated/uncooperative)
"""

import json
from pathlib import Path
import re


def extract_components(input_text):
    """
    Extract context, evidence, and response from repair input format.
    
    Input format: [REPAIR] [VIOLATION=X] [CONTEXT] ... [EVIDENCE] ... [RESPONSE] ...
    """
    components = {
        'context': '',
        'evidence': '',
        'response': ''
    }
    
    # Extract context
    context_match = re.search(r'\[CONTEXT\](.*?)\[EVIDENCE\]', input_text, re.DOTALL)
    if context_match:
        components['context'] = context_match.group(1).strip()
    
    # Extract evidence
    evidence_match = re.search(r'\[EVIDENCE\](.*?)\[RESPONSE\]', input_text, re.DOTALL)
    if evidence_match:
        components['evidence'] = evidence_match.group(1).strip()
    
    # Extract response (violated)
    response_match = re.search(r'\[RESPONSE\](.*?)$', input_text, re.DOTALL)
    if response_match:
        components['response'] = response_match.group(1).strip()
    
    return components


def create_dpo_dataset():
    """Convert repair training data to DPO preference format."""
    
    print("="*70)
    print("CREATING DPO PREFERENCE DATASET")
    print("="*70)
    
    # Load repair training data
    repair_data_path = Path('data_processed/repair_data/repair_train.json')
    
    if not repair_data_path.exists():
        print(f"\nError: {repair_data_path} not found!")
        print("Make sure you've run data preparation (Chapter 10)")
        return
    
    print(f"\nLoading repair data from {repair_data_path}...")
    with open(repair_data_path, 'r', encoding='utf-8') as f:
        repair_data = json.load(f)
    
    print(f"Loaded {len(repair_data):,} repair examples")
    
    # Convert to DPO format
    dpo_data = []
    skipped = 0
    
    print("\nConverting to DPO format...")
    for idx, example in enumerate(repair_data):
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1:,}/{len(repair_data):,}...")
        
        try:
            # Parse input
            components = extract_components(example['input_text'])
            
            # Skip if parsing failed
            if not components['context'] or not components['response']:
                skipped += 1
                continue
            
            # Create prompt
            if components['evidence']:
                prompt = f"Context: {components['context']}\nEvidence: {components['evidence']}\n\nGenerate a cooperative response:"
            else:
                prompt = f"Context: {components['context']}\n\nGenerate a cooperative response:"
            
            # Create DPO example
            dpo_example = {
                'prompt': prompt,
                'chosen': example['target_text'],  # Repaired (cooperative)
                'rejected': components['response'],  # Violated (uncooperative)
            }
            
            dpo_data.append(dpo_example)
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"\nSuccessfully converted {len(dpo_data):,} examples")
    print(f"Skipped {skipped:,} examples due to parsing errors")
    
    # Create train/val split
    split_ratio = 0.9
    split_point = int(len(dpo_data) * split_ratio)
    
    train_data = dpo_data[:split_point]
    val_data = dpo_data[split_point:]
    
    # Save to files
    output_dir = Path('data_processed/dpo_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'dpo_train.json'
    val_path = output_dir / 'dpo_val.json'
    
    print(f"\nSaving datasets...")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"  Train: {len(train_data):,} examples -> {train_path}")
    print(f"  Val: {len(val_data):,} examples -> {val_path}")
    
    # Show sample
    print(f"\n{'='*70}")
    print("SAMPLE DPO EXAMPLE")
    print('='*70)
    sample = train_data[0]
    print(f"\nPrompt:")
    print(f"{sample['prompt'][:200]}...\n")
    print(f"Chosen (cooperative):")
    print(f"{sample['chosen'][:150]}...\n")
    print(f"Rejected (violated):")
    print(f"{sample['rejected'][:150]}...\n")
    
    # Calculate statistics
    avg_prompt_len = sum(len(ex['prompt'].split()) for ex in train_data) / len(train_data)
    avg_chosen_len = sum(len(ex['chosen'].split()) for ex in train_data) / len(train_data)
    avg_rejected_len = sum(len(ex['rejected'].split()) for ex in train_data) / len(train_data)
    
    print(f"\nDataset Statistics:")
    print(f"  Average prompt length: {avg_prompt_len:.1f} tokens")
    print(f"  Average chosen length: {avg_chosen_len:.1f} tokens")
    print(f"  Average rejected length: {avg_rejected_len:.1f} tokens")
    
    print(f"\n{'='*70}")
    print("DPO DATA PREPARATION COMPLETE!")
    print('='*70)
    print(f"\nNext steps:")
    print(f"1. Upload dpo_data/ folder to Kaggle")
    print(f"2. Follow CHAPTER_13_IMPLEMENTATION.md")
    print(f"3. Run DPO training on Kaggle GPU")


if __name__ == "__main__":
    create_dpo_dataset()
