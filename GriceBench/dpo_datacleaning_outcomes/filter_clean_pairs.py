"""
Filter scored_data.json for pairs where ALL 4 Gricean maxim margins are positive.
This produces the cleanest possible DPO training set with no conflicting signals.
"""

import json
import csv
from pathlib import Path

def filter_all_positive_margins(input_path: str, output_json: str, output_csv: str):
    """Filter pairs where all 4 margins > 0."""
    
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total pairs in dataset: {len(data)}")
    
    # Filter for all positive margins
    clean_pairs = []
    stats = {
        'total': len(data),
        'quantity_negative': 0,
        'quality_negative': 0,
        'relation_negative': 0,
        'manner_negative': 0,
        'all_positive': 0
    }
    
    for item in data:
        margins = item.get('margins', {})
        
        qty = margins.get('quantity', 0)
        qlt = margins.get('quality', 0)
        rel = margins.get('relation', 0)
        man = margins.get('manner', 0)
        
        # Track which margins are problematic
        if qty <= 0:
            stats['quantity_negative'] += 1
        if qlt <= 0:
            stats['quality_negative'] += 1
        if rel <= 0:
            stats['relation_negative'] += 1
        if man <= 0:
            stats['manner_negative'] += 1
        
        # Keep only if ALL margins are positive
        if qty > 0 and qlt > 0 and rel > 0 and man > 0:
            clean_pairs.append(item)
            stats['all_positive'] += 1
    
    print("\n" + "="*60)
    print("FILTERING RESULTS")
    print("="*60)
    print(f"Total pairs:              {stats['total']:,}")
    print(f"Quantity margin <= 0:     {stats['quantity_negative']:,} ({100*stats['quantity_negative']/stats['total']:.1f}%)")
    print(f"Quality margin <= 0:      {stats['quality_negative']:,} ({100*stats['quality_negative']/stats['total']:.1f}%)")
    print(f"Relation margin <= 0:     {stats['relation_negative']:,} ({100*stats['relation_negative']/stats['total']:.1f}%)")
    print(f"Manner margin <= 0:       {stats['manner_negative']:,} ({100*stats['manner_negative']/stats['total']:.1f}%)")
    print("-"*60)
    print(f"ALL POSITIVE (clean):     {stats['all_positive']:,} ({100*stats['all_positive']/stats['total']:.1f}%)")
    print("="*60)
    
    # Save as JSON
    print(f"\nSaving {len(clean_pairs)} clean pairs to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(clean_pairs, f, indent=2, ensure_ascii=False)
    
    # Save as CSV for easy viewing
    print(f"Saving CSV version to {output_csv}...")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'chosen', 'rejected', 'quantity_margin', 'quality_margin', 'relation_margin', 'manner_margin', 'avg_margin'])
        
        for item in clean_pairs:
            margins = item.get('margins', {})
            writer.writerow([
                item.get('prompt', ''),
                item.get('chosen', ''),
                item.get('rejected', ''),
                margins.get('quantity', 0),
                margins.get('quality', 0),
                margins.get('relation', 0),
                margins.get('manner', 0),
                item.get('avg_margin', 0)
            ])
    
    print(f"\n✅ Done! {len(clean_pairs)} clean pairs ready for DPO training.")
    return clean_pairs, stats

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    
    input_file = script_dir / "scored_data.json"
    output_json = script_dir / "clean_dpo_pairs.json"
    output_csv = script_dir / "clean_dpo_pairs.csv"
    
    clean_pairs, stats = filter_all_positive_margins(
        str(input_file),
        str(output_json),
        str(output_csv)
    )
