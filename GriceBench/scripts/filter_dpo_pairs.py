"""
Track 3: DPO Preference Pair Filtering

Filters DPO training pairs to keep only those with clear preference margins,
improving training signal quality.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

class DPOPairFilter:
    """Filter DPO preference pairs by margin quality"""
    
    def __init__(self, min_margin=0.15):
        self.min_margin = min_margin
        self.maxims = ['quantity', 'quality', 'relation', 'manner']
    
    def calculate_margin(self, chosen_scores, rejected_scores):
        """Calculate preference margin (rejected - chosen)"""
        margins = {}
        for maxim in self.maxims:
            chosen = chosen_scores.get(maxim, 0.5)
            rejected = rejected_scores.get(maxim, 0.5)
            margins[maxim] = rejected - chosen  # Positive = chosen is better
        
        return margins
    
    def filter_pairs(self, data):
        """Filter pairs by margin quality"""
        # NOTE: Current DPO data doesn't have detector scores yet
        # This will be populated after detector retraining
        # For now, just copy all data
        
        print("\n⚠️  NOTE: DPO data doesn't have detector scores yet")
        print("   Filtering will be applied after detector V2 generates scores")
        print("   For now, copying all data to output file")
        
        filtered = data  # Keep all for now
        stats = {
            'total': len(data),
            'kept': len(data),
            'removed': 0,
            'margin_stats': {m: [] for m in self.maxims}
        }
        
        return filtered, stats
    
    def analyze_margins(self, stats):
        """Analyze margin statistics"""
        print("\n" + "="*60)
        print("MARGIN ANALYSIS")
        print("="*60)
        
        for maxim in self.maxims:
            margins = stats['margin_stats'][maxim]
            if not margins:
                continue
            
            m = np.array(margins)
            
            print(f"\n{maxim.upper()}:")
            print(f"  Mean margin:     {m.mean():.3f}")
            print(f"  Std:             {m.std():.3f}")
            print(f"  Min:             {m.min():.3f}")
            print(f"  Max:             {m.max():.3f}")
            
            # Quality indicators
            clear = (m > 0.2).mean()
            very_clear = (m > 0.3).mean()
            
            print(f"  % Clear (>0.2):  {clear*100:.1f}%")
            print(f"  % Very clear (>0.3): {very_clear*100:.1f}%")
    
    def run(self, input_file, output_file):
        """Run complete filtering pipeline"""
        print("\n" + "="*60)
        print("TRACK 3: DPO PREFERENCE PAIR FILTERING")
        print("="*60)
        print(f"\nMinimum margin: {self.min_margin}")
        print(f"(Keep only pairs where rejected - chosen > {self.min_margin})")
        
        # Load data
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"\nLoading data from {input_file}...")
        with open(input_path) as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} preference pairs")
        
        # Filter
        print("\nFiltering...")
        filtered_data, stats = self.filter_pairs(data)
        
        # Report statistics
        print("\n" + "="*60)
        print("FILTERING RESULTS")
        print("="*60)
        print(f"\nTotal pairs:       {stats['total']}")
        print(f"Kept:              {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
        print(f"Removed:           {stats['removed']} ({stats['removed']/stats['total']*100:.1f}%)")
        
        # Analyze margins
        self.analyze_margins(stats)
        
        # Save filtered data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"\n💾 Saved {len(filtered_data)} filtered pairs to {output_file}")
        
        return filtered_data

if __name__ == "__main__":
    filter = DPOPairFilter(min_margin=0.15)
    
    input_file = "data_processed/dpo_data/dpo_train.json"
    output_file = "data_processed/dpo_data/dpo_train_filtered.json"
    
    filtered_data = filter.run(input_file, output_file)
    
    print("\n" + "="*60)
    print("FILTERING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review filtered preference pairs")
    print("  2. Upload to Kaggle as new dataset")
    print("  3. Retrain DPO with multi-objective loss")
    print("="*60)
