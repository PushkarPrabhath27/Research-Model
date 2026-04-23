"""
Track 2: Create Hybrid Training Dataset

Combines filtered weak labels with gold annotations,
oversampling gold examples to ensure high-quality signal.
"""

import json
import random
from pathlib import Path
from collections import Counter

class HybridDatasetCreator:
    """Create hybrid training set from weak and gold labels"""
    
    def __init__(self, gold_weight=5):
        self.gold_weight = gold_weight
        self.maxims = ['quantity', 'quality', 'relation', 'manner']
    
    def load_data(self, weak_file, gold_file):
        """Load weak and gold datasets"""
        weak_path = Path(weak_file)
        gold_path = Path(gold_file)
        
        if not weak_path.exists():
            raise FileNotFoundError(f"Weak labels not found: {weak_file}")
        if not gold_path.exists():
            raise FileNotFoundError(f"Gold labels not found: {gold_file}")
        
        with open(weak_path) as f:
            weak_data = json.load(f)
        
        with open(gold_path) as f:
            gold_data = json.load(f)
        
        return weak_data, gold_data
    
    def create_hybrid_dataset(self, weak_data, gold_data):
        """Combine weak and gold data with oversampling"""
        # Oversample gold examples
        hybrid_data = gold_data * self.gold_weight
        
        # Add weak labels
        hybrid_data.extend(weak_data)
        
        # Shuffle
        random.seed(42)  # For reproducibility
        random.shuffle(hybrid_data)
        
        return hybrid_data
    
    def analyze_dataset(self, data, name="Dataset"):
        """Analyze dataset composition"""
        print(f"\n{name} Analysis:")
        print("-"*60)
        
        # Count violations
        violation_counts = {m: 0 for m in self.maxims}
        total = len(data)
        
        for item in data:
            for maxim in self.maxims:
                key = f'{maxim}_violation'
                if key in item and item[key] == 1:
                    violation_counts[maxim] += 1
        
        print(f"Total examples: {total}")
        print(f"\nViolation distribution:")
        for maxim in self.maxims:
            count = violation_counts[maxim]
            pct = count / total * 100 if total > 0 else 0
            print(f"  {maxim.capitalize():12s}: {count:6d} ({pct:5.1f}%)")
    
    def run(self, weak_file, gold_file, output_file):
        """Run complete hybrid dataset creation"""
        print("\n" + "="*60)
        print("TRACK 2: HYBRID DATASET CREATION")
        print("="*60)
        print(f"\nGold weight: {self.gold_weight}x")
        print(f"(Each gold example repeated {self.gold_weight} times)")
        
        # Load data
        print(f"\nLoading data...")
        weak_data, gold_data = self.load_data(weak_file, gold_file)
        
        print(f"✓ Loaded {len(weak_data)} weak examples")
        print(f"✓ Loaded {len(gold_data)} gold examples")
        
        # Analyze original datasets
        self.analyze_dataset(weak_data, "Filtered Weak Labels")
        self.analyze_dataset(gold_data, "Gold Labels")
        
        # Create hybrid
        print(f"\nCreating hybrid dataset...")
        hybrid_data = self.create_hybrid_dataset(weak_data, gold_data)
        
        # Analyze hybrid
        self.analyze_dataset(hybrid_data, "Hybrid Dataset")
        
        print(f"\nComposition:")
        print(f"  Gold examples: {len(gold_data)} × {self.gold_weight} = {len(gold_data)*self.gold_weight}")
        print(f"  Weak examples: {len(weak_data)}")
        print(f"  Total:         {len(hybrid_data)}")
        print(f"  Gold ratio:    {len(gold_data)*self.gold_weight/len(hybrid_data)*100:.1f}%")
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(hybrid_data, f, indent=2)
        
        print(f"\n💾 Saved hybrid dataset to {output_file}")
        
        return hybrid_data

if __name__ == "__main__":
    creator = HybridDatasetCreator(gold_weight=5)
    
    weak_file = "data_processed/detector_data/detector_train_filtered.json"
    gold_file = "data_processed/gold_annotation_set.json"
    output_file = "data_processed/detector_data/detector_train_hybrid.json"
    
    hybrid_data = creator.run(weak_file, gold_file, output_file)
    
    print("\n" + "="*60)
    print("HYBRID DATASET COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review hybrid dataset composition")
    print("  2. Upload to Kaggle as new dataset")
    print("  3. Retrain detector with focal loss")
    print("="*60)
