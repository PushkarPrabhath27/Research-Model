"""
Track 2: Data Quality Fixes - Weak Label Filtering

This script filters weak labels to keep only high-confidence examples,
reducing noise and improving detector training quality.

Based on the diagnostic finding that weak labels cluster around 0.5,
we keep only examples where the labeler was confident (>0.75 or <0.25).
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

class WeakLabelFilter:
    """Filter weak labels to keep only confident examples"""
    
    def __init__(self, confidence_threshold=0.75):
        self.confidence_threshold = confidence_threshold
        self.maxims = ['quantity', 'quality', 'relation', 'manner']
        
    def is_confident(self, score):
        """Check if a score is confident (far from 0.5)"""
        return score > self.confidence_threshold or score < (1 - self.confidence_threshold)
    
    def filter_dataset(self, data):
        """Filter dataset to keep only confident examples"""
        filtered = []
        stats = {
            'total': len(data),
            'kept': 0,
            'removed': 0,
            'removed_reasons': Counter()
        }
        
        for item in data:
            # Check if all maxims have confident labels
            confident = True
            uncertain_maxims = []
            
            for maxim in self.maxims:
                key = f'{maxim}_violation'
                if key in item:
                    score = item[key]
                    if not self.is_confident(score):
                        confident = False
                        uncertain_maxims.append(maxim)
            
            if confident:
                # Binarize labels (convert probabilities to 0/1)
                binary_item = item.copy()
                for maxim in self.maxims:
                    key = f'{maxim}_violation'
                    if key in item:
                        score = item[key]
                        binary_item[key] = 1 if score > 0.5 else 0
                
                filtered.append(binary_item)
                stats['kept'] += 1
            else:
                stats['removed'] += 1
                stats['removed_reasons'][', '.join(uncertain_maxims)] += 1
        
        return filtered, stats
    
    def analyze_filtered_data(self, filtered_data):
        """Analyze the filtered dataset"""
        print("\n" + "="*60)
        print("FILTERED DATA ANALYSIS")
        print("="*60)
        
        # Count violations per maxim
        violation_counts = {m: 0 for m in self.maxims}
        total = len(filtered_data)
        
        for item in filtered_data:
            for maxim in self.maxims:
                key = f'{maxim}_violation'
                if key in item and item[key] == 1:
                    violation_counts[maxim] += 1
        
        print(f"\nClass Distribution (n={total}):")
        print("-"*60)
        for maxim in self.maxims:
            count = violation_counts[maxim]
            pct = count / total * 100 if total > 0 else 0
            print(f"{maxim.capitalize():12s}: {count:6d} violations ({pct:5.1f}%)")
        
        # Calculate class weights for balanced training
        class_weights = {}
        for maxim in self.maxims:
            violations = violation_counts[maxim]
            non_violations = total - violations
            if violations > 0:
                # Weight = (total / 2) / violations
                weight = (total / 2) / violations
            else:
                weight = 1.0
            class_weights[maxim] = weight
        
        print(f"\nRecommended Class Weights:")
        print("-"*60)
        for maxim in self.maxims:
            print(f"{maxim.capitalize():12s}: {class_weights[maxim]:.3f}")
        
        return class_weights
    
    def run(self, input_file, output_file):
        """Run the complete filtering pipeline"""
        print("\n" + "="*60)
        print("TRACK 2: WEAK LABEL FILTERING")
        print("="*60)
        print(f"\nConfidence threshold: {self.confidence_threshold}")
        print(f"(Keep only scores > {self.confidence_threshold} or < {1-self.confidence_threshold})")
        
        # Load data
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"\nLoading data from {input_file}...")
        with open(input_path) as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} examples")
        
        # Filter
        print("\nFiltering...")
        filtered_data, stats = self.filter_dataset(data)
        
        # Report statistics
        print("\n" + "="*60)
        print("FILTERING RESULTS")
        print("="*60)
        print(f"\nTotal examples:    {stats['total']}")
        print(f"Kept:              {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
        print(f"Removed:           {stats['removed']} ({stats['removed']/stats['total']*100:.1f}%)")
        
        if stats['removed'] > 0:
            print(f"\nRemoval reasons (uncertain maxims):")
            for reason, count in stats['removed_reasons'].most_common(10):
                print(f"  {reason}: {count}")
        
        # Analyze filtered data
        class_weights = self.analyze_filtered_data(filtered_data)
        
        # Save filtered data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        
        print(f"\n💾 Saved {len(filtered_data)} filtered examples to {output_file}")
        
        # Save class weights
        weights_file = output_path.parent / "class_weights_filtered.json"
        with open(weights_file, 'w') as f:
            json.dump(class_weights, f, indent=2)
        
        print(f"💾 Saved class weights to {weights_file}")
        
        return filtered_data, class_weights

if __name__ == "__main__":
    # Filter weak labels
    filter = WeakLabelFilter(confidence_threshold=0.75)
    
    # Try both possible file names
    input_files = [
        "data_processed/gricebench_weak_labeled.json",
        "data_processed/gricebench_weak_50k.json"
    ]
    
    input_file = None
    for f in input_files:
        if Path(f).exists():
            input_file = f
            break
    
    if input_file is None:
        print("❌ No weak label file found!")
        print("   Tried:", input_files)
        exit(1)
    
    output_file = "data_processed/detector_data/detector_train_filtered.json"
    
    filtered_data, class_weights = filter.run(input_file, output_file)
    
    print("\n" + "="*60)
    print("FILTERING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review filtered data quality")
    print("  2. Create hybrid dataset (filtered + gold)")
    print("  3. Retrain detector with focal loss")
    print("="*60)
