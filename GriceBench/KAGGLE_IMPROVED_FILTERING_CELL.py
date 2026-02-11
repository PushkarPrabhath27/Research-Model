# ============================================
# IMPROVED DPO DATA FILTERING
# Copy-paste this into your Kaggle DPO scoring notebook
# ============================================

import json
import numpy as np
from pathlib import Path

print("="*60)
print("IMPROVED DPO DATA FILTERING (STRICTER)")
print("="*60)

# Load scored data
with open(CONFIG['dpo_train']) as f:
    scored_data = json.load(f)

with open(CONFIG['dpo_val']) as f:
    scored_val = json.load(f)

print(f"\nOriginal data:")
print(f"  Training: {len(scored_data)} pairs")
print(f"  Validation: {len(scored_val)} pairs")

# ============================================
# STRICT FILTERING: ALL MARGINS > 0.15
# ============================================

def strict_filter(data, min_margin=0.15):
    """
    Keep only pairs where ALL 4 maxims have positive margins > threshold
    This ensures clean, strong training signals
    """
    filtered = []
    stats = {
        'kept': 0,
        'removed': 0,
        'removed_reasons': {
            'quantity_negative': 0,
            'quality_negative': 0,
            'relation_negative': 0,
            'manner_negative': 0,
            'weak_signal': 0
        },
        'margin_stats': {m: [] for m in ['quantity', 'quality', 'relation', 'manner']}
    }
    
    for item in data:
        chosen_scores = item.get('chosen_scores', {})
        rejected_scores = item.get('rejected_scores', {})
        
        if not chosen_scores or not rejected_scores:
            stats['removed'] += 1
            continue
        
        # Calculate margins
        margins = {}
        for maxim in ['quantity', 'quality', 'relation', 'manner']:
            margin = rejected_scores[maxim] - chosen_scores[maxim]
            margins[maxim] = margin
        
        # Check if ALL margins are above threshold
        all_good = True
        removal_reason = None
        
        for maxim, margin in margins.items():
            if margin < min_margin:
                all_good = False
                if margin < 0:
                    stats['removed_reasons'][f'{maxim}_negative'] += 1
                    removal_reason = f'{maxim}_negative'
                else:
                    stats['removed_reasons']['weak_signal'] += 1
                    removal_reason = 'weak_signal'
                break
        
        if all_good:
            # Add margins to item
            item['margins'] = margins
            item['avg_margin'] = sum(margins.values()) / len(margins)
            filtered.append(item)
            stats['kept'] += 1
            
            # Track margins
            for maxim, margin in margins.items():
                stats['margin_stats'][maxim].append(margin)
        else:
            stats['removed'] += 1
    
    return filtered, stats

# Filter training data
print(f"\n{'='*60}")
print(f"FILTERING WITH MINIMUM MARGIN: 0.15")
print(f"{'='*60}")

filtered_train, train_stats = strict_filter(scored_data, min_margin=0.15)

print(f"\nTraining pairs:")
print(f"  Original: {len(scored_data)}")
print(f"  Filtered: {len(filtered_train)}")
print(f"  Kept:     {len(filtered_train)/len(scored_data)*100:.1f}%")
print(f"  Removed:  {train_stats['removed']}")

print(f"\nRemoval reasons:")
for reason, count in train_stats['removed_reasons'].items():
    if count > 0:
        print(f"  {reason:20s}: {count:4d} ({count/len(scored_data)*100:5.1f}%)")

# Filter validation data
filtered_val, val_stats = strict_filter(scored_val, min_margin=0.15)

print(f"\nValidation pairs:")
print(f"  Original: {len(scored_val)}")
print(f"  Filtered: {len(filtered_val)}")
print(f"  Kept:     {len(filtered_val)/len(scored_val)*100:.1f}%")

# ============================================
# VERIFY MARGIN QUALITY
# ============================================

print(f"\n{'='*60}")
print(f"FILTERED DATA QUALITY CHECK")
print(f"{'='*60}")

print(f"\nMargin Statistics (Training):\n")

for maxim in ['quantity', 'quality', 'relation', 'manner']:
    margins = np.array(train_stats['margin_stats'][maxim])
    
    print(f"{maxim.upper()}:")
    print(f"  Mean:     {margins.mean():.3f}")
    print(f"  Std:      {margins.std():.3f}")
    print(f"  Min:      {margins.min():.3f}")
    print(f"  Max:      {margins.max():.3f}")
    print(f"  All > 0:  {'✅' if margins.min() > 0 else '❌'}")
    print(f"  All >0.15: {'✅' if margins.min() > 0.15 else '❌'}")
    print()

avg_margins = np.array([item['avg_margin'] for item in filtered_train])
print("AVERAGE MARGIN:")
print(f"  Mean:     {avg_margins.mean():.3f}")
print(f"  Std:      {avg_margins.std():.3f}")
print(f"  Min:      {avg_margins.min():.3f}")

# ============================================
# DECISION POINT
# ============================================

print(f"\n{'='*60}")
print(f"FILTERING DECISION")
print(f"{'='*60}")

if len(filtered_train) < 500:
    print(f"\n⚠️  WARNING: Only {len(filtered_train)} training pairs!")
    print(f"   This might be too few for effective training.")
    print(f"\n   Options:")
    print(f"   1. Lower threshold to 0.10 (get ~1000-1500 pairs)")
    print(f"   2. Proceed with {len(filtered_train)} pairs (higher quality)")
    print(f"\n   Recommendation: Try threshold 0.10 if < 800 pairs")
    
    # Try with lower threshold
    print(f"\n   Testing with threshold 0.10...")
    filtered_train_alt, train_stats_alt = strict_filter(scored_data, min_margin=0.10)
    print(f"   Result: {len(filtered_train_alt)} pairs")
    
    if len(filtered_train_alt) > 800:
        print(f"\n   ✅ Using threshold 0.10 ({len(filtered_train_alt)} pairs)")
        filtered_train = filtered_train_alt
        filtered_val, _ = strict_filter(scored_val, min_margin=0.10)
        threshold_used = 0.10
    else:
        print(f"\n   ⚠️  Still too few. Proceeding with 0.15 threshold.")
        threshold_used = 0.15
else:
    print(f"\n✅ {len(filtered_train)} training pairs is sufficient!")
    print(f"   Quality: All margins > 0.15")
    print(f"   Proceeding with strict filtering.")
    threshold_used = 0.15

# ============================================
# SAVE FILTERED DATA
# ============================================

output_dir = Path(CONFIG['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'dpo_train_filtered_v2.json', 'w') as f:
    json.dump(filtered_train, f, indent=2)

with open(output_dir / 'dpo_val_filtered_v2.json', 'w') as f:
    json.dump(filtered_val, f, indent=2)

print(f"\n{'='*60}")
print(f"✅ IMPROVED FILTERING COMPLETE!")
print(f"{'='*60}")

print(f"\nSaved files:")
print(f"  - dpo_train_filtered_v2.json ({len(filtered_train)} pairs)")
print(f"  - dpo_val_filtered_v2.json ({len(filtered_val)} pairs)")
print(f"  - Threshold used: {threshold_used}")

print(f"\n📥 Download from {output_dir}/")
print(f"\nNext steps:")
print(f"  1. Download these files")
print(f"  2. Upload as new dataset: gricebench-dpo-filtered-v2")
print(f"  3. Re-run DPO training with this cleaner data")
print(f"  4. Expected: 60-70% cooperative rate!")
print(f"{'='*60}")
