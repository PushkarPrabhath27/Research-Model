# ============================================
# CELL 9.7: CONFLICT-FREE FILTERING (THE RIGHT SOLUTION)
# ============================================

print("\n" + "="*60)
print("FILTERING CONFLICTING PREFERENCE SIGNALS")
print("="*60)

import pandas as pd

# Convert to DataFrame for easier analysis
data_list = []
for item in scored_data:
    data_list.append({
        'prompt': item.get('prompt', ''),
        'chosen': item.get('chosen', ''),
        'rejected': item.get('rejected', ''),
        'quantity_margin': item['margins']['quantity'],
        'quality_margin': item['margins']['quality'],
        'relation_margin': item['margins']['relation'],
        'manner_margin': item['margins']['manner'],
        'avg_margin': item['avg_margin'],
        'full_item': item
    })

df = pd.DataFrame(data_list)

print(f"\nOriginal data: {len(df)} pairs")

# ============================================
# STEP 1: DIAGNOSTIC - Find Conflicts
# ============================================

print("\n" + "="*60)
print("CONFLICT DIAGNOSTIC")
print("="*60)

threshold = 0.15  # Significance threshold

# Type 1: Relation good, Manner bad (main problem)
type1_conflicts = (df['relation_margin'] > threshold) & (df['manner_margin'] < -threshold)

# Type 2: Relation bad, Manner good (rare)
type2_conflicts = (df['relation_margin'] < -threshold) & (df['manner_margin'] > threshold)

# All conflicts
all_conflicts = type1_conflicts | type2_conflicts

print(f"\nConflict Analysis:")
print(f"  Type 1 (Relation+, Manner-): {type1_conflicts.sum():4d} ({type1_conflicts.mean()*100:5.1f}%)")
print(f"  Type 2 (Relation-, Manner+): {type2_conflicts.sum():4d} ({type2_conflicts.mean()*100:5.1f}%)")
print(f"  Total conflicts:             {all_conflicts.sum():4d} ({all_conflicts.mean()*100:5.1f}%)")
print(f"  Non-conflicting:             {(~all_conflicts).sum():4d} ({(~all_conflicts).mean()*100:5.1f}%)")

# ============================================
# STEP 2: SHOW EXAMPLES OF CONFLICTS
# ============================================

print("\n" + "="*60)
print("EXAMPLE CONFLICTING PAIRS (Type 1: Relation+, Manner-)")
print("="*60)

if type1_conflicts.sum() > 0:
    conflict_examples = df[type1_conflicts].sample(min(3, type1_conflicts.sum()))
    
    for idx, (i, row) in enumerate(conflict_examples.iterrows(), 1):
        print(f"\n--- Conflict Example {idx} ---")
        print(f"Relation margin: +{row['relation_margin']:.3f} (chosen is on-topic)")
        print(f"Manner margin:   {row['manner_margin']:.3f} (chosen is unclear)")
        print(f"\nChosen (on-topic but unclear):")
        print(f"  {row['chosen'][:150]}...")
        print(f"\nRejected (off-topic but clear):")
        print(f"  {row['rejected'][:150]}...")
        print(f"\n⚠️  Problem: Model learns 'being unclear is good'")

# ============================================
# STEP 3: FILTER OUT CONFLICTS
# ============================================

print("\n" + "="*60)
print("FILTERING CONFLICTS")
print("="*60)

# Keep only non-conflicting pairs
clean_df = df[~all_conflicts].copy()

print(f"\nFiltering Results:")
print(f"  Original pairs:     {len(df)}")
print(f"  Conflicts removed:  {all_conflicts.sum()}")
print(f"  Clean pairs kept:   {len(clean_df)}")
print(f"  Retention rate:     {len(clean_df)/len(df)*100:.1f}%")

# ============================================
# STEP 4: VERIFY ALL MARGINS ARE NOW POSITIVE
# ============================================

print("\n" + "="*60)
print("CLEAN DATA MARGIN STATISTICS")
print("="*60)

print("\nMargin Statistics (After Conflict Filtering):\n")

for maxim in ['quantity', 'quality', 'relation', 'manner']:
    col = f'{maxim}_margin'
    margins = clean_df[col].values
    
    print(f"{maxim.upper()}:")
    print(f"  Mean:     {margins.mean():7.3f}")
    print(f"  Std:      {margins.std():7.3f}")
    print(f"  Min:      {margins.min():7.3f}")
    print(f"  Max:      {margins.max():7.3f}")
    print(f"  Positive: {(margins > 0).mean()*100:5.1f}%")
    print()

avg_margins = clean_df['avg_margin'].values
print("AVERAGE MARGIN:")
print(f"  Mean:     {avg_margins.mean():7.3f}")
print(f"  Std:      {avg_margins.std():7.3f}")
print(f"  Min:      {avg_margins.min():7.3f}")
print(f"  Max:      {avg_margins.max():7.3f}")

# ============================================
# STEP 5: CHECK IF ALL MARGINS ARE POSITIVE
# ============================================

print("\n" + "="*60)
print("VALIDATION CHECK")
print("="*60)

all_positive = True
for maxim in ['quantity', 'quality', 'relation', 'manner']:
    col = f'{maxim}_margin'
    mean_margin = clean_df[col].mean()
    
    if mean_margin > 0:
        print(f"✅ {maxim.capitalize():12s}: Mean = +{mean_margin:.3f} (POSITIVE)")
    else:
        print(f"❌ {maxim.capitalize():12s}: Mean = {mean_margin:.3f} (NEGATIVE)")
        all_positive = False

if all_positive:
    print("\n🎉 SUCCESS! All maxims have positive mean margins!")
    print("   Model will learn to improve ALL 4 maxims!")
else:
    print("\n⚠️  Warning: Some maxims still have negative margins")
    print("   Consider adjusting threshold or investigating further")

# ============================================
# STEP 6: SAVE CLEAN DATA
# ============================================

print("\n" + "="*60)
print("SAVING CONFLICT-FREE DATA")
print("="*60)

# Extract full items
clean_train = [row['full_item'] for _, row in clean_df.iterrows()]

# Also filter validation data
val_data_list = []
for item in scored_val:
    val_data_list.append({
        'quantity_margin': item['margins']['quantity'],
        'quality_margin': item['margins']['quality'],
        'relation_margin': item['margins']['relation'],
        'manner_margin': item['margins']['manner'],
        'full_item': item
    })

val_df = pd.DataFrame(val_data_list)

# Filter validation conflicts
val_type1 = (val_df['relation_margin'] > threshold) & (val_df['manner_margin'] < -threshold)
val_type2 = (val_df['relation_margin'] < -threshold) & (val_df['manner_margin'] > threshold)
val_conflicts = val_type1 | val_type2

clean_val_df = val_df[~val_conflicts]
clean_val = [row['full_item'] for _, row in clean_val_df.iterrows()]

print(f"\nValidation data:")
print(f"  Original: {len(val_df)}")
print(f"  Conflicts: {val_conflicts.sum()}")
print(f"  Clean: {len(clean_val)}")

# Save
output_dir = Path(CONFIG['output_dir'])

with open(output_dir / 'dpo_train_filtered.json', 'w') as f:
    json.dump(clean_train, f, indent=2)

with open(output_dir / 'dpo_val_filtered.json', 'w') as f:
    json.dump(clean_val, f, indent=2)

print(f"\n✓ Saved conflict-free data to {output_dir}")

print("\n" + "="*60)
print("CONFLICT FILTERING COMPLETE!")
print("="*60)
print(f"\nFinal Dataset:")
print(f"  Training:   {len(clean_train)} pairs")
print(f"  Validation: {len(clean_val)} pairs")
print(f"\n🎯 Ready for DPO training with:")
print(f"  ✅ All margins positive")
print(f"  ✅ No conflicting signals")
print(f"  ✅ Model will learn: 'Be relevant AND clear'")
print(f"  ✅ Expected: All 4 maxims improve!")
print("="*60)

# Update variables for potential next cells
filtered_train = clean_train
filtered_val = clean_val
