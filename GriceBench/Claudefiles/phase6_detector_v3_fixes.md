# Phase 6 Detector V3 - Critical Fixes

## Fix #1: Add pos_weight to Loss Function

### Location: Cell 7 (Training Loop)

**REPLACE THIS:**
```python
criterion = nn.BCEWithLogitsLoss()
```

**WITH THIS:**
```python
# Calculate pos_weight from training data
train_labels = np.array([ex.labels for ex in train_data])
pos_counts = train_labels.sum(axis=0)
neg_counts = len(train_labels) - pos_counts
pos_weight_values = neg_counts / (pos_counts + 1e-6)

pos_weight_tensor = torch.tensor(pos_weight_values, dtype=torch.float32).to(device)

print(f"\n📊 Class Imbalance Correction:")
for i, name in enumerate(MAXIM_NAMES):
    print(f"  {name}: pos_weight = {pos_weight_values[i]:.2f} " 
          f"(pos={int(pos_counts[i])}, neg={int(neg_counts[i])})")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
```

**Why:** Without this, model will collapse to predicting all zeros for minority classes.

---

## Fix #2: Add Real-Time Training Health Checks

### Location: Cell 7 (Inside training loop, after validation)

**ADD THIS after `val_f1_opt, val_per_class_opt, _, _, _ = evaluate(...)`:**

```python
# ---- Real-time health checks ----
if epoch >= 2:  # Start checking from epoch 2
    # Check 1: Model collapse (all predictions near 0.5)
    pred_variance = val_probs.var()
    if pred_variance < 0.01:
        logger.warning(f"⚠️ ALERT: Prediction variance = {pred_variance:.4f} (very low!)")
        logger.warning(f"   Model may be collapsing. Consider:")
        logger.warning(f"   - Checking pos_weight is applied")
        logger.warning(f"   - Reducing learning rate")
        logger.warning(f"   - Stopping training")
    
    # Check 2: Overfitting (val loss increasing)
    if len(training_history) >= 2:
        prev_val_loss = training_history[-1]['val_loss']
        if val_loss > prev_val_loss + 0.05:  # Significant increase
            logger.warning(f"⚠️ ALERT: Val loss increased {prev_val_loss:.3f} → {val_loss:.3f}")
            logger.warning(f"   Possible overfitting")
    
    # Check 3: Suspiciously high F1 early
    if epoch <= 3 and val_f1_opt > 0.90:
        logger.warning(f"⚠️ ALERT: F1 = {val_f1_opt:.3f} at epoch {epoch}")
        logger.warning(f"   Suspiciously high for early training")
        logger.warning(f"   May indicate data leakage or synthetic patterns")
```

**Why:** Catch problems during training, not after 50 minutes of wasted GPU time.

---

## Fix #3: Add Separate Held-Out Test Set

### Location: NEW Cell 4b (Insert after current Cell 4)

**ADD THIS ENTIRE NEW CELL:**

```python
# ============================================================================
# CELL 4b: CREATE TRUE HELD-OUT TEST SET (NEVER IN TRAINING)
# ============================================================================
logger.info("=" * 60)
logger.info("CREATING HELD-OUT TEST SET")
logger.info("=" * 60)

# Reserve 20% of Phase 4 data as held-out test (never used in training)
# Use a different random seed to ensure independence
random.seed(999)  # Different from training seed (42)

# Stratify by generation method to ensure diverse test set
holdout_violations = []
holdout_clean = []

# Group by generation method
method_groups = defaultdict(list)
for ex in violations:
    method_groups[ex.generation_method].append(ex)

# Take 20% from each method
for method, examples in method_groups.items():
    n_holdout = max(1, int(len(examples) * 0.20))
    random.shuffle(examples)
    holdout_violations.extend(examples[:n_holdout])

# Take 20% of clean examples
random.shuffle(clean_examples)
n_clean_holdout = int(len(clean_examples) * 0.20)
holdout_clean = clean_examples[:n_clean_holdout]

# Remove holdout from main pools
violations_for_training = [ex for ex in violations if ex not in holdout_violations]
clean_for_training = [ex for ex in clean_examples if ex not in holdout_clean]

print(f"\n📊 Held-Out Test Set:")
print(f"  Violations: {len(holdout_violations)}")
print(f"  Clean: {len(holdout_clean)}")
print(f"  Total: {len(holdout_violations) + len(holdout_clean)}")

print(f"\n📊 Available for Training/Val:")
print(f"  Violations: {len(violations_for_training)}")
print(f"  Clean: {len(clean_for_training)}")

# Update variables for rest of notebook
violations = violations_for_training
clean_examples = clean_for_training
holdout_test_data = holdout_violations + holdout_clean

# Reset seed for consistent training
random.seed(SEED)

logger.info(f"✅ Held-out test set created: {len(holdout_test_data)} examples")
tracker.mark('Held-Out Test Creation', 'PASS', {
    'holdout_size': len(holdout_test_data),
    'training_pool_size': len(violations) + len(clean_examples),
})
```

### Then MODIFY Cell 5 (Stratified Split):

**CHANGE THIS LINE:**
```python
all_data = violations + clean_examples
```

**TO THIS:**
```python
# Use only training pool (holdout already removed)
all_data = violations + clean_examples
logger.info(f"Training pool (excluding held-out): {len(all_data)}")
```

### Then ADD NEW Cell after current Cell 8:

**ADD CELL 8b: Final Held-Out Test Evaluation**

```python
# ============================================================================
# CELL 8b: FINAL HELD-OUT TEST (TRULY UNSEEN DATA)
# ============================================================================
logger.info("=" * 60)
logger.info("🎯 FINAL HELD-OUT TEST (COMPLETELY UNSEEN)")
logger.info("=" * 60)

# Create dataset for held-out test
holdout_dataset = GriceDataset(holdout_test_data, tokenizer, CONFIG.max_length)
holdout_loader = DataLoader(holdout_dataset, batch_size=CONFIG.batch_size * 2, 
                            shuffle=False, num_workers=2, pin_memory=True)

# Evaluate on held-out set
holdout_f1, holdout_per_class, holdout_loss, holdout_probs, holdout_labels = evaluate(
    model, holdout_loader, device, best_thresholds
)

print(f"\n{'='*60}")
print(f"🏆 HELD-OUT TEST RESULTS (COMPLETELY UNSEEN)")
print(f"{'='*60}")
print(f"\nMacro F1: {holdout_f1:.4f}")
print(f"Loss: {holdout_loss:.4f}")
print(f"\nPer-Maxim Performance:")
for name in MAXIM_NAMES:
    sc = holdout_per_class[name]
    print(f"  {name}: F1={sc['f1']:.3f}, P={sc['precision']:.3f}, R={sc['recall']:.3f}")

# Compare to in-distribution test set
print(f"\n📊 Generalization Check:")
print(f"  In-distribution test F1: {test_f1:.4f}")
print(f"  Held-out test F1:        {holdout_f1:.4f}")
print(f"  Generalization gap:      {abs(test_f1 - holdout_f1):.4f}")

if abs(test_f1 - holdout_f1) > 0.10:
    print(f"\n⚠️ WARNING: Large generalization gap (>0.10)")
    print(f"   Model may be overfitting to Phase 4 patterns")
elif abs(test_f1 - holdout_f1) > 0.05:
    print(f"\n⚠️ NOTICE: Moderate generalization gap (>0.05)")
    print(f"   Some overfitting to training distribution")
else:
    print(f"\n✅ Good generalization (gap < 0.05)")

# Health check on held-out scores
if holdout_f1 > 0.95:
    print(f"\n⚠️ CRITICAL: Held-out F1 = {holdout_f1:.3f} is suspiciously high!")
    print(f"   This strongly suggests:")
    print(f"   1. Data leakage (check code carefully)")
    print(f"   2. Phase 4 patterns too easy/synthetic")
    print(f"   3. Model memorizing rather than learning")
elif holdout_f1 > 0.80:
    print(f"\n✅ EXCELLENT: Held-out F1 = {holdout_f1:.3f}")
elif holdout_f1 > 0.65:
    print(f"\n✅ GOOD: Held-out F1 = {holdout_f1:.3f}")
else:
    print(f"\n⚠️ LOW: Held-out F1 = {holdout_f1:.3f}")
    print(f"   Model may need more data or better hyperparameters")

tracker.mark('Held-Out Test', 'PASS', {
    'holdout_f1': holdout_f1,
    'holdout_loss': holdout_loss,
    'generalization_gap': abs(test_f1 - holdout_f1),
})
```

**Why:** This is the ONLY way to know if your model truly generalizes.

---

## Fix #4: Add Model Prediction Distribution Analysis

### Location: NEW Cell 9b (Insert after error analysis)

**ADD THIS CELL:**

```python
# ============================================================================
# CELL 9b: PREDICTION DISTRIBUTION ANALYSIS
# ============================================================================
logger.info("=" * 60)
logger.info("PREDICTION DISTRIBUTION ANALYSIS")
logger.info("=" * 60)

# Analyze prediction distributions
print(f"\n📊 Prediction Probability Distributions:")

for i, name in enumerate(MAXIM_NAMES):
    probs_for_class = test_probs[:, i]
    
    print(f"\n  {name}:")
    print(f"    Mean: {probs_for_class.mean():.3f}")
    print(f"    Std:  {probs_for_class.std():.3f}")
    print(f"    Min:  {probs_for_class.min():.3f}")
    print(f"    Max:  {probs_for_class.max():.3f}")
    
    # Check for collapse signatures
    if probs_for_class.std() < 0.10:
        print(f"    ⚠️ WARNING: Very low variance - possible collapse!")
    
    # Check distribution
    bins = np.histogram(probs_for_class, bins=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])[0]
    print(f"    Distribution:")
    print(f"      [0.0-0.1): {bins[0]} ({100*bins[0]/len(probs_for_class):.1f}%)")
    print(f"      [0.1-0.3): {bins[1]} ({100*bins[1]/len(probs_for_class):.1f}%)")
    print(f"      [0.3-0.5): {bins[2]} ({100*bins[2]/len(probs_for_class):.1f}%)")
    print(f"      [0.5-0.7): {bins[3]} ({100*bins[3]/len(probs_for_class):.1f}%)")
    print(f"      [0.7-0.9): {bins[4]} ({100*bins[4]/len(probs_for_class):.1f}%)")
    print(f"      [0.9-1.0]: {bins[5]} ({100*bins[5]/len(probs_for_class):.1f}%)")
    
    # Health check
    if bins[2] + bins[3] > 0.80 * len(probs_for_class):  # 80% in middle bins
        print(f"    ⚠️ WARNING: {name} predictions clustered around 0.5!")
        print(f"       Model not learning to separate this class")

print(f"\n✅ Prediction distribution analysis complete")
```

**Why:** Helps diagnose model collapse and poor calibration.

---

## Summary of Changes

### Files to Modify:
1. **Cell 7**: Add pos_weight to criterion + real-time health checks
2. **NEW Cell 4b**: Create held-out test set
3. **Cell 5**: Update to use training pool only
4. **NEW Cell 8b**: Evaluate on held-out test
5. **NEW Cell 9b**: Add prediction distribution analysis
6. **Cell 10**: Add held-out results to JSON output

### Expected Impact:
- **Fix #1** (pos_weight): Prevents model collapse (F1 0.0 → 0.70+)
- **Fix #2** (health checks): Catches problems during training
- **Fix #3** (held-out test): True generalization measure
- **Fix #4** (distribution analysis): Diagnoses calibration issues

### Time Investment:
- Making changes: 15 minutes
- Re-running training: 30 minutes
- **Total: 45 minutes**

### Expected Results After Fixes:
- If data loads correctly: **F1 = 0.70-0.85** (realistic)
- If data is missing/synthetic: **Crashes with clear error message**
- If overfitting: **Caught by real-time alerts**
- If generalizes well: **Held-out F1 within 0.05 of test F1**
