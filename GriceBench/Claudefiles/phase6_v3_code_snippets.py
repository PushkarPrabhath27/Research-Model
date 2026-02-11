# ============================================================================
# READY-TO-PASTE CODE SNIPPETS FOR PHASE 6 DETECTOR V3
# ============================================================================

# ============================================================================
# FIX #1: UPDATE CELL 7 - ADD POS_WEIGHT TO LOSS
# ============================================================================
# FIND THIS LINE in Cell 7:
#   criterion = nn.BCEWithLogitsLoss()
#
# REPLACE WITH:

# Calculate pos_weight from training data (addresses class imbalance)
train_labels_array = np.array([ex.labels for ex in train_data])
pos_counts = train_labels_array.sum(axis=0)
neg_counts = len(train_labels_array) - pos_counts
pos_weight_values = neg_counts / (pos_counts + 1e-6)

pos_weight_tensor = torch.tensor(pos_weight_values, dtype=torch.float32).to(device)

logger.info(f"\n📊 Class Imbalance Correction (pos_weight):")
for i, name in enumerate(MAXIM_NAMES):
    logger.info(f"  {name}: weight={pos_weight_values[i]:.2f} "
                f"(pos={int(pos_counts[i])}, neg={int(neg_counts[i])})")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
logger.info(f"✅ BCE Loss with pos_weight initialized")


# ============================================================================
# FIX #2: UPDATE CELL 7 - ADD HEALTH CHECKS IN TRAINING LOOP
# ============================================================================
# FIND THIS SECTION in Cell 7 (after val_f1_opt calculation):
#   epoch_result = {
#       'epoch': epoch,
#       ...
#   }
#
# INSERT THIS BEFORE epoch_result = {...}:

    # ============================================================
    # REAL-TIME HEALTH CHECKS
    # ============================================================
    if epoch >= 2:  # Start monitoring from epoch 2
        # Check 1: Model collapse (predictions clustered)
        pred_variance = val_probs.var()
        pred_mean = val_probs.mean()
        
        if pred_variance < 0.01:
            logger.warning(f"⚠️ COLLAPSE ALERT: Pred variance={pred_variance:.4f}")
            logger.warning(f"   Mean={pred_mean:.3f}, Model may be collapsing!")
        
        # Check 2: Predictions stuck near threshold
        near_threshold = np.sum((val_probs > 0.4) & (val_probs < 0.6)) / val_probs.size
        if near_threshold > 0.7:  # 70% of predictions near 0.5
            logger.warning(f"⚠️ THRESHOLD ALERT: {near_threshold:.1%} predictions near 0.5")
            logger.warning(f"   Model not separating classes well")
        
        # Check 3: Validation loss increasing (overfitting)
        if len(training_history) >= 2:
            prev_val_loss = training_history[-1]['val_loss']
            if val_loss > prev_val_loss + 0.05:
                logger.warning(f"⚠️ OVERFITTING: Val loss {prev_val_loss:.3f} → {val_loss:.3f}")
        
        # Check 4: Suspiciously high F1 too early
        if epoch <= 3 and val_f1_opt > 0.90:
            logger.warning(f"⚠️ SUSPICIOUS: F1={val_f1_opt:.3f} at epoch {epoch}")
            logger.warning(f"   May indicate data leakage or synthetic patterns")
        
        # Check 5: Per-class variance (all classes should have different distributions)
        class_variances = val_probs.var(axis=0)
        if np.any(class_variances < 0.01):
            collapsed_classes = [MAXIM_NAMES[i] for i in range(4) if class_variances[i] < 0.01]
            logger.warning(f"⚠️ CLASS COLLAPSE: {', '.join(collapsed_classes)} have low variance")


# ============================================================================
# FIX #3A: NEW CELL 4b - CREATE HELD-OUT TEST SET
# ============================================================================
# INSERT THIS AS A NEW CELL AFTER CURRENT CELL 4:

# ============================================================================
# CELL 4b: CREATE TRUE HELD-OUT TEST SET (NEVER IN TRAINING)
# ============================================================================
logger.info("=" * 60)
logger.info("🎯 CREATING HELD-OUT TEST SET (COMPLETELY UNSEEN)")
logger.info("=" * 60)

# Use different seed for independence
random.seed(999)  # Different from main seed (42)

# Stratify by generation method to ensure diverse test set
holdout_violations = []
method_groups = defaultdict(list)

for ex in violations:
    method_groups[ex.generation_method].append(ex)

# Take 20% from each generation method
for method, examples in method_groups.items():
    n_holdout = max(1, int(len(examples) * 0.20))
    random.shuffle(examples)
    holdout_violations.extend(examples[:n_holdout])
    logger.info(f"  {method}: {n_holdout} examples for holdout")

# Take 20% of clean examples
random.shuffle(clean_examples)
n_clean_holdout = int(len(clean_examples) * 0.20)
holdout_clean = clean_examples[:n_clean_holdout]

# Create holdout sets (violations + clean)
holdout_test_data = holdout_violations + holdout_clean

# Remove from training pool
violations_for_training = [ex for ex in violations if ex not in holdout_violations]
clean_for_training = [ex for ex in clean_examples if ex not in holdout_clean]

print(f"\n📊 Data Split:")
print(f"  Held-out test: {len(holdout_test_data)} ({len(holdout_violations)} viol + {len(holdout_clean)} clean)")
print(f"  Training pool: {len(violations_for_training) + len(clean_for_training)}")

# Verify no overlap
assert len(set(ex.text for ex in holdout_violations) & 
          set(ex.text for ex in violations_for_training)) == 0, \
    "❌ CRITICAL: Overlap between holdout and training violations!"

assert len(set(ex.text for ex in holdout_clean) & 
          set(ex.text for ex in clean_for_training)) == 0, \
    "❌ CRITICAL: Overlap between holdout and training clean!"

print(f"✅ No overlap between holdout and training pool")

# Update main variables for rest of notebook
violations = violations_for_training
clean_examples = clean_for_training

# Reset seed for consistent training
random.seed(SEED)

logger.info(f"✅ Held-out test set: {len(holdout_test_data)} examples")
tracker.mark('Held-Out Creation', 'PASS', {
    'holdout_size': len(holdout_test_data),
    'training_pool': len(violations) + len(clean_examples),
})


# ============================================================================
# FIX #3B: NEW CELL 8b - EVALUATE ON HELD-OUT TEST
# ============================================================================
# INSERT THIS AS A NEW CELL AFTER CURRENT CELL 8 (test evaluation):

# ============================================================================
# CELL 8b: HELD-OUT TEST EVALUATION (COMPLETELY UNSEEN)
# ============================================================================
logger.info("=" * 60)
logger.info("🏆 HELD-OUT TEST (COMPLETELY UNSEEN - ULTIMATE TEST)")
logger.info("=" * 60)

# Create dataset for held-out test
holdout_dataset = GriceDataset(holdout_test_data, tokenizer, CONFIG.max_length)
holdout_loader = DataLoader(
    holdout_dataset, 
    batch_size=CONFIG.batch_size * 2, 
    shuffle=False, 
    num_workers=2, 
    pin_memory=True
)

# Evaluate on held-out set (using best thresholds from validation)
holdout_f1, holdout_per_class, holdout_loss, holdout_probs, holdout_labels = evaluate(
    model, holdout_loader, device, best_thresholds
)

print(f"\n{'='*60}")
print(f"🏆 HELD-OUT TEST RESULTS (NEVER SEEN IN TRAINING)")
print(f"{'='*60}")
print(f"\nMacro F1: {holdout_f1:.4f}")
print(f"Loss: {holdout_loss:.4f}")
print(f"\nPer-Maxim Performance:")
for name in MAXIM_NAMES:
    sc = holdout_per_class[name]
    flag = " ⚠️" if sc['f1'] > 0.95 else " ✓" if sc['f1'] > 0.70 else " ⚠️"
    print(f"  {name}: F1={sc['f1']:.3f}, P={sc['precision']:.3f}, R={sc['recall']:.3f}{flag}")

# Generalization gap analysis
print(f"\n{'='*60}")
print(f"📊 GENERALIZATION ANALYSIS")
print(f"{'='*60}")
print(f"\nIn-distribution test:  F1 = {test_f1:.4f}")
print(f"Held-out test:         F1 = {holdout_f1:.4f}")

gen_gap = abs(test_f1 - holdout_f1)
print(f"Generalization gap:    {gen_gap:.4f}")

if gen_gap > 0.10:
    print(f"\n⚠️ LARGE GAP (>0.10): Significant overfitting detected!")
    print(f"   Model memorizing training patterns rather than learning generalizable features")
    print(f"   Recommendations:")
    print(f"   - Reduce model capacity (smaller model)")
    print(f"   - More regularization (dropout, weight decay)")
    print(f"   - More diverse training data")
elif gen_gap > 0.05:
    print(f"\n⚠️ MODERATE GAP (>0.05): Some overfitting")
    print(f"   Model generalizes reasonably but could be more robust")
else:
    print(f"\n✅ SMALL GAP (<0.05): Excellent generalization!")
    print(f"   Model learned robust features, not memorizing patterns")

# Final health assessment
print(f"\n{'='*60}")
print(f"🎯 FINAL MODEL HEALTH ASSESSMENT")
print(f"{'='*60}")

if holdout_f1 > 0.95:
    print(f"\n🚨 CRITICAL ALERT: Held-out F1 = {holdout_f1:.3f}")
    print(f"   This is SUSPICIOUSLY HIGH. Strongly suggests:")
    print(f"   1. Data leakage (check held-out creation)")
    print(f"   2. Phase 4 violations too synthetic/easy")
    print(f"   3. Model memorizing patterns, not learning")
    print(f"   ⚠️ DO NOT DEPLOY THIS MODEL - INVESTIGATE FIRST")
elif holdout_f1 > 0.85:
    print(f"\n✅ EXCELLENT: Held-out F1 = {holdout_f1:.3f}")
    print(f"   Strong performance with low generalization gap")
    print(f"   Model ready for Phase 7 evaluation")
elif holdout_f1 > 0.70:
    print(f"\n✅ GOOD: Held-out F1 = {holdout_f1:.3f}")
    print(f"   Solid performance, acceptable for deployment")
    print(f"   Consider additional tuning for production")
elif holdout_f1 > 0.55:
    print(f"\n⚠️ MODERATE: Held-out F1 = {holdout_f1:.3f}")
    print(f"   Better than baseline (55%) but needs improvement")
    print(f"   Recommendations:")
    print(f"   - More training data")
    print(f"   - Better hyperparameter tuning")
    print(f"   - Larger model (DeBERTa-base instead of small)")
else:
    print(f"\n❌ LOW: Held-out F1 = {holdout_f1:.3f}")
    print(f"   Performance below target")
    print(f"   Critical issues - do not proceed to Phase 7")

tracker.mark('Held-Out Evaluation', 'PASS', {
    'holdout_f1': holdout_f1,
    'holdout_loss': holdout_loss,
    'generalization_gap': gen_gap,
    'per_class': {k: v['f1'] for k, v in holdout_per_class.items()},
})


# ============================================================================
# FIX #4: NEW CELL 9b - PREDICTION DISTRIBUTION ANALYSIS
# ============================================================================
# INSERT THIS AS A NEW CELL AFTER CURRENT CELL 9 (error analysis):

# ============================================================================
# CELL 9b: PREDICTION DISTRIBUTION ANALYSIS
# ============================================================================
logger.info("=" * 60)
logger.info("📊 PREDICTION DISTRIBUTION ANALYSIS")
logger.info("=" * 60)

print(f"\n{'='*60}")
print(f"📊 MODEL CALIBRATION ANALYSIS")
print(f"{'='*60}")

for i, name in enumerate(MAXIM_NAMES):
    probs_for_class = test_probs[:, i]
    
    print(f"\n{name}:")
    print(f"  Mean:     {probs_for_class.mean():.3f}")
    print(f"  Std Dev:  {probs_for_class.std():.3f}")
    print(f"  Min/Max:  {probs_for_class.min():.3f} / {probs_for_class.max():.3f}")
    print(f"  Median:   {np.median(probs_for_class):.3f}")
    
    # Distribution bins
    bins = np.histogram(probs_for_class, bins=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])[0]
    total = len(probs_for_class)
    
    print(f"  Distribution:")
    print(f"    [0.0-0.1): {bins[0]:4d} ({100*bins[0]/total:5.1f}%) {'█' * int(50*bins[0]/total)}")
    print(f"    [0.1-0.3): {bins[1]:4d} ({100*bins[1]/total:5.1f}%) {'█' * int(50*bins[1]/total)}")
    print(f"    [0.3-0.5): {bins[2]:4d} ({100*bins[2]/total:5.1f}%) {'█' * int(50*bins[2]/total)}")
    print(f"    [0.5-0.7): {bins[3]:4d} ({100*bins[3]/total:5.1f}%) {'█' * int(50*bins[3]/total)}")
    print(f"    [0.7-0.9): {bins[4]:4d} ({100*bins[4]/total:5.1f}%) {'█' * int(50*bins[4]/total)}")
    print(f"    [0.9-1.0]: {bins[5]:4d} ({100*bins[5]/total:5.1f}%) {'█' * int(50*bins[5]/total)}")
    
    # Health checks
    if probs_for_class.std() < 0.10:
        print(f"  ⚠️ LOW VARIANCE: Model not confident in separating this class")
    
    middle_bins_pct = (bins[2] + bins[3]) / total
    if middle_bins_pct > 0.70:
        print(f"  ⚠️ CLUSTERED: {middle_bins_pct:.1%} predictions around 0.5")
        print(f"     Model struggling to separate {name}")
    
    extreme_bins_pct = (bins[0] + bins[5]) / total
    if extreme_bins_pct > 0.80:
        print(f"  ⚠️ OVERCONFIDENT: {extreme_bins_pct:.1%} predictions at extremes")
        print(f"     Model may be too confident (check for overfitting)")

# Overall calibration
all_probs_flat = test_probs.flatten()
print(f"\n{'='*60}")
print(f"Overall Model Calibration:")
print(f"  Mean prediction: {all_probs_flat.mean():.3f}")
print(f"  Std Dev:         {all_probs_flat.std():.3f}")

if all_probs_flat.std() < 0.15:
    print(f"  ⚠️ WARNING: Very low overall variance")
    print(f"     Model may have collapsed or is poorly calibrated")
elif all_probs_flat.std() > 0.35:
    print(f"  ⚠️ WARNING: Very high variance")
    print(f"     Model may be overconfident or unstable")
else:
    print(f"  ✅ Healthy variance - good calibration")

tracker.mark('Distribution Analysis', 'PASS')


# ============================================================================
# FIX #5: UPDATE CELL 10 - ADD HELD-OUT RESULTS TO JSON
# ============================================================================
# FIND THIS SECTION in Cell 10:
#   results = {
#       ...
#       'test': {
#           'macro_f1': test_f1,
#           ...
#       },
#       ...
#   }
#
# ADD THIS AFTER 'test': {...},:

    'holdout_test': {
        'macro_f1': holdout_f1,
        'loss': holdout_loss,
        'per_class': {name: holdout_per_class[name] for name in MAXIM_NAMES},
        'generalization_gap': abs(test_f1 - holdout_f1),
    },
