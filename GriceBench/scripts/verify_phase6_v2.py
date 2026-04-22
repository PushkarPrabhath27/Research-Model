"""Comprehensive verification of Phase 6 Detector V2 notebook."""
import json, sys, io

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

nb = json.load(open(
    r'c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\KAGGLE_PHASE6_DETECTOR_V2.ipynb',
    'r', encoding='utf-8'
))

cells = nb['cells']
src = ''.join(
    ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    for c in cells
)

print("=" * 60)
print("PHASE 6 DETECTOR V2 -- NOTEBOOK VERIFICATION")
print("=" * 60)

all_pass = True

def check(condition, desc):
    global all_pass
    icon = "PASS" if condition else "FAIL"
    if not condition:
        all_pass = False
    print(f"  [{icon}] {desc}")
    return condition

# 1. Structure
print("\n[1] STRUCTURE")
check(len(cells) == 15, f"Cell count = {len(cells)} (expected 15)")
code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
check(code_cells == 14, f"Code cells = {code_cells} (expected 14)")

# 2. Kaggle metadata
print("\n[2] KAGGLE METADATA")
kaggle = nb['metadata']['kaggle']
check(kaggle.get('isGpuEnabled', False), "GPU enabled")
check(kaggle.get('isInternetEnabled', False), "Internet enabled")
check('gricebench' in str(kaggle.get('dataSources', '')).lower(), "Dataset configured")

# 3. Fix #1: pos_weight
print("\n[3] FIX #1: pos_weight (prevents model collapse)")
check("BCEWithLogitsLoss(pos_weight=" in src, "pos_weight in loss function")
check("pos_weight_values" in src, "pos_weight calculation")
check("register_buffer('pos_weight'" in src, "pos_weight buffer registration")
check("np.clip(pos_weight_values" in src, "pos_weight capping (prevents extreme weights)")
check("pos_counts" in src, "positive class counting")
check("neg_counts" in src, "negative class counting")

# 4. Fix #2: Health checks
print("\n[4] FIX #2: Real-time health checks")
check("COLLAPSE ALERT" in src, "Model collapse detection")
check("THRESHOLD ALERT" in src, "Threshold clustering detection")
check("OVERFITTING" in src, "Overfitting detection")
check("SUSPICIOUS" in src, "Suspicious F1 detection")
check("CLASS COLLAPSE" in src, "Per-class collapse detection")
check("health_alerts" in src, "Alert tracking list")

# 5. Fix #3: Held-out test
print("\n[5] FIX #3: Held-out test set")
check("holdout_test_data" in src, "Holdout variable")
check("random.seed(999)" in src, "Independent seed (999 != 42)")
check("holdout_violations" in src, "Holdout violations")
check("holdout_clean" in src, "Holdout clean examples")
check("holdout_f1" in src, "Holdout F1 evaluation")
check("gen_gap" in src, "Generalization gap calculation")
check("holdout_loader" in src, "Holdout DataLoader")
check("COMPLETELY UNSEEN" in src, "Explicit unseen labeling")

# 6. Fix #4: Calibration
print("\n[6] FIX #4: Calibration analysis")
check("CALIBRATION ANALYSIS" in src, "Calibration cell present")
check("np.histogram" in src, "Distribution binning")
check("LOW VARIANCE" in src, "Variance warning")
check("CLUSTERED" in src, "Clustering warning")

# 7. Fix #5: Results JSON
print("\n[7] FIX #5: Comprehensive results JSON")
check("'holdout_test':" in src, "Holdout section in results")
check("'generalization_gap':" in src, "Generalization gap in results")
check("'health_alerts':" in src, "Health alerts in results")
check("'pos_weight':" in src, "pos_weight in results")
check("'data_verification':" in src, "Data verification proof")
check("'holdout_per_method':" in src, "Per-method holdout results")

# 8. Data assertions
print("\n[8] DATA ASSERTIONS")
check("natural_violations.json" in src, "Correct filename")
check("assert len(violations) >= CONFIG.min_phase4_violations" in src, "Min violations assert")
check("assert count >= 100" in src, "Per-maxim minimum assert")
check("assert len(clean_examples) >= 500" in src, "Min clean examples assert")
check("DATA LEAKAGE" in src, "Leakage assertion")

# 9. Bug checks
print("\n[9] BUG CHECKS")
total_mem_bug = src.count("total_mem") - src.count("total_memory")
check(total_mem_bug <= 0, f"total_mem bug: {total_mem_bug} instances")
check("improved_violations.json" not in src, "No wrong filename (improved_violations.json)")

# 10. Error analysis & per-method eval
print("\n[10] EVALUATION FEATURES")
check("error_by_maxim" in src, "Error analysis by maxim")
check("error_by_method" in src, "Error analysis by generation method")
check("method_results" in src, "Per-generation-method evaluation")
check("Top 10 Worst" in src, "Top misclassification display")

# 11. Cell summary
print("\n[11] CELL CONTENTS")
for i, c in enumerate(cells):
    cell_src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    first_line = cell_src.strip().split('\n')[0][:75]
    print(f"  Cell {i:2d} [{c['cell_type']:4s}]: {first_line}")

# Final
print("\n" + "=" * 60)
if all_pass:
    print("ALL VERIFICATION CHECKS PASSED")
else:
    print("SOME CHECKS FAILED -- review above")
print("=" * 60)
