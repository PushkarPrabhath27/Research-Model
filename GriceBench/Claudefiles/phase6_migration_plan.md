# Phase 6 Detector V2 → V3 Migration Plan

## Executive Decision: Should You Fix or Keep Current?

### ✅ Fix if ANY of these apply:
- [ ] You want publishable/defensible results
- [ ] You plan to use this in Phase 7 or production
- [ ] You want to know TRUE model performance
- [ ] You have 45-60 minutes available
- [ ] Previous runs gave F1 > 0.90 (suspiciously high)

### ❌ Keep current if ALL of these apply:
- [ ] This is just a proof-of-concept  
- [ ] You won't use results beyond Phase 6
- [ ] You're abandoning this approach anyway
- [ ] You have <30 minutes available

**Recommendation:** FIX IT. The issues are critical and will bite you later.

---

## The 5 Critical Issues (In Order of Severity)

### 🔴 Issue #1: Missing pos_weight (BLOCKS SUCCESS)
**Severity:** CRITICAL - Will cause model collapse (F1=0.0)  
**Evidence:** Your Phase 6 history shows F1=0.0 without pos_weight  
**Time to fix:** 2 minutes  
**Must fix:** YES

### 🔴 Issue #2: No held-out test set (INVALIDATES RESULTS)  
**Severity:** HIGH - Can't trust your F1 scores  
**Evidence:** All data from same pool = can memorize patterns  
**Time to fix:** 5 minutes  
**Must fix:** YES (for any production use)

### 🟡 Issue #3: No real-time health checks (WASTES TIME)
**Severity:** MEDIUM - Will waste GPU time on failed runs  
**Evidence:** Previous runs collapsed after 3 epochs (wasted 30+ min)  
**Time to fix:** 3 minutes  
**Must fix:** RECOMMENDED

### 🟡 Issue #4: No calibration analysis (MISS PROBLEMS)
**Severity:** MEDIUM - Won't detect poor confidence  
**Evidence:** Can't diagnose why model fails on edge cases  
**Time to fix:** 2 minutes  
**Must fix:** NICE TO HAVE

### 🟢 Issue #5: No per-method breakdown in training (LOW)
**Severity:** LOW - Already have this in test set  
**Evidence:** Have it in Cell 8 already  
**Time to fix:** 0 minutes (already done)  
**Must fix:** NO

---

## Migration Paths

### Path A: MINIMAL FIX (15 minutes)
**What:** Fix only Issue #1 (pos_weight)  
**Why:** Prevents model collapse  
**Expected F1:** 0.60-0.75 (realistic)  
**Use case:** Quick validation that data loads

**Steps:**
1. Open Cell 7
2. Replace `criterion = nn.BCEWithLogitsLoss()` with code from Fix #1
3. Run notebook
4. Check if F1 > 0.50

---

### Path B: PRODUCTION FIX (45 minutes) ⭐ RECOMMENDED
**What:** Fix Issues #1, #2, #3  
**Why:** Gets you production-ready results  
**Expected F1:** 0.70-0.82 (held-out)  
**Use case:** Proceeding to Phase 7

**Steps:**
1. Apply Fix #1 (pos_weight) - 2 min
2. Insert Cell 4b (held-out creation) - 5 min
3. Insert Cell 8b (held-out evaluation) - 5 min
4. Apply Fix #2 (health checks) - 3 min
5. Run notebook - 30 min
6. Analyze held-out results

**Success criteria:**
- ✅ Training completes without health alerts
- ✅ Held-out F1 within 0.05 of test F1
- ✅ Held-out F1 between 0.70-0.85
- ✅ No class with F1 < 0.60

---

### Path C: RESEARCH-GRADE FIX (60 minutes)
**What:** Fix all issues + calibration analysis  
**Why:** Publishable results with full diagnostics  
**Expected F1:** 0.70-0.82 (with confidence)  
**Use case:** Publishing paper, defending to stakeholders

**Steps:**
1. All steps from Path B
2. Insert Cell 9b (calibration analysis) - 2 min
3. Add holdout results to JSON (Fix #5) - 1 min
4. Run full diagnostic suite
5. Generate comprehensive report

**Deliverables:**
- Full JSON with all metrics
- Calibration plots
- Error analysis
- Generalization gap analysis

---

## Step-by-Step Implementation (Path B)

### Step 1: Backup Current Notebook
```bash
# In Kaggle, use "File > Save Version"
# Or download as KAGGLE_PHASE6_DETECTOR_V2_BACKUP.ipynb
```

### Step 2: Apply Fix #1 (pos_weight)
1. **Open Cell 7** (Training Loop)
2. **Find line:** `criterion = nn.BCEWithLogitsLoss()`
3. **Replace with** code from `phase6_v3_code_snippets.py` (Fix #1 section)
4. **Verify:** Cell should now have ~15 extra lines calculating pos_weight

### Step 3: Insert Cell 4b (Held-Out Test Creation)
1. **After Cell 4**, click "+ Code"
2. **Paste** entire Fix #3A section from `phase6_v3_code_snippets.py`
3. **Verify:** New cell should start with "CELL 4b: CREATE TRUE HELD-OUT"

### Step 4: Insert Cell 8b (Held-Out Evaluation)
1. **After Cell 8** (current test evaluation), click "+ Code"
2. **Paste** entire Fix #3B section from `phase6_v3_code_snippets.py`
3. **Verify:** New cell should evaluate `holdout_test_data`

### Step 5: Apply Fix #2 (Health Checks)
1. **Open Cell 7** (Training Loop)
2. **Find line:** `epoch_result = {`
3. **INSERT BEFORE** that line the health checks from Fix #2
4. **Verify:** Should see 5 health checks (variance, threshold, loss, F1, class)

### Step 6: Run Notebook
1. **Click "Run All"** or restart kernel + run all cells
2. **Expected time:** 30-40 minutes on T4 GPU
3. **Monitor for:** Health check warnings during training

### Step 7: Analyze Results

**What to look for in output:**

```
✅ GOOD SIGNS:
- "✅ ALL DATA ASSERTIONS PASSED" in Cell 4
- No health warnings in epochs 1-3
- Val F1 improves for 2-3 epochs then plateaus
- Held-out F1 within 0.05 of test F1
- Held-out F1 between 0.70-0.85

⚠️ WARNING SIGNS:
- "⚠️ COLLAPSE ALERT" → Model collapsing, check pos_weight
- "⚠️ SUSPICIOUS" → F1 too high too early, possible data issue
- Generalization gap > 0.10 → Overfitting
- Held-out F1 > 0.95 → Data leakage or synthetic

❌ FAILURE SIGNS:
- FileNotFoundError → natural_violations.json missing
- Assertion failed → Phase 4 data <1000 examples
- Held-out F1 < 0.60 → Model not learning
```

---

## Expected Results Timeline

### Baseline (Current V2):
```
Train: 50 min
Val F1: 0.947 (suspiciously high)
Test F1: 0.947 (same pool as val)
Held-out F1: ??? (doesn't exist)
Confidence: LOW (can't trust these scores)
```

### After Path A (Minimal Fix):
```
Train: 30 min
Val F1: 0.72 (more realistic)
Test F1: 0.70 (close to val)
Held-out F1: ??? (still doesn't exist)
Confidence: MEDIUM (pos_weight fixed, but no true test)
```

### After Path B (Production Fix):
```
Train: 35 min
Val F1: 0.76
Test F1: 0.74
Held-out F1: 0.73 (within 0.03 of test ✓)
Confidence: HIGH (all metrics align, ready for Phase 7)
```

### After Path C (Research-Grade):
```
Train: 35 min
Val F1: 0.76
Test F1: 0.74
Held-out F1: 0.73
+ Calibration plots
+ Error analysis by generation method
+ Per-class confidence distributions
Confidence: VERY HIGH (publication-ready)
```

---

## Troubleshooting Guide

### Problem: "FileNotFoundError: natural_violations.json"
**Cause:** Phase 4 data not uploaded to Kaggle dataset  
**Fix:** 
1. Go to your Kaggle dataset `gricebench-scientific-fix`
2. Click "New Version"
3. Upload `natural_violations.json`
4. Re-run notebook

### Problem: "Assertion failed: Only 200 violations"
**Cause:** Wrong file loaded or file corrupted  
**Fix:**
1. Check file size: should be >500KB
2. Check file structure: `jq '.[0]' natural_violations.json`
3. Should have keys: `violated_response`, `original_response`, `labels`

### Problem: "⚠️ COLLAPSE ALERT" during training
**Cause:** pos_weight not applied correctly  
**Fix:**
1. Check Cell 7 has `criterion = nn.BCEWithLogitsLoss(pos_weight=...)`
2. Print `criterion.pos_weight` to verify it's not None
3. Restart kernel and re-run

### Problem: Held-out F1 > 0.95 (too high)
**Cause:** Data leakage or synthetic patterns  
**Fix:**
1. Check Cell 4b has `random.seed(999)` (different seed)
2. Verify assertion: `len(holdout & training) == 0`
3. Review Phase 4 data quality - may be too synthetic

### Problem: Large generalization gap (>0.10)
**Cause:** Overfitting to training distribution  
**Fix:**
1. Reduce epochs (use best_epoch from validation)
2. Increase dropout (0.1 → 0.2)
3. Add more diverse training data

---

## Final Checklist Before Running

- [ ] Backed up current notebook
- [ ] Applied Fix #1 (pos_weight) in Cell 7
- [ ] Inserted Cell 4b (held-out creation)
- [ ] Inserted Cell 8b (held-out evaluation)  
- [ ] Applied Fix #2 (health checks) in Cell 7
- [ ] Verified `natural_violations.json` exists in dataset
- [ ] GPU enabled in notebook settings
- [ ] Have 40-50 minutes for training
- [ ] Ready to analyze results

---

## Success Metrics

### Minimum Acceptable (to proceed to Phase 7):
- ✅ Held-out F1 ≥ 0.65
- ✅ Generalization gap ≤ 0.10
- ✅ All classes F1 ≥ 0.50
- ✅ No health warnings during training

### Target (production-ready):
- ✅ Held-out F1 ≥ 0.75
- ✅ Generalization gap ≤ 0.05
- ✅ All classes F1 ≥ 0.65
- ✅ Calibration variance healthy (0.15-0.35)

### Excellent (publication-ready):
- ✅ Held-out F1 ≥ 0.80
- ✅ Generalization gap ≤ 0.03
- ✅ All classes F1 ≥ 0.75
- ✅ Consistent per-method performance

---

## What to Do After Running

### If Results are Good (Held-out F1 > 0.70, gap < 0.05):
1. ✅ Save the notebook version
2. ✅ Download `detector_v2_results.json`
3. ✅ Download `best_model.pt`
4. ✅ Proceed to Phase 7 evaluation
5. ✅ Document these metrics in your report

### If Results are Suspicious (Held-out F1 > 0.95):
1. ⚠️ DO NOT proceed to Phase 7
2. ⚠️ Investigate data quality
3. ⚠️ Check for leakage in Cell 4b
4. ⚠️ Review Phase 4 generation methods
5. ⚠️ Consider re-generating Phase 4 with harder violations

### If Results are Low (Held-out F1 < 0.60):
1. ⚠️ Check health warnings - did model collapse?
2. ⚠️ Verify pos_weight is applied
3. ⚠️ Try larger model (deberta-v3-base instead of small)
4. ⚠️ Increase training data (need more Phase 4 examples)
5. ⚠️ Tune hyperparameters (learning rate, epochs)

---

## Time Investment Summary

| Path | Setup | Training | Analysis | Total |
|------|-------|----------|----------|-------|
| A: Minimal | 5 min | 30 min | 5 min | 40 min |
| B: Production | 15 min | 35 min | 10 min | 60 min |
| C: Research | 20 min | 35 min | 15 min | 70 min |

**Recommendation:** Invest 60 minutes in Path B for production-ready results.

---

## My Final Recommendation

**Use Path B (Production Fix)** because:

1. ✅ Only 15 min more than Path A
2. ✅ Gives you TRUE performance metrics
3. ✅ Catches problems during training (saves time)
4. ✅ Ready for Phase 7 immediately
5. ✅ Defensible if questioned

The alternative (keeping current V2):
- ❌ Can't trust the 0.947 F1 score
- ❌ Will likely fail in Phase 7
- ❌ Will need to redo training anyway
- ❌ Wastes more time long-term

**Total time saved by fixing now: 2-3 hours** (vs. debugging later)
