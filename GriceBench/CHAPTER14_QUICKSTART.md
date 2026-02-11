# Chapter 14: DPO Generator Evaluation - Quick Start Guide

## ✅ What's Ready

You now have a complete evaluation script:
```
GriceBench/scripts/evaluate_dpo_generator.py
```

This script will:
1. ✅ Load your DPO model with LoRA adapters
2. ✅ Load baseline GPT-2 Medium for comparison
3. ✅ Load your trained detector model
4. ✅ Generate responses from both models
5. ✅ Run detector on all responses
6. ✅ Calculate violation rates and statistical significance
7. ✅ Save comprehensive results

## 📦 Dependencies Installed

- ✅ `peft` (for LoRA model loading)
- ✅ `scipy` (for statistical tests)
- ✅ All other dependencies already installed

## 🚀 How to Run

### Step 1: Test Model Loading (2-3 minutes)

First, verify all models load correctly:

```bash
C:\Users\pushk\python310\python.exe scripts\evaluate_dpo_generator.py --test-load --dpo-model "c:\Users\pushk\OneDrive\Documents\Research Model\dpo_final" --detector-model "models\detector"
```

**Expected output:**
```
======================================================================
INITIALIZING DPO GENERATOR EVALUATOR
======================================================================

[1/4] Loading baseline GPT-2 Medium...
✓ Baseline loaded: 354,823,168 parameters

[2/4] Loading DPO model with LoRA adapters...
✓ DPO model loaded with LoRA adapters

[3/4] Loading detector model...
✓ Detector loaded

[4/4] Setting models to evaluation mode...
✓ All models ready

======================================================================
MODEL LOADING COMPLETE
======================================================================

✅ Model loading test successful!
```

### Step 2: Run Small-Scale Test (5-10 minutes)

Test with 10 examples to verify everything works:

```bash
C:\Users\pushk\python310\python.exe scripts\evaluate_dpo_generator.py --num-examples 10 --dpo-model "c:\Users\pushk\OneDrive\Documents\Research Model\dpo_final" --detector-model "models\detector" --test-data "data_processed\dpo_data\dpo_val.json"
```

**What happens:**
- Loads all models
- Generates 10 responses from each model
- Runs detector on all 20 responses
- Calculates violation rates
- Prints results
- Saves to `results/generator_evaluation/`

**Expected runtime:** 5-10 minutes on CPU

### Step 3: Run Full Evaluation (15-30 minutes)

Run on 100-500 examples for comprehensive results:

```bash
C:\Users\pushk\python310\python.exe scripts\evaluate_dpo_generator.py --num-examples 100 --dpo-model "c:\Users\pushk\OneDrive\Documents\Research Model\dpo_final" --detector-model "models\detector" --test-data "data_processed\dpo_data\dpo_val.json"
```

**Expected runtime:** 15-30 minutes for 100 examples on CPU

## 📊 Understanding the Results

### Console Output

You'll see a table like this:

```
======================================================================
EVALUATION RESULTS
======================================================================

📊 VIOLATION RATES:
----------------------------------------------------------------------
Maxim           Baseline     DPO          Improvement      p-value   
----------------------------------------------------------------------
Quantity         45.0%       32.0%       +13.0% (+28.9%)  0.0030 **
Quality          38.0%       25.0%       +13.0% (+34.2%)  0.0120 *
Relation         22.0%       18.0%        +4.0% (+18.2%)  0.1250
Manner           31.0%       24.0%        +7.0% (+22.6%)  0.0450 *
----------------------------------------------------------------------
Cooperative      28.0%       42.0%       +14.0% (+19.4%)
======================================================================

✅ SUMMARY:
  • Maxims improved: 4/4
  • Statistically significant: 3/4
  • Overall cooperative rate improved: True

🎉 EXCELLENT! DPO training significantly improved generation quality!
```

**Significance markers:**
- `***` = p < 0.001 (highly significant)
- `**` = p < 0.01 (very significant)
- `*` = p < 0.05 (significant)
- No marker = not statistically significant

### Saved Files

Results are saved to `results/generator_evaluation/`:

1. **`dpo_vs_baseline.json`** - Full results in JSON format
2. **`violation_rates.csv`** - Metrics table for Excel/analysis
3. **`example_outputs.txt`** - 20 example responses for manual inspection

## 🔍 What to Look For

### ✅ Good Signs

- **Violation rates decreased** for most/all maxims
- **Improvements are statistically significant** (p < 0.05)
- **Overall cooperative rate increased** by 10%+
- **Generated responses look fluent** in example outputs

### ⚠️ Warning Signs

- **No improvement** or worse performance
- **High p-values** (> 0.05) - improvements not significant
- **Gibberish responses** in example outputs
- **Very high violation rates** (> 80%) for both models

## 🐛 Troubleshooting

### Error: "FileNotFoundError: dpo_final"

**Fix:** Use absolute path for DPO model:
```bash
--dpo-model "c:\Users\pushk\OneDrive\Documents\Research Model\dpo_final"
```

### Error: "FileNotFoundError: models/detector"

**Fix:** Check detector model location:
```bash
dir models\detector
```
Should show files like `config.json`, `pytorch_model.bin`, etc.

### Error: "CUDA out of memory"

**Fix:** Script uses CPU by default. If you accidentally enabled CUDA, add:
```bash
--device cpu
```

### Slow Performance

**Expected:** CPU inference is slow (~1-2 sec per response)
- 10 examples = 5-10 minutes
- 100 examples = 15-30 minutes
- 500 examples = 1-2 hours

**This is normal!** Your laptop handles inference fine, just slower than GPU.

### Models Load but No Output

**Fix:** Check test data path:
```bash
dir data_processed\dpo_data\dpo_val.json
```

If missing, use alternative test data:
```bash
--test-data "data_processed\test_examples.json"
```

## 📝 Next Steps After Running

### If Results Look Good (Improvements > 10%)

1. ✅ Review example outputs manually
2. ✅ Proceed to Phase 2: Human Evaluation Prep
3. ✅ Run on larger dataset (500 examples) for paper

### If Results Look Mixed (Some improvements)

1. 📊 Analyze which maxims improved
2. 📝 Focus paper on those specific maxims
3. 🔍 Inspect examples to understand why
4. ✅ Still proceed to Phase 2

### If Results Look Bad (No improvements)

1. 🔍 Check example outputs - are they fluent?
2. 🐛 Verify DPO model loaded correctly
3. 📊 Check detector predictions - are they reasonable?
4. 💬 Share results with me for debugging

## 🎯 Command Reference

### Quick Commands

**Test loading:**
```bash
C:\Users\pushk\python310\python.exe scripts\evaluate_dpo_generator.py --test-load --dpo-model "c:\Users\pushk\OneDrive\Documents\Research Model\dpo_final" --detector-model "models\detector"
```

**Small test (10 examples):**
```bash
C:\Users\pushk\python310\python.exe scripts\evaluate_dpo_generator.py --num-examples 10 --dpo-model "c:\Users\pushk\OneDrive\Documents\Research Model\dpo_final" --detector-model "models\detector"
```

**Full evaluation (100 examples):**
```bash
C:\Users\pushk\python310\python.exe scripts\evaluate_dpo_generator.py --num-examples 100 --dpo-model "c:\Users\pushk\OneDrive\Documents\Research Model\dpo_final" --detector-model "models\detector"
```

**Large evaluation (500 examples):**
```bash
C:\Users\pushk\python310\python.exe scripts\evaluate_dpo_generator.py --num-examples 500 --dpo-model "c:\Users\pushk\OneDrive\Documents\Research Model\dpo_final" --detector-model "models\detector"
```

### All Available Options

```
--dpo-model PATH          Path to DPO model with LoRA adapters
--detector-model PATH     Path to detector model
--test-data PATH          Path to test data JSON file
--num-examples N          Number of examples to evaluate
--output-dir PATH         Where to save results
--device cpu/cuda         Device to run on (use 'cpu')
--test-load               Only test model loading
```

## ⏱️ Time Estimates

| Task | Examples | Time (CPU) |
|------|----------|------------|
| Model loading test | 0 | 2-3 min |
| Small test | 10 | 5-10 min |
| Medium evaluation | 100 | 15-30 min |
| Large evaluation | 500 | 1-2 hours |

**Recommendation:** Start with 10 examples, then run 100 for final results.

---

**Ready to start? Run the test loading command first!** 🚀
