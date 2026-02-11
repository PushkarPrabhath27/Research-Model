# Chapter 14: DPO Evaluation on Kaggle - Complete Beginner Guide

## 🎯 What You'll Do

You'll run a Kaggle notebook that:
1. Loads your DPO model and baseline GPT-2
2. Generates 100 responses from each
3. Runs detector on all responses
4. Calculates violation rates and improvements
5. Saves results for download

**Time:** 30-40 minutes total (mostly waiting for GPU)

---

## 📋 Prerequisites

Before starting, you need:
- ✅ Kaggle account (free)
- ✅ Your DPO model (`dpo_final` folder - 25MB)
- ✅ Your detector model (from `GriceBench/models/detector`)
- ✅ Test data (`dpo_val.json`)

---

## 🚀 Step-by-Step Guide

### **STEP 1: Upload Your Models to Kaggle (10 minutes)**

#### 1.1: Create DPO Model Dataset

1. **Go to Kaggle Datasets**
   - Open browser → https://www.kaggle.com/datasets
   - Click **"New Dataset"** button (top right, blue button)

2. **Upload DPO Model**
   - Click **"Upload Files"** or drag-and-drop
   - Select your entire `dpo_final` folder
   - Files to upload:
     - `adapter_config.json`
     - `adapter_model.safetensors` (25MB)
     - `README.md`
     - `tokenizer.json`
     - `tokenizer_config.json`
     - `merges.txt`
     - `vocab.json`
     - `special_tokens_map.json`
     - `training_args.bin`

3. **Configure Dataset**
   - **Title:** `DPO Generator Model`
   - **Subtitle:** (leave blank or add description)
   - **Tags:** Add `nlp`, `gpt2`, `lora`
   - **License:** Choose any (e.g., "Apache 2.0")
   - **Visibility:** Private (recommended) or Public

4. **Create Dataset**
   - Click **"Create"** button (bottom right)
   - Wait for upload to complete (1-2 minutes)
   - **Copy the dataset path** (looks like: `/kaggle/input/dpo-generator-model`)
   - **Save this path** - you'll need it later!

#### 1.2: Create Detector Model Dataset

1. **Repeat same process** for detector model
   - Go to https://www.kaggle.com/datasets
   - Click **"New Dataset"**
   - Upload files from `GriceBench/models/detector/`:
     - `config.json`
     - `pytorch_model.bin` (or `model.safetensors`)
     - `tokenizer.json`
     - `tokenizer_config.json`
     - `vocab.txt` (or similar)
     - Other tokenizer files

2. **Configure:**
   - **Title:** `Gricean Maxim Detector`
   - **Tags:** `nlp`, `detector`, `deberta`
   - Click **"Create"**

3. **Copy detector dataset path** (e.g., `/kaggle/input/gricean-maxim-detector`)

#### 1.3: Create Test Data Dataset

1. **Repeat for test data**
   - Upload `GriceBench/data_processed/dpo_data/dpo_val.json`
   - **Title:** `DPO Test Data`
   - Click **"Create"**

2. **Copy test data path** (e.g., `/kaggle/input/dpo-test-data/dpo_val.json`)

**✅ Checkpoint:** You should now have 3 dataset URLs saved!

---

### **STEP 2: Create Kaggle Notebook (5 minutes)**

1. **Go to Kaggle Notebooks**
   - Open https://www.kaggle.com/code
   - Click **"New Notebook"** (top right)

2. **Choose Notebook Type**
   - Select **"Notebook"** (not Script)
   - This opens a Jupyter-style interface

3. **Enable GPU**
   - Look at right sidebar
   - Find **"Accelerator"** section
   - Click dropdown
   - Select **"GPU T4 x2"** (or "GPU P100" if available)
   - **IMPORTANT:** Make sure GPU is enabled!

4. **Add Your Datasets**
   - Right sidebar → **"Input"** section
   - Click **"+ Add Data"**
   - Click **"Your Datasets"** tab
   - Find and click your 3 datasets:
     - DPO Generator Model
     - Gricean Maxim Detector
     - DPO Test Data
   - Click **"Add"** for each

5. **Verify Datasets Added**
   - In right sidebar under "Input", you should see all 3 datasets
   - Click on each to see the file paths
   - **Write down the exact paths** - you'll need them!

**Example paths:**
```
DPO Model: /kaggle/input/dpo-generator-model/dpo_final
Detector: /kaggle/input/gricean-maxim-detector
Test Data: /kaggle/input/dpo-test-data/dpo_val.json
```

---

### **STEP 3: Copy Code into Notebook (5 minutes)**

1. **Open the code file**
   - On your computer, open `GriceBench/KAGGLE_CHAPTER14_EVALUATION.py`
   - This file has all the code you need

2. **Copy Cell 1**
   - In the file, find the section marked `# CELL 1: Install Required Packages`
   - Copy everything from `# CELL 1` to the next `# CELL 2` (don't include CELL 2)
   - In Kaggle notebook, paste into the first cell
   - The cell should start with `import sys`

3. **Add More Cells**
   - Click **"+ Code"** button below the first cell
   - This creates a new empty cell
   - Copy `# CELL 2` content and paste
   - Repeat for all 10 cells

**OR Easier Method:**
- Copy ALL the code from `KAGGLE_CHAPTER14_EVALUATION.py`
- Paste into first cell
- Kaggle will automatically split it into cells at the `# ===` lines

4. **Update Paths in Cell 2**
   - Find Cell 2 (the one with `CONFIG = {`)
   - Update these three lines with YOUR dataset paths:
   
   ```python
   'dpo_model_path': '/kaggle/input/YOUR-DPO-DATASET/dpo_final',
   'detector_model_path': '/kaggle/input/YOUR-DETECTOR-DATASET',
   'test_data_path': '/kaggle/input/YOUR-TEST-DATA/dpo_val.json',
   ```
   
   Replace `YOUR-DPO-DATASET`, `YOUR-DETECTOR-DATASET`, `YOUR-TEST-DATA` with your actual dataset names!

**✅ Checkpoint:** You should have 10 code cells with correct paths!

---

### **STEP 4: Run the Notebook (20-30 minutes)**

#### 4.1: Save Your Notebook First
- Click **"Save Version"** (top right)
- Choose **"Save & Run All"** (recommended)
- OR choose **"Quick Save"** if you want to run manually

#### 4.2: Run All Cells (Recommended)
- If you chose "Save & Run All", skip to 4.3
- If not, click **"Run All"** button (top toolbar)
- Alternatively, run each cell one by one:
  - Click on Cell 1
  - Press **Shift + Enter** (runs cell and moves to next)
  - Repeat for all cells

#### 4.3: Monitor Progress

**Cell 1 (Package Check):** ~5 seconds
- Should print package versions
- Look for ✓ marks

**Cell 2 (Configuration):** ~1 second
- Prints your configuration
- **CHECK:** Make sure paths look correct!

**Cell 3 (Load Models):** ~2-3 minutes
- Downloads GPT-2 Medium (350MB) - only first time
- Loads DPO model with LoRA
- Loads detector
- **WATCH FOR:** Progress bars for downloads
- **SUCCESS:** Should print "MODEL LOADING COMPLETE"

**Cell 4 (Helper Functions):** ~1 second
- Defines functions
- Should print "Helper functions defined!"

**Cell 5 (Load Test Data):** ~1 second
- Loads your test data
- Should print number of examples loaded

**Cell 6 (Run Evaluation):** ~15-20 minutes ⏰
- **THIS IS THE LONG ONE**
- Shows progress bar: "Generating & Detecting"
- Generates 100 responses from baseline
- Generates 100 responses from DPO
- Runs detector on all 200 responses
- **BE PATIENT:** This takes time even with GPU!
- **SUCCESS:** Should print "Evaluation complete!"

**Cell 7 (Calculate Metrics):** ~1 second
- Calculates violation rates
- Should print "Metrics calculated!"

**Cell 8 (Display Results):** ~1 second
- **THIS IS THE IMPORTANT ONE**
- Shows violation rates table
- Shows improvements
- **LOOK FOR:**
  - Positive improvements (DPO better than baseline)
  - "EXCELLENT!" or "GOOD!" message

**Cell 9 (Show Examples):** ~1 second
- Shows 5 example responses
- Compare baseline vs DPO responses

**Cell 10 (Save Results):** ~1 second
- Saves results to files
- Should print "RESULTS SAVED!"

#### 4.4: What to Do If Something Fails

**Error: "FileNotFoundError"**
- **Problem:** Dataset path is wrong
- **Fix:** Go back to Cell 2, check paths match your datasets exactly

**Error: "CUDA out of memory"**
- **Problem:** GPU ran out of memory
- **Fix:** In Cell 2, change `'num_examples': 100` to `'num_examples': 50`

**Error: "Dataset not found"**
- **Problem:** Dataset not added to notebook
- **Fix:** Right sidebar → Input → Add your datasets

**Cell stuck/frozen:**
- **Fix:** Click **"Interrupt"** button (top toolbar)
- Then **"Restart & Run All"**

---

### **STEP 5: Download Results (2 minutes)**

1. **Find Output Files**
   - Look at right sidebar
   - Click **"Output"** tab (next to "Input")
   - You should see:
     - `dpo_evaluation_results.json`
     - `violation_rates.csv`

2. **Download Files**
   - Click the **three dots (⋮)** next to each file
   - Click **"Download"**
   - Files will download to your computer

3. **Save to Project Folder**
   - Move downloaded files to:
     `GriceBench/results/generator_evaluation/`

**✅ DONE!** You now have your evaluation results!

---

## 📊 Understanding Your Results

### **The Results Table**

When Cell 8 runs, you'll see something like:

```
📊 VIOLATION RATES:
----------------------------------------------------------------------
Maxim           Baseline     DPO          Improvement      
----------------------------------------------------------------------
Quantity         45.0%       32.0%       +13.0% (+28.9%)
Quality          38.0%       25.0%       +13.0% (+34.2%)
Relation         22.0%       18.0%        +4.0% (+18.2%)
Manner           31.0%       24.0%        +7.0% (+22.6%)
----------------------------------------------------------------------
Cooperative      28.0%       42.0%       +14.0% (+50.0%)
```

**What this means:**
- **Baseline:** Violation rates for regular GPT-2
- **DPO:** Violation rates for your DPO-trained model
- **Improvement:** How much DPO reduced violations (positive = good!)
- **Cooperative:** % of responses with NO violations (higher = better)

### **Good Results Look Like:**
- ✅ All maxims show positive improvement
- ✅ Improvements > 10% for most maxims
- ✅ Cooperative rate increased by 10%+
- ✅ Message says "EXCELLENT!" or "GOOD!"

### **If Results Are Mixed:**
- ⚠️ Some maxims improved, some didn't
- ⚠️ Small improvements (< 5%)
- ⚠️ Message says "Mixed results"
- **Don't worry!** Even partial improvement is publishable

---

## 🔧 Troubleshooting Guide

### **Problem: Can't find dataset paths**

**Solution:**
1. In Kaggle notebook, right sidebar → Input
2. Click on your dataset
3. Click "Copy API command"
4. The path is in that command

### **Problem: Notebook runs out of time**

**Solution:**
- Kaggle has 12-hour limit
- This notebook takes ~30 minutes
- If it times out, just restart and run again

### **Problem: GPU not available**

**Solution:**
1. Right sidebar → Accelerator
2. Make sure GPU is selected
3. If no GPU available, wait a few minutes and try again
4. Kaggle gives 30 hours/week of free GPU

### **Problem: Results don't look good**

**Solution:**
- This is okay! It's still publishable
- Check Cell 9 to see example responses
- If responses look fluent, detector might be too strict
- If responses are gibberish, model might not have loaded correctly

---

## 📝 What to Do After Running

1. **Download both result files**
2. **Review the results table** - note which maxims improved
3. **Read example outputs** (Cell 9) - do they make sense?
4. **Share results with me** - I'll help interpret them
5. **We'll proceed to Phase 2** - Human evaluation prep

---

## ⏱️ Time Breakdown

| Step | Time | What You Do |
|------|------|-------------|
| Upload datasets | 10 min | Upload 3 folders to Kaggle |
| Create notebook | 5 min | Set up notebook, add datasets |
| Copy code | 5 min | Paste code, update paths |
| Run notebook | 20-30 min | Click "Run All", wait |
| Download results | 2 min | Download 2 files |
| **TOTAL** | **~40 min** | **Mostly waiting** |

---

## 🎯 Quick Checklist

Before running, make sure:
- [ ] Created 3 Kaggle datasets (DPO model, detector, test data)
- [ ] Saved all 3 dataset paths
- [ ] Created new Kaggle notebook
- [ ] Enabled GPU (Accelerator → GPU T4 x2)
- [ ] Added all 3 datasets to notebook (Input section)
- [ ] Copied all 10 cells of code
- [ ] Updated paths in Cell 2 CONFIG
- [ ] Saved notebook

Then:
- [ ] Click "Run All"
- [ ] Wait 20-30 minutes
- [ ] Check Cell 8 for results
- [ ] Download result files from Output tab

---

## 🆘 Need Help?

**If you get stuck:**
1. Take a screenshot of the error
2. Note which cell failed
3. Share with me and I'll help debug

**Common issues I can help with:**
- Dataset path problems
- Model loading errors
- GPU/memory issues
- Interpreting results

---

## 🎉 Success Criteria

You'll know it worked when:
- ✅ Cell 3 prints "MODEL LOADING COMPLETE"
- ✅ Cell 6 shows progress bar completing
- ✅ Cell 8 shows results table
- ✅ Cell 10 prints "RESULTS SAVED!"
- ✅ Output tab has 2 downloadable files

**Then you're done with Phase 1!** 🚀

---

## 📸 Visual Guide (What You'll See)

### **Kaggle Notebook Interface:**
```
┌─────────────────────────────────────────────────────────┐
│  [Kaggle Logo]  Your Notebook Name        [Save] [Run] │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Code Cell 1:                              Right        │
│  ┌──────────────────────────┐             Sidebar:     │
│  │ import sys               │             ┌──────────┐ │
│  │ print("Python version")  │             │ Input    │ │
│  │ ...                      │             │ ├─ DPO   │ │
│  └──────────────────────────┘             │ ├─ Det   │ │
│  [Output shows here]                      │ └─ Data  │ │
│                                            │          │ │
│  Code Cell 2:                              │ Output   │ │
│  ┌──────────────────────────┐             │ ├─ .json│ │
│  │ CONFIG = {               │             │ └─ .csv │ │
│  │   'dpo_model_path': ...  │             └──────────┘ │
│  └──────────────────────────┘                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### **Results Table (Cell 8):**
```
📊 VIOLATION RATES:
----------------------------------------------------------------------
Maxim           Baseline     DPO          Improvement      
----------------------------------------------------------------------
Quantity         45.0%       32.0%       +13.0% (+28.9%)  ← Good!
Quality          38.0%       25.0%       +13.0% (+34.2%)  ← Good!
Relation         22.0%       18.0%        +4.0% (+18.2%)  ← OK
Manner           31.0%       24.0%        +7.0% (+22.6%)  ← Good!
----------------------------------------------------------------------
Cooperative      28.0%       42.0%       +14.0% (+50.0%)  ← Excellent!
======================================================================

✅ SUMMARY:
  • Maxims improved: 4/4
  • Overall cooperative rate improved: True

🎉 EXCELLENT! DPO training significantly improved generation quality!
```

---

**Ready to start? Begin with STEP 1!** 🚀
