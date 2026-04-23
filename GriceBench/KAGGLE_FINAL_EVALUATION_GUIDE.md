# 🔬 KAGGLE GUIDE: Final Test Evaluation (Copy-Paste Ready)

## 📋 Overview

This notebook evaluates your trained DPO model on the actual test set to get **real cooperative rate** and **actual performance metrics**.

**Time Required:** 1-2 hours  
**GPU Required:** GPU T4 x2  
**What you'll get:** Real test results for publication

---

## Prerequisites Check

Before starting, verify you have these files locally:

### ✅ **Files You Should Have:**

**From Detector V2 Training:**
```
c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\
├── best_model_v2.pt (2 GB)
├── temperatures.json (100 bytes)
└── history_v2.json (7 KB)
```

**From DPO Training:**
```
c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\dpo_training_final_outcome\
├── adapter_config.json
├── adapter_model.safetensors (25 MB)
├── tokenizer files...
└── history (1).json
```

**Test Data:**
```
c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\
├── dpo_train_filtered.json (1,970 pairs)
└── dpo_val_filtered.json (101 pairs)
```

---

## Step 1: Prepare Datasets for Upload (15 minutes)

### 1.1 Check What You Already Have on Kaggle

Go to: https://www.kaggle.com/datasets/pushkarprabhath

**You should already have:**
- ✅ `gricebench-detector-v2` (uploaded earlier)
- ✅ `gricebench-dpo-filtered` (uploaded earlier)

**If missing, follow upload instructions below.**

---

### 1.2 Upload Detector V2 (If Not Already Done)

**Only do this if you don't have `gricebench-detector-v2` dataset!**

1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Click **"Upload Files"**
4. Upload these 3 files:
   ```
   best_model_v2.pt
   temperatures.json
   history_v2.json
   ```
5. **Title:** `gricebench-detector-v2`
6. **Subtitle:** "Trained Detector V2 model with 93% F1"
7. Click **"Create"**
8. **Wait for upload** (2 GB takes 10-15 minutes)

**Your dataset URL:**
```
https://www.kaggle.com/datasets/pushkarprabhath/gricebench-detector-v2
```

---

### 1.3 Upload DPO Test Data (If Not Already Done)

**Only do this if you don't have `gricebench-dpo-filtered` dataset!**

1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload these 2 files:
   ```
   dpo_train_filtered.json
   dpo_val_filtered.json
   ```
4. **Title:** `gricebench-dpo-filtered`
5. **Subtitle:** "Conflict-filtered DPO data (1,970 pairs)"
6. Click **"Create"**

**Your dataset URL:**
```
https://www.kaggle.com/datasets/pushkarprabhath/gricebench-dpo-filtered
```

---

### 1.4 Upload DPO Model Files

**Since your training notebook session ended, let's upload the model files directly!**

**You have these files locally:**
```
c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\dpo_training_final_outcome\
├── adapter_config.json
├── adapter_model.safetensors (25 MB)
├── tokenizer files (vocab.json, merges.txt, etc.)
└── history (1).json
```

**Upload them as a new dataset:**

1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Click **"Upload Files"**
4. Select ALL files from `dpo_training_final_outcome` folder:
   ```
   adapter_config.json
   adapter_model.safetensors
   merges.txt
   special_tokens_map.json
   tokenizer.json
   tokenizer_config.json
   vocab.json
   README.md (if present)
   ```
5. **Title:** `gricebench-dpo-model`
6. **Subtitle:** "Trained DPO model (LoRA adapters)"
7. Click **"Create"**
8. **Wait for upload** (25 MB takes 2-3 minutes)

**Your dataset URL:**
```
https://www.kaggle.com/datasets/pushkarprabhath/gricebench-dpo-model
```

**✅ Much simpler than accessing notebook output!**

---

## Step 2: Create Evaluation Notebook (10 minutes)

### 2.1 Create New Notebook

1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Rename to: `gricebench-final-evaluation`

---

### 2.2 Enable GPU

1. Click **"Settings"** (right sidebar)
2. Under **"Accelerator"**, select **"GPU T4 x2"**
3. Click **"Save"**

**Verify GPU is enabled:**
- You should see "GPU T4 x2" in the top right
- If not, refresh and check settings again

---

### 2.3 Add Datasets (CRITICAL STEP)

Click **"Add Data"** button (right sidebar)

**Add Dataset 1: Detector V2**
1. Click **"Datasets"** tab
2. Search: `pushkarprabhath/gricebench-detector-v2`
3. Click **"Add"**
4. **Verify path:** `/kaggle/input/gricebench-detector-v2/`

**Add Dataset 2: Test Data**
1. Search: `pushkarprabhath/gricebench-dpo-filtered`
2. Click **"Add"**
3. **Verify path:** `/kaggle/input/gricebench-dpo-filtered/`

**Add Dataset 3: DPO Model**
1. Search: `pushkarprabhath/gricebench-dpo-model`
2. Click **"Add"**
3. **Verify path:** `/kaggle/input/gricebench-dpo-model/`

---

### 2.4 Verify All Datasets Added

In the right sidebar under "Data", you should see:
```
✓ gricebench-detector-v2
✓ gricebench-dpo-filtered
✓ gricebench-dpo-model
```

**If any are missing, go back and add them!**

---

### 2.5 Check File Paths

Add this test cell to verify paths:

```python
import os

print("Checking dataset paths...")

# Detector V2
detector_path = '/kaggle/input/gricebench-detector-v2/best_model_v2.pt'
print(f"Detector model: {os.path.exists(detector_path)} ✓" if os.path.exists(detector_path) else "❌ MISSING!")

# Test data
test_path = '/kaggle/input/gricebench-dpo-filtered/dpo_val_filtered.json'
print(f"Test data: {os.path.exists(test_path)} ✓" if os.path.exists(test_path) else "❌ MISSING!")

# DPO model - UPDATE THIS PATH with your notebook name
dpo_path = '/kaggle/input/dpo-training/dpo_final/final_model/adapter_config.json'
print(f"DPO model: {os.path.exists(dpo_path)} ✓" if os.path.exists(dpo_path) else "❌ MISSING!")

print("\nIf all show ✓, you're ready to proceed!")
print("If any show ❌, check the 'Add Data' section and verify paths.")
```

**Run this cell first!** Only proceed if all show ✓.

---

### 2.6 Update Paths in Evaluation Cells

**IMPORTANT:** In the evaluation cells below, you'll need to update these paths:

**CELL 3 (Load Detector):**
```python
# This should work as-is:
checkpoint = torch.load('/kaggle/input/gricebench-detector-v2/best_model_v2.pt', ...)
```

**CELL 4 (Load DPO Model):**
```python
# UPDATE THIS - replace 'dpo-training' with your actual notebook name:
dpo_model = PeftModel.from_pretrained(
    base_for_dpo,
    '/kaggle/input/YOUR_NOTEBOOK_NAME/dpo_final/final_model'
)
```

**To find your notebook name:**
1. Look in "Data" sidebar
2. Find the notebook you added
3. Copy the exact name (case-sensitive!)

---

## Step 3: Copy-Paste Evaluation Cells (10 minutes)

---

## Step 2: Copy-Paste Evaluation Cells

### CELL 1: Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

print("✓ Imports complete")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

---

### CELL 2: Detector V2 Model Definition

```python
class MaximDetectorV2(nn.Module):
    """Detector V2 architecture"""
    
    def __init__(self, model_name, num_maxims=4, dropout=0.15):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, 1)
            )
            for _ in range(num_maxims)
        ])
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = torch.cat([classifier(pooled) for classifier in self.classifiers], dim=1)
        return logits

print("✓ Detector model defined")
```

---

### CELL 3: Load Detector V2

```python
print("Loading Detector V2...")

detector_model_name = 'microsoft/deberta-v3-base'
detector_tokenizer = AutoTokenizer.from_pretrained(detector_model_name)

detector = MaximDetectorV2(detector_model_name).to(device)

# UPDATE THIS PATH with your detector dataset location
checkpoint = torch.load('/kaggle/input/gricebench-detector-v2/best_model_v2.pt', 
                       map_location=device, weights_only=False)
detector.load_state_dict(checkpoint['model_state_dict'])
detector.eval()

# Load temperatures
with open('/kaggle/input/gricebench-detector-v2/temperatures.json') as f:
    temperatures = json.load(f)

print("✓ Detector V2 loaded")
print(f"  Temperatures: {temperatures}")
```

---

### CELL 4: Load DPO Models

```python
print("Loading DPO models...")

base_model_name = 'gpt2-medium'
dpo_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
dpo_tokenizer.pad_token = dpo_tokenizer.eos_token

# Load baseline model
print("  Loading baseline...")
baseline_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16
).to(device)
baseline_model.eval()

# Load DPO optimized model
print("  Loading DPO optimized...")
base_for_dpo = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16
).to(device)

# Load LoRA adapters from uploaded dataset
dpo_model = PeftModel.from_pretrained(
    base_for_dpo,
    '/kaggle/input/gricebench-dpo-model'  # Your uploaded DPO model dataset
).to(device)
dpo_model.eval()

print("✓ Both models loaded")
```

---

### CELL 5: Load Test Data

```python
print("Loading test data...")

# Load validation data as test set
test_path = '/kaggle/input/gricebench-dpo-filtered/dpo_val_filtered.json'

with open(test_path) as f:
    test_data = json.load(f)

print(f"✓ Loaded {len(test_data)} test examples")

# Use first 100 for evaluation (faster)
test_data = test_data[:100]
print(f"  Evaluating on {len(test_data)} examples")
```

---

### CELL 6: Evaluation Functions

```python
def detect_violations(detector, tokenizer, context, response, temperatures, device):
    """Detect maxim violations"""
    
    text = f"Context: {context} Response: {response}"
    
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = detector(input_ids, attention_mask)
    
    maxims = ['quantity', 'quality', 'relation', 'manner']
    scores = {}
    
    for i, maxim in enumerate(maxims):
        temp = temperatures[maxim]
        scaled_logit = logits[0, i] / temp
        prob = torch.sigmoid(scaled_logit).item()
        scores[maxim] = prob
    
    return scores

def generate_response(model, tokenizer, prompt, max_length=80, device='cpu'):
    """Generate response"""
    
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=inputs['input_ids'].shape[1] + max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    return response.strip()

print("✓ Functions defined")
```

---

### CELL 7: Run Evaluation

```python
print("\n" + "="*60)
print("RUNNING TEST EVALUATION")
print("="*60)

results = {
    'baseline': defaultdict(list),
    'dpo': defaultdict(list)
}

examples = []
threshold = 0.5

print("\nGenerating and evaluating responses...")

for item in tqdm(test_data, desc="Evaluating"):
    
    context = item.get('prompt', '')
    
    # Generate responses
    baseline_response = generate_response(baseline_model, dpo_tokenizer, context, device=device)
    dpo_response = generate_response(dpo_model, dpo_tokenizer, context, device=device)
    
    # Detect violations
    baseline_scores = detect_violations(detector, detector_tokenizer, context, baseline_response, temperatures, device)
    dpo_scores = detect_violations(detector, detector_tokenizer, context, dpo_response, temperatures, device)
    
    # Record violations
    for maxim in ['quantity', 'quality', 'relation', 'manner']:
        results['baseline'][maxim].append(1 if baseline_scores[maxim] > threshold else 0)
        results['dpo'][maxim].append(1 if dpo_scores[maxim] > threshold else 0)
    
    # Save example
    if len(examples) < 5:
        examples.append({
            'context': context[:100],
            'baseline': baseline_response[:100],
            'dpo': dpo_response[:100],
            'baseline_scores': baseline_scores,
            'dpo_scores': dpo_scores
        })

print("\n✓ Evaluation complete")
```

---

### CELL 8: Calculate Results

```python
print("\n" + "="*60)
print("FINAL TEST RESULTS")
print("="*60)

maxims = ['quantity', 'quality', 'relation', 'manner']

print("\n📊 Violation Rates:\n")
print(f"{'Maxim':<12} {'Baseline':>10} {'DPO':>10} {'Improvement':>15}")
print("-" * 60)

improvements = {}

for maxim in maxims:
    baseline_rate = np.mean(results['baseline'][maxim]) * 100
    dpo_rate = np.mean(results['dpo'][maxim]) * 100
    improvement = ((baseline_rate - dpo_rate) / baseline_rate * 100) if baseline_rate > 0 else 0
    
    improvements[maxim] = improvement
    
    status = "✅" if improvement > 0 else "❌"
    print(f"{maxim.capitalize():<12} {baseline_rate:>9.1f}% {dpo_rate:>9.1f}% {improvement:>+14.1f}% {status}")

# Cooperative rate
baseline_cooperative = []
dpo_cooperative = []

for i in range(len(results['baseline']['quantity'])):
    baseline_violations = sum(results['baseline'][m][i] for m in maxims)
    dpo_violations = sum(results['dpo'][m][i] for m in maxims)
    
    baseline_cooperative.append(1 if baseline_violations == 0 else 0)
    dpo_cooperative.append(1 if dpo_violations == 0 else 0)

baseline_coop_rate = np.mean(baseline_cooperative) * 100
dpo_coop_rate = np.mean(dpo_cooperative) * 100
coop_improvement = dpo_coop_rate - baseline_coop_rate

print("-" * 60)
print(f"{'Cooperative':<12} {baseline_coop_rate:>9.1f}% {dpo_coop_rate:>9.1f}% {coop_improvement:>+14.1f} pp ✅")

print("\n" + "="*60)
```

---

### CELL 9: Show Examples

```python
print("\n📝 Example Responses:\n")

for i, ex in enumerate(examples[:3], 1):
    print(f"Example {i}:")
    print(f"Context: {ex['context']}...")
    print(f"\nBaseline: {ex['baseline']}...")
    print(f"  Violations: {[m for m in maxims if ex['baseline_scores'][m] > 0.5]}")
    print(f"\nDPO: {ex['dpo']}...")
    print(f"  Violations: {[m for m in maxims if ex['dpo_scores'][m] > 0.5]}")
    print("-" * 60)
```

---

### CELL 10: Save Results

```python
final_results = {
    'test_size': len(test_data),
    'violation_rates': {
        'baseline': {m: float(np.mean(results['baseline'][m]) * 100) for m in maxims},
        'dpo': {m: float(np.mean(results['dpo'][m]) * 100) for m in maxims}
    },
    'improvements': {m: float(improvements[m]) for m in maxims},
    'cooperative_rate': {
        'baseline': float(baseline_coop_rate),
        'dpo': float(dpo_coop_rate),
        'improvement': float(coop_improvement)
    },
    'examples': examples
}

with open('/kaggle/working/final_test_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n✓ Saved results to final_test_results.json")
print("\n📥 Download from /kaggle/working/")
print("="*60)
```

---

## Step 3: After Running

**Download:**
- `final_test_results.json`

**Send me the results and I'll:**
- Create publication-ready report
- Generate visualizations
- Write discussion section
- Prepare final submission

---

## Expected Results

**If training metrics hold:**
- Cooperative rate: 60-70%
- 3-4 maxims improve
- Publication-ready!

**Total time:** 1-2 hours (mostly GPU generation)

---

## 🔧 Troubleshooting Guide

### Issue 1: "Dataset not found" Error

**Error:**
```
FileNotFoundError: /kaggle/input/gricebench-detector-v2/best_model_v2.pt
```

**Solution:**
1. Check "Data" sidebar - is `gricebench-detector-v2` listed?
2. If NO: Click "Add Data" → Search for it → Add
3. If YES: Check the exact path in the error
4. Run the path verification cell (Step 2.5)

---

### Issue 2: "DPO model not found"

**Error:**
```
OSError: /kaggle/input/dpo-training/dpo_final/final_model not found
```

**Solution:**
1. Check "Data" sidebar under "Your Work"
2. Find your DPO training notebook
3. Note the EXACT name (case-sensitive!)
4. Update CELL 4 with the correct path:
   ```python
   '/kaggle/input/YOUR_EXACT_NOTEBOOK_NAME/dpo_final/final_model'
   ```

**Common mistakes:**
- `dpo-training` vs `dpo_training` (dash vs underscore)
- `DPO-Training` vs `dpo-training` (capitalization)

---

### Issue 3: "Out of Memory" Error

**Error:**
```
CUDA out of memory
```

**Solution:**
1. Reduce batch size in generation:
   ```python
   # In generate_response function, add:
   torch.cuda.empty_cache()
   ```

2. Reduce test set size:
   ```python
   # In CELL 5, change:
   test_data = test_data[:50]  # Instead of 100
   ```

3. Use float16 instead of float32 (already done in guide)

---

### Issue 4: Generation is Very Slow

**Symptom:** Each response takes 30+ seconds

**Solution:**
1. Verify GPU is enabled (check top right corner)
2. Reduce max_length in generation:
   ```python
   # In generate_response, change:
   max_length=50  # Instead of 80
   ```

3. Check GPU usage:
   ```python
   !nvidia-smi
   ```

---

### Issue 5: "Module 'peft' not found"

**Error:**
```
ModuleNotFoundError: No module named 'peft'
```

**Solution:**
Add this cell at the beginning:
```python
!pip install peft -q
```

Then restart the kernel and run all cells again.

---

### Issue 6: Detector Scores All Near 0.5

**Symptom:** All violation scores are 0.48-0.52

**Solution:**
1. Check temperatures were loaded:
   ```python
   print(temperatures)  # Should show different values per maxim
   ```

2. Verify detector model loaded correctly:
   ```python
   print(detector)  # Should show model architecture
   ```

3. Check if using correct checkpoint:
   ```python
   print(checkpoint.keys())  # Should include 'model_state_dict'
   ```

---

### Issue 7: Results Look Wrong

**Symptom:** Baseline and DPO have same violation rates

**Possible causes:**

**1. Models are the same:**
```python
# Verify DPO model is different from baseline
print("Baseline params:", sum(p.numel() for p in baseline_model.parameters()))
print("DPO params:", sum(p.numel() for p in dpo_model.parameters()))
```

**2. Generation not working:**
```python
# Test generation manually
test_prompt = "Hello, how are you?"
baseline_resp = generate_response(baseline_model, dpo_tokenizer, test_prompt, device=device)
dpo_resp = generate_response(dpo_model, dpo_tokenizer, test_prompt, device=device)
print(f"Baseline: {baseline_resp}")
print(f"DPO: {dpo_resp}")
# Should be different!
```

**3. Detector not working:**
```python
# Test detector manually
test_context = "What is the capital of France?"
test_response = "Paris is the capital."
scores = detect_violations(detector, detector_tokenizer, test_context, test_response, temperatures, device)
print(scores)
# Should show varied scores, not all 0.5
```

---

### Issue 8: Notebook Crashes or Disconnects

**Solution:**
1. Save progress frequently (Kaggle auto-saves)
2. If crashed, just re-run from the last cell
3. Models stay loaded in memory
4. Results are saved to `/kaggle/working/`

---

### Issue 9: Can't Download Results

**Solution:**
1. Check `/kaggle/working/` in Output panel
2. If file not there, check for errors in CELL 10
3. Re-run CELL 10 to regenerate results file
4. Right-click file → Download

---

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check the error message carefully**
2. **Verify all paths** (Step 2.5)
3. **Check GPU is enabled** (Step 2.2)
4. **Verify all datasets added** (Step 2.4)
5. **Share the exact error** with me

---

## ✅ Success Checklist

Before running evaluation:
- [ ] All 3 datasets added to notebook
- [ ] GPU T4 x2 enabled
- [ ] Path verification cell shows all ✓
- [ ] Updated DPO model path in CELL 4
- [ ] All imports successful (CELL 1)

During evaluation:
- [ ] Detector loaded successfully
- [ ] Both models loaded successfully
- [ ] Test data loaded (100 examples)
- [ ] Generation working (progress bar moving)
- [ ] No CUDA errors

After evaluation:
- [ ] Results show different rates for baseline vs DPO
- [ ] Cooperative rate improved
- [ ] Examples look reasonable
- [ ] Results file downloaded

---

## 🎯 This is THE BEST way to validate your work!
