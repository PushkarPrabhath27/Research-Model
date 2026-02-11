# 🚀 KAGGLE GUIDE: DPO Data Scoring & Filtering (Copy-Paste Ready)

## 📋 Overview

This guide scores DPO data with Detector V2 and filters by margin quality - **all on Kaggle GPU** (fast!).

**Time Required:** 30-45 minutes  
**GPU Required:** GPU T4 x2  
**What it does:** Scores 4,562 DPO pairs and filters to ~3,000 high-quality pairs

---

## Step 1: Upload Files to Kaggle (10 minutes)

### 1.1 Upload Detector V2 Model
1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload these 3 files:
   - `best_model_v2.pt` (from your local folder)
   - `temperatures.json`
   - `history_v2.json`
4. **Title:** `gricebench-detector-v2`
5. Click **"Create"**

### 1.2 Upload DPO Data
1. Click **"New Dataset"** again
2. Upload these files from `data_processed/dpo_data/`:
   - `dpo_train.json`
   - `dpo_val.json`
3. **Title:** `gricebench-dpo-raw`
4. Click **"Create"**

---

## Step 2: Create Kaggle Notebook (5 minutes)

1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Rename to: `dpo-scoring-and-filtering`
4. **Enable GPU:** Settings → Accelerator → **GPU T4 x2**
5. **Add Datasets:**
   - `gricebench-detector-v2`
   - `gricebench-dpo-raw`

---

## Step 3: Copy-Paste Cells (15 minutes)

### CELL 1: Setup and Imports

```python
# ============================================
# CELL 1: Setup and Imports
# ============================================

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("✓ Imports complete")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

**Run this cell**

---

### CELL 2: Configuration

```python
# ============================================
# CELL 2: Configuration
# ============================================

CONFIG = {
    # Detector V2 paths - Configured for: pushkarprabhath/gricebench-detector-v2
    'model_checkpoint': '/kaggle/input/gricebench-detector-v2/best_model_v2.pt',
    'temperatures': '/kaggle/input/gricebench-detector-v2/temperatures.json',
    
    # DPO data paths - Configured for: pushkarprabhath/gricebench-dpo-raw
    'dpo_train': '/kaggle/input/gricebench-dpo-raw/dpo_train.json',
    'dpo_val': '/kaggle/input/gricebench-dpo-raw/dpo_val.json',
    
    # Model
    'model_name': 'microsoft/deberta-v3-base',
    'max_length': 512,
    
    # Filtering
    'min_margin': 0.15,  # Keep pairs with margin > 0.15
    
    # Output
    'output_dir': '/kaggle/working/dpo_filtered',
    'device': device
}

print("Configuration:")
for key, val in CONFIG.items():
    if key != 'device':
        print(f"  {key}: {val}")
```

**✅ Already configured for your datasets!** Just copy-paste as-is.

**Run this cell**

---

### CELL 3: Model Architecture

```python
# ============================================
# CELL 3: Model Architecture (Same as Training)
# ============================================

class MaximDetectorV2(nn.Module):
    """Improved detector with deeper classification heads"""
    
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
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = torch.cat([
            classifier(pooled)
            for classifier in self.classifiers
        ], dim=1)
        return logits

print("✓ Model architecture defined")
```

**Run this cell**

---

### CELL 4: Load Model and Tokenizer

```python
# ============================================
# CELL 4: Load Model and Tokenizer
# ============================================

print("Loading Detector V2...")

tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
model = MaximDetectorV2(CONFIG['model_name']).to(CONFIG['device'])

# Load trained weights
checkpoint = torch.load(CONFIG['model_checkpoint'], map_location=CONFIG['device'], weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded")

# Load temperature scaling
with open(CONFIG['temperatures']) as f:
    temperatures = json.load(f)

print(f"✓ Temperatures loaded: {temperatures}")
```

**Run this cell** (Takes 2-3 minutes to download model)

---

### CELL 5: Scoring Function

```python
# ============================================
# CELL 5: Scoring Function
# ============================================

def score_response(context, response, evidence=None):
    """Score a response for maxim violations"""
    
    # Construct input text
    if evidence:
        text = f"Context: {context} Evidence: {evidence} Response: {response}"
    else:
        text = f"Context: {context} Response: {response}"
    
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=CONFIG['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(CONFIG['device'])
    attention_mask = encoding['attention_mask'].to(CONFIG['device'])
    
    # Get logits
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    # Apply temperature scaling and sigmoid
    maxims = ['quantity', 'quality', 'relation', 'manner']
    scores = {}
    
    for i, maxim in enumerate(maxims):
        temp = temperatures[maxim]
        scaled_logit = logits[0, i] / temp
        prob = torch.sigmoid(scaled_logit).item()
        scores[maxim] = prob
    
    return scores

print("✓ Scoring function defined")
```

**Run this cell**

---

### CELL 6: Score DPO Training Data

```python
# ============================================
# CELL 6: Score DPO Training Data
# ============================================

print("\n" + "="*60)
print("SCORING DPO TRAINING DATA")
print("="*60)

# Load DPO training data
with open(CONFIG['dpo_train']) as f:
    dpo_train = json.load(f)

print(f"\nLoaded {len(dpo_train)} training pairs")

# Score each pair
scored_data = []

for item in tqdm(dpo_train, desc="Scoring training pairs"):
    # Extract fields
    prompt = item.get('prompt', item.get('context', ''))
    chosen = item.get('chosen', item.get('chosen_response', ''))
    rejected = item.get('rejected', item.get('rejected_response', ''))
    
    # Score chosen response
    chosen_scores = score_response(prompt, chosen)
    
    # Score rejected response
    rejected_scores = score_response(prompt, rejected)
    
    # Add scores to item
    scored_item = item.copy()
    scored_item['chosen_scores'] = chosen_scores
    scored_item['rejected_scores'] = rejected_scores
    
    # Calculate margins
    margins = {
        maxim: rejected_scores[maxim] - chosen_scores[maxim]
        for maxim in ['quantity', 'quality', 'relation', 'manner']
    }
    scored_item['margins'] = margins
    scored_item['avg_margin'] = sum(margins.values()) / len(margins)
    
    scored_data.append(scored_item)

print(f"\n✓ Scored {len(scored_data)} pairs")
```

**Run this cell** (Takes ~10-15 minutes on GPU)

---

### CELL 7: Score DPO Validation Data

```python
# ============================================
# CELL 7: Score DPO Validation Data
# ============================================

print("\n" + "="*60)
print("SCORING DPO VALIDATION DATA")
print("="*60)

with open(CONFIG['dpo_val']) as f:
    dpo_val = json.load(f)

print(f"\nLoaded {len(dpo_val)} validation pairs")

scored_val = []

for item in tqdm(dpo_val, desc="Scoring validation pairs"):
    prompt = item.get('prompt', item.get('context', ''))
    chosen = item.get('chosen', item.get('chosen_response', ''))
    rejected = item.get('rejected', item.get('rejected_response', ''))
    
    chosen_scores = score_response(prompt, chosen)
    rejected_scores = score_response(prompt, rejected)
    
    scored_item = item.copy()
    scored_item['chosen_scores'] = chosen_scores
    scored_item['rejected_scores'] = rejected_scores
    
    margins = {
        maxim: rejected_scores[maxim] - chosen_scores[maxim]
        for maxim in ['quantity', 'quality', 'relation', 'manner']
    }
    scored_item['margins'] = margins
    scored_item['avg_margin'] = sum(margins.values()) / len(margins)
    
    scored_val.append(scored_item)

print(f"\n✓ Scored {len(scored_val)} validation pairs")
```

**Run this cell** (Takes ~3-5 minutes)

---

### CELL 8: Margin Statistics

```python
# ============================================
# CELL 8: Margin Statistics
# ============================================

print("\n" + "="*60)
print("MARGIN STATISTICS (Before Filtering)")
print("="*60)

margins_by_maxim = {m: [] for m in ['quantity', 'quality', 'relation', 'manner']}
avg_margins = []

for item in scored_data:
    for maxim, margin in item['margins'].items():
        margins_by_maxim[maxim].append(margin)
    avg_margins.append(item['avg_margin'])

print("\nMargin Statistics (rejected - chosen):")
print("Positive margin = chosen is better\n")

for maxim in ['quantity', 'quality', 'relation', 'manner']:
    margins = np.array(margins_by_maxim[maxim])
    print(f"{maxim.upper()}:")
    print(f"  Mean:   {margins.mean():.3f}")
    print(f"  Std:    {margins.std():.3f}")
    print(f"  >0.15:  {(margins > 0.15).mean()*100:.1f}%")
    print(f"  >0.20:  {(margins > 0.20).mean()*100:.1f}%")
    print()

avg_margins = np.array(avg_margins)
print("AVERAGE MARGIN:")
print(f"  Mean:   {avg_margins.mean():.3f}")
print(f"  >0.15:  {(avg_margins > 0.15).mean()*100:.1f}%")
print(f"  >0.20:  {(avg_margins > 0.20).mean()*100:.1f}%")
```

**Run this cell**

---

### CELL 9: Filter by Margin Quality

```python
# ============================================
# CELL 9: Filter by Margin Quality
# ============================================

print("\n" + "="*60)
print("FILTERING BY MARGIN QUALITY")
print("="*60)

min_margin = CONFIG['min_margin']
print(f"\nMinimum margin: {min_margin}")
print("(Keeping pairs where avg margin > 0.15)\n")

filtered_train = []
filtered_val = []

# Filter training data
for item in scored_data:
    if item['avg_margin'] > min_margin:
        filtered_train.append(item)

# Filter validation data
for item in scored_val:
    if item['avg_margin'] > min_margin:
        filtered_val.append(item)

print(f"Training pairs:")
print(f"  Original: {len(scored_data)}")
print(f"  Filtered: {len(filtered_train)}")
print(f"  Kept:     {len(filtered_train)/len(scored_data)*100:.1f}%")
print(f"  Removed:  {len(scored_data)-len(filtered_train)}")

print(f"\nValidation pairs:")
print(f"  Original: {len(scored_val)}")
print(f"  Filtered: {len(filtered_val)}")
print(f"  Kept:     {len(filtered_val)/len(scored_val)*100:.1f}%")

# Save filtered data
output_dir = Path(CONFIG['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'dpo_train_filtered.json', 'w') as f:
    json.dump(filtered_train, f, indent=2)

with open(output_dir / 'dpo_val_filtered.json', 'w') as f:
    json.dump(filtered_val, f, indent=2)

print(f"\n✓ Saved filtered data to {output_dir}")
```

**Run this cell**

---

### CELL 10: Final Statistics

```python
# ============================================
# CELL 10: Final Statistics
# ============================================

print("\n" + "="*60)
print("FILTERED DATA STATISTICS")
print("="*60)

# Calculate filtered margin stats
filtered_margins = {m: [] for m in ['quantity', 'quality', 'relation', 'manner']}
filtered_avg_margins = []

for item in filtered_train:
    for maxim, margin in item['margins'].items():
        filtered_margins[maxim].append(margin)
    filtered_avg_margins.append(item['avg_margin'])

print("\nFiltered Margin Statistics:\n")

for maxim in ['quantity', 'quality', 'relation', 'manner']:
    margins = np.array(filtered_margins[maxim])
    print(f"{maxim.upper()}:")
    print(f"  Mean:   {margins.mean():.3f}")
    print(f"  Std:    {margins.std():.3f}")
    print(f"  Min:    {margins.min():.3f}")
    print(f"  Max:    {margins.max():.3f}")
    print()

filtered_avg_margins = np.array(filtered_avg_margins)
print("AVERAGE MARGIN (Filtered):")
print(f"  Mean:   {filtered_avg_margins.mean():.3f}")
print(f"  Std:    {filtered_avg_margins.std():.3f}")
print(f"  Min:    {filtered_avg_margins.min():.3f}")
print(f"  Max:    {filtered_avg_margins.max():.3f}")

print("\n" + "="*60)
print("DPO SCORING & FILTERING COMPLETE!")
print("="*60)
print("\nGenerated files:")
print(f"  - dpo_train_filtered.json ({len(filtered_train)} pairs)")
print(f"  - dpo_val_filtered.json ({len(filtered_val)} pairs)")
print("\n📥 Download from /kaggle/working/dpo_filtered/")
print("="*60)
```

**Run this cell**

---

## Step 4: Download Filtered Data (5 minutes)

1. In Output panel, navigate to `/kaggle/working/dpo_filtered/`
2. Download:
   - `dpo_train_filtered.json`
   - `dpo_val_filtered.json`
3. Save to: `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\data_processed\dpo_data\`

---

## ✅ Success Criteria

- [ ] ~3,000-3,500 training pairs kept (65-75%)
- [ ] Average margin > 0.20
- [ ] All maxims have positive mean margins
- [ ] Files downloaded successfully

---

## 🎉 Next Steps

After downloading filtered data:
1. Upload to Kaggle as new dataset: `gricebench-dpo-filtered`
2. Follow `KAGGLE_DPO_OPTIMIZED_GUIDE.md` for DPO training
3. Train optimized DPO model (3-4 hours)
4. Achieve 70%+ cooperative rate!

**This approach is MUCH faster than local CPU scoring!** 🚀
