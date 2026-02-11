# 🚀 Multi-Stage DPO Training - Complete Kaggle Guide

## Overview

**Approach:** Two-stage DPO training
- **Stage 1:** Optimize content (Quantity + Relation)
- **Stage 2:** Optimize style (Manner)

**Total Time:** 2 hours  
**Expected Result:** 70-80% cooperative rate

---

## Prerequisites

**You need:**
1. Original scored DPO data (2,530 pairs with min_margin=0.05)
2. Detector V2 model
3. Test data

**Upload to Kaggle:**
- `gricebench-dpo-scored` (original scored data, NOT conflict-filtered)
- `gricebench-detector-v2`
- `gricebench-dpo-filtered` (for test set)

---

## PART 1: DATA PREPARATION (45 minutes)

### CELL 1: Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

print("✓ Imports complete")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

---

### CELL 2: Load Original Scored Data

```python
# ============================================
# LOAD ORIGINAL SCORED DATA
# ============================================

print("="*60)
print("LOADING ORIGINAL SCORED DATA")
print("="*60)

# IMPORTANT: Use ORIGINAL scored data (before conflict filtering)
# This has the strong signals we need!

# UPDATE THIS PATH with your original scored data location
with open('/kaggle/input/YOUR_ORIGINAL_SCORED_DATA/dpo_train_scored.json') as f:
    original_data = json.load(f)

print(f"\nOriginal data: {len(original_data)} pairs")

# Calculate margins for filtering
for item in original_data:
    chosen = item['chosen_scores']
    rejected = item['rejected_scores']
    
    item['quantity_margin'] = rejected['quantity'] - chosen['quantity']
    item['quality_margin'] = rejected['quality'] - chosen['quality']
    item['relation_margin'] = rejected['relation'] - chosen['relation']
    item['manner_margin'] = rejected['manner'] - chosen['manner']

print("✓ Margins calculated")
```

---

### CELL 3: Create Content Dataset (Stage 1)

```python
# ============================================
# STAGE 1 DATA: CONTENT-FOCUSED
# ============================================

print("\n" + "="*60)
print("CREATING CONTENT-FOCUSED DATASET")
print("="*60)

content_pairs = []

for item in original_data:
    # Keep if EITHER Quantity OR Relation has STRONG signal
    strong_quantity = abs(item['quantity_margin']) > 0.3
    strong_relation = abs(item['relation_margin']) > 0.3
    
    # And overall content signal is positive
    content_signal = item['quantity_margin'] + item['relation_margin']
    
    if (strong_quantity or strong_relation) and content_signal > 0.1:
        content_pairs.append(item)

print(f"\nContent-focused pairs: {len(content_pairs)}")
print(f"Retention: {len(content_pairs)/len(original_data)*100:.1f}%")

# Verify signal strength
margins = {
    'quantity': [item['quantity_margin'] for item in content_pairs],
    'relation': [item['relation_margin'] for item in content_pairs],
    'manner': [item['manner_margin'] for item in content_pairs]
}

print("\nContent Dataset Margins:")
for maxim in ['quantity', 'relation', 'manner']:
    mean = np.mean(margins[maxim])
    std = np.std(margins[maxim])
    print(f"  {maxim.capitalize():10s}: {mean:+.3f} (std: {std:.3f})")

print(f"\n✅ Expected: Quantity ~+0.4, Relation ~+0.3")
print(f"   (Manner will be negative - we ignore it in Stage 1)")

# Save
output_dir = Path('/kaggle/working/multistage_data')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'stage1_content_train.json', 'w') as f:
    json.dump(content_pairs, f, indent=2)

print(f"\n✓ Saved: {len(content_pairs)} pairs")
```

---

### CELL 4: Create Manner Dataset (Stage 2)

```python
# ============================================
# STAGE 2 DATA: MANNER-FOCUSED
# ============================================

print("\n" + "="*60)
print("CREATING MANNER-FOCUSED DATASET")
print("="*60)

manner_pairs = []

for item in original_data:
    # STRONG Manner signal
    strong_manner = item['manner_margin'] > 0.2
    
    # Content not terrible
    content_ok = (
        item['relation_margin'] > -0.1 and
        item['quantity_margin'] > -0.1
    )
    
    if strong_manner and content_ok:
        manner_pairs.append(item)

print(f"\nManner-focused pairs: {len(manner_pairs)}")
print(f"Retention: {len(manner_pairs)/len(original_data)*100:.1f}%")

# Verify signal strength
margins = {
    'quantity': [item['quantity_margin'] for item in manner_pairs],
    'relation': [item['relation_margin'] for item in manner_pairs],
    'manner': [item['manner_margin'] for item in manner_pairs]
}

print("\nManner Dataset Margins:")
for maxim in ['quantity', 'relation', 'manner']:
    mean = np.mean(margins[maxim])
    std = np.std(margins[maxim])
    print(f"  {maxim.capitalize():10s}: {mean:+.3f} (std: {std:.3f})")

print(f"\n✅ Expected: Manner ~+0.28, Quantity/Relation neutral")

# Save
with open(output_dir / 'stage2_manner_train.json', 'w') as f:
    json.dump(manner_pairs, f, indent=2)

print(f"\n✓ Saved: {len(manner_pairs)} pairs")
```

---

## PART 2: STAGE 1 TRAINING - CONTENT (30 minutes)

### CELL 5: Stage 1 Configuration

```python
# ============================================
# STAGE 1: CONTENT OPTIMIZATION CONFIG
# ============================================

CONFIG_STAGE1 = {
    # Data
    'train_data': '/kaggle/working/multistage_data/stage1_content_train.json',
    
    # Model
    'base_model': 'gpt2-medium',
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    
    # Training
    'batch_size': 1,
    'gradient_accumulation': 4,
    'learning_rate': 5e-5,
    'num_epochs': 2,
    'max_length': 512,
    
    # DPO
    'beta': 0.05,
    
    # CRITICAL: Only optimize Quantity + Relation
    'maxim_weights': {
        'quantity': 1.0,   # OPTIMIZE
        'quality': 0.0,    # IGNORE
        'relation': 1.0,   # OPTIMIZE
        'manner': 0.0      # IGNORE (fix in Stage 2)
    },
    
    'output_dir': '/kaggle/working/stage1_content_model',
    'device': device
}

print("="*60)
print("STAGE 1: CONTENT OPTIMIZATION")
print("="*60)
print(f"\nFocus: Quantity + Relation")
print(f"Ignoring: Quality + Manner")
print(f"Epochs: {CONFIG_STAGE1['num_epochs']}")
```

---

### CELL 6-10: Training Components

**Use the same cells from previous DPO training guide:**
- CELL 6: Detector V2 Model Definition
- CELL 7: Multi-Objective DPO Loss
- CELL 8: Dataset Class
- CELL 9: Helper Functions
- CELL 10: Training Loop

**But use `CONFIG_STAGE1` instead of `CONFIG`**

---

### CELL 11: Run Stage 1 Training

```python
# ============================================
# RUN STAGE 1 TRAINING
# ============================================

print("\n" + "="*60)
print("STARTING STAGE 1 TRAINING")
print("="*60)

# Load data
train_dataset = DPODataset(
    CONFIG_STAGE1['train_data'],
    dpo_tokenizer,
    CONFIG_STAGE1['max_length']
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG_STAGE1['batch_size'],
    shuffle=True,
    collate_fn=collate_fn
)

print(f"Training pairs: {len(train_dataset)}")

# Setup training
criterion = MultiObjectiveDPOLoss(CONFIG_STAGE1['maxim_weights'])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG_STAGE1['learning_rate'])
beta_scheduler = AdaptiveBetaScheduler(CONFIG_STAGE1['beta'], CONFIG_STAGE1['beta'], 100)

# Train
history_stage1 = {'train_loss': [], 'maxim_losses': []}

for epoch in range(CONFIG_STAGE1['num_epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG_STAGE1['num_epochs']}")
    
    train_loss, maxim_losses = train_epoch(
        model, reference_model, train_loader, criterion,
        optimizer, lr_scheduler, beta_scheduler, CONFIG_STAGE1['device']
    )
    
    history_stage1['train_loss'].append(train_loss)
    history_stage1['maxim_losses'].append(maxim_losses)
    
    print(f"Loss: {train_loss:.4f}")
    for m, l in maxim_losses.items():
        if CONFIG_STAGE1['maxim_weights'][m] > 0:
            print(f"  {m}: {l:.4f}")

# Save Stage 1 model
output_dir = Path(CONFIG_STAGE1['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(output_dir / 'final_model')
dpo_tokenizer.save_pretrained(output_dir / 'final_model')

print("\n✅ STAGE 1 COMPLETE!")
print(f"Saved to: {output_dir}/final_model")
```

---

## PART 3: STAGE 2 TRAINING - MANNER (30 minutes)

### CELL 12: Stage 2 Configuration

```python
# ============================================
# STAGE 2: MANNER OPTIMIZATION CONFIG
# ============================================

CONFIG_STAGE2 = {
    # Data
    'train_data': '/kaggle/working/multistage_data/stage2_manner_train.json',
    
    # Model - START FROM STAGE 1
    'stage1_model': '/kaggle/working/stage1_content_model/final_model',
    'base_model': 'gpt2-medium',
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    
    # Training
    'batch_size': 1,
    'gradient_accumulation': 4,
    'learning_rate': 3e-5,  # LOWER LR
    'num_epochs': 2,
    'max_length': 512,
    
    # DPO
    'beta': 0.03,  # LOWER BETA
    
    # CRITICAL: Only optimize Manner
    'maxim_weights': {
        'quantity': 0.0,   # IGNORE (already good)
        'quality': 0.0,    # IGNORE
        'relation': 0.0,   # IGNORE (already good)
        'manner': 1.0      # OPTIMIZE
    },
    
    'output_dir': '/kaggle/working/stage2_final_model',
    'device': device
}

print("="*60)
print("STAGE 2: MANNER OPTIMIZATION")
print("="*60)
print(f"\nFocus: Manner only")
print(f"Starting from: Stage 1 checkpoint")
print(f"Lower LR: {CONFIG_STAGE2['learning_rate']}")
print(f"Lower Beta: {CONFIG_STAGE2['beta']}")
```

---

### CELL 13: Load Stage 1 Model

```python
# ============================================
# LOAD STAGE 1 MODEL AS STARTING POINT
# ============================================

print("\nLoading Stage 1 model...")

# Load base model
stage1_base = AutoModelForCausalLM.from_pretrained(
    CONFIG_STAGE2['base_model'],
    torch_dtype=torch.float16
).to(device)

# Load Stage 1 LoRA adapters
stage1_model = PeftModel.from_pretrained(
    stage1_base,
    CONFIG_STAGE2['stage1_model']
).to(device)

print("✅ Stage 1 model loaded")

# Now create NEW LoRA adapters for Stage 2
# We'll train these on top of Stage 1
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=CONFIG_STAGE2['lora_r'],
    lora_alpha=CONFIG_STAGE2['lora_alpha'],
    lora_dropout=CONFIG_STAGE2['lora_dropout'],
    target_modules=['c_attn', 'c_proj', 'c_fc'],
    bias="none"
)

# Merge Stage 1 adapters and create new ones for Stage 2
stage1_merged = stage1_model.merge_and_unload()
stage2_model = get_peft_model(stage1_merged, lora_config)

print("✅ Stage 2 model initialized from Stage 1")
```

---

### CELL 14: Run Stage 2 Training

```python
# ============================================
# RUN STAGE 2 TRAINING
# ============================================

print("\n" + "="*60)
print("STARTING STAGE 2 TRAINING")
print("="*60)

# Load data
train_dataset_s2 = DPODataset(
    CONFIG_STAGE2['train_data'],
    dpo_tokenizer,
    CONFIG_STAGE2['max_length']
)

train_loader_s2 = DataLoader(
    train_dataset_s2,
    batch_size=CONFIG_STAGE2['batch_size'],
    shuffle=True,
    collate_fn=collate_fn
)

print(f"Training pairs: {len(train_dataset_s2)}")

# Setup training
criterion_s2 = MultiObjectiveDPOLoss(CONFIG_STAGE2['maxim_weights'])
optimizer_s2 = torch.optim.AdamW(stage2_model.parameters(), lr=CONFIG_STAGE2['learning_rate'])
beta_scheduler_s2 = AdaptiveBetaScheduler(CONFIG_STAGE2['beta'], CONFIG_STAGE2['beta'], 100)

# Use Stage 1 as reference model
reference_model_s2 = stage1_merged.eval()

# Train
history_stage2 = {'train_loss': [], 'maxim_losses': []}

for epoch in range(CONFIG_STAGE2['num_epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG_STAGE2['num_epochs']}")
    
    train_loss, maxim_losses = train_epoch(
        stage2_model, reference_model_s2, train_loader_s2, criterion_s2,
        optimizer_s2, lr_scheduler, beta_scheduler_s2, CONFIG_STAGE2['device']
    )
    
    history_stage2['train_loss'].append(train_loss)
    history_stage2['maxim_losses'].append(maxim_losses)
    
    print(f"Loss: {train_loss:.4f}")
    for m, l in maxim_losses.items():
        if CONFIG_STAGE2['maxim_weights'][m] > 0:
            print(f"  {m}: {l:.4f}")

# Save Stage 2 (final) model
output_dir = Path(CONFIG_STAGE2['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

stage2_model.save_pretrained(output_dir / 'final_model')
dpo_tokenizer.save_pretrained(output_dir / 'final_model')

print("\n✅ STAGE 2 COMPLETE!")
print(f"Saved to: {output_dir}/final_model")
```

---

## PART 4: EVALUATION (15 minutes)

### CELL 15: Load All Models for Comparison

```python
# ============================================
# LOAD ALL MODELS FOR COMPARISON
# ============================================

print("="*60)
print("LOADING MODELS FOR EVALUATION")
print("="*60)

# Baseline
print("\n1. Loading baseline...")
baseline_model = AutoModelForCausalLM.from_pretrained(
    'gpt2-medium',
    torch_dtype=torch.float16
).to(device)
baseline_model.eval()

# Stage 1
print("2. Loading Stage 1 model...")
stage1_eval = AutoModelForCausalLM.from_pretrained(
    'gpt2-medium',
    torch_dtype=torch.float16
).to(device)
stage1_eval = PeftModel.from_pretrained(
    stage1_eval,
    CONFIG_STAGE1['output_dir'] + '/final_model'
).to(device)
stage1_eval.eval()

# Stage 2 (final)
print("3. Loading Stage 2 (final) model...")
stage2_eval = AutoModelForCausalLM.from_pretrained(
    'gpt2-medium',
    torch_dtype=torch.float16
).to(device)
stage2_eval = PeftModel.from_pretrained(
    stage2_eval,
    CONFIG_STAGE2['output_dir'] + '/final_model'
).to(device)
stage2_eval.eval()

print("\n✅ All models loaded")
```

---

### CELL 16: Run Comparative Evaluation

```python
# ============================================
# COMPARATIVE EVALUATION
# ============================================

print("\n" + "="*60)
print("RUNNING COMPARATIVE EVALUATION")
print("="*60)

# Load test data
with open('/kaggle/input/gricebench-dpo-filtered/dpo_val_filtered.json') as f:
    test_data = json.load(f)

test_data = test_data[:100]  # Use 100 examples

print(f"\nTest set: {len(test_data)} examples")

# Evaluate all three models
models = {
    'baseline': baseline_model,
    'stage1': stage1_eval,
    'stage2_final': stage2_eval
}

results = {}

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    
    model_results = {
        'baseline': defaultdict(list),
        'dpo': defaultdict(list)
    }
    
    for item in tqdm(test_data, desc=f"{model_name}"):
        context = item.get('prompt', '')
        
        # Generate response
        response = generate_response(model, dpo_tokenizer, context, device=device)
        
        # Detect violations
        scores = detect_violations(detector, detector_tokenizer, context, response, temperatures, device)
        
        # Record
        for maxim in ['quantity', 'quality', 'relation', 'manner']:
            model_results['dpo'][maxim].append(1 if scores[maxim] > 0.5 else 0)
    
    results[model_name] = model_results

print("\n✅ Evaluation complete")
```

---

### CELL 17: Display Results

```python
# ============================================
# RESULTS COMPARISON
# ============================================

print("\n" + "="*60)
print("MULTI-STAGE DPO RESULTS")
print("="*60)

maxims = ['quantity', 'quality', 'relation', 'manner']

print(f"\n{'Maxim':<12} {'Baseline':>10} {'Stage 1':>10} {'Stage 2':>10} {'Improvement':>15}")
print("-" * 70)

for maxim in maxims:
    baseline_rate = np.mean(results['baseline']['dpo'][maxim]) * 100
    stage1_rate = np.mean(results['stage1']['dpo'][maxim]) * 100
    stage2_rate = np.mean(results['stage2_final']['dpo'][maxim]) * 100
    improvement = ((baseline_rate - stage2_rate) / baseline_rate * 100) if baseline_rate > 0 else 0
    
    status = "✅" if improvement > 0 else "❌"
    print(f"{maxim.capitalize():<12} {baseline_rate:>9.1f}% {stage1_rate:>9.1f}% {stage2_rate:>9.1f}% {improvement:>+14.1f}% {status}")

# Cooperative rate
def calc_cooperative(results_dict):
    coop = []
    for i in range(len(results_dict['dpo']['quantity'])):
        violations = sum(results_dict['dpo'][m][i] for m in maxims)
        coop.append(1 if violations == 0 else 0)
    return np.mean(coop) * 100

baseline_coop = calc_cooperative(results['baseline'])
stage1_coop = calc_cooperative(results['stage1'])
stage2_coop = calc_cooperative(results['stage2_final'])
coop_improvement = stage2_coop - baseline_coop

print("-" * 70)
print(f"{'Cooperative':<12} {baseline_coop:>9.1f}% {stage1_coop:>9.1f}% {stage2_coop:>9.1f}% {coop_improvement:>+14.1f} pp ✅")

print("\n" + "="*60)
print("✅ MULTI-STAGE DPO TRAINING COMPLETE!")
print("="*60)

# Save results
final_results = {
    'baseline': {m: float(np.mean(results['baseline']['dpo'][m]) * 100) for m in maxims},
    'stage1': {m: float(np.mean(results['stage1']['dpo'][m]) * 100) for m in maxims},
    'stage2_final': {m: float(np.mean(results['stage2_final']['dpo'][m]) * 100) for m in maxims},
    'cooperative': {
        'baseline': float(baseline_coop),
        'stage1': float(stage1_coop),
        'stage2_final': float(stage2_coop)
    }
}

with open('/kaggle/working/multistage_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n📥 Download multistage_results.json from /kaggle/working/")
```

---

## Expected Results

**Stage 1 (Content):**
- Quantity: Significant improvement
- Relation: Significant improvement
- Manner: Slight degradation (expected)

**Stage 2 (Final):**
- Quantity: Maintained
- Relation: Maintained
- Manner: Significant improvement
- **Cooperative: 70-80%** ✅

---

## Success Criteria

✅ All 4 maxims improve over baseline
✅ Cooperative rate 65-75% (minimum 60%)
✅ No maxim degrades >5% from baseline
✅ Top-tier publishable results

**Total time: 2 hours** 🚀
