# 🚀 KAGGLE GUIDE: DPO Optimization Training (Copy-Paste Ready)

## 📋 Overview

This guide provides **exact copy-paste cells** for training the optimized DPO model with multi-objective loss and adaptive beta.

**Time Required:** 3-4 hours  
**GPU Required:** GPU T4 x2  
**Cost:** Free (Kaggle provides 30 hours/week)

**Prerequisites:** Detector V2 must be trained first!

---

## Step 1: Prepare DPO Data (I'll handle this locally)

**You don't need to do anything for this step - I'm handling it!**

I'll:
1. Load Detector V2
2. Score all DPO training/validation data
3. Filter pairs by margin quality
4. Save filtered data ready for upload

---

## Step 2: Upload DPO Data to Kaggle (10 minutes)

### 2.1 Go to Kaggle Datasets
1. Open browser: https://www.kaggle.com/datasets
2. Click **"New Dataset"**

### 2.2 Upload Files
I'll tell you which files to upload after I generate them locally.

Expected files:
- `dpo_train_filtered.json` (filtered preference pairs)
- `dpo_val.json` (validation data)

### 2.3 Configure Dataset
- **Title:** `gricebench-dpo-optimized`
- **Subtitle:** Filtered DPO training data with detector scores
- **Visibility:** Public
- Click **"Create"**

---

## Step 3: Create Kaggle Notebook (5 minutes)

### 3.1 Create New Notebook
1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Rename to: `dpo-optimization-training`

### 3.2 Enable GPU
1. Click **"Settings"** (right sidebar)
2. Under **"Accelerator"**, select **"GPU T4 x2"**
3. Click **"Save"**

### 3.3 Add Datasets
1. Click **"Add Data"**
2. Add: `gricebench-dpo-optimized` (your DPO data)
3. Add: `gpt2-medium` (search in Kaggle datasets)

---

## Step 4: Copy-Paste Cells (15 minutes)

### CELL 1: Setup and Imports
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 1: Setup and Imports
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
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
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 2: Configuration
# ============================================

CONFIG = {
    # Paths - UPDATE WITH YOUR DATASET IDS
    'base_model': 'gpt2-medium',
    'train_data_path': '/kaggle/input/gricebench-dpo-optimized/dpo_train_filtered.json',
    'val_data_path': '/kaggle/input/gricebench-dpo-optimized/dpo_val.json',
    
    # LoRA
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'lora_target_modules': ['c_attn', 'c_proj', 'c_fc'],
    
    # Training
    'batch_size': 1,
    'gradient_accumulation_steps': 4,
    'learning_rate': 5e-5,
    'num_epochs': 3,
    'max_length': 512,
    
    # DPO - Adaptive Beta
    'beta_start': 0.03,
    'beta_end': 0.08,
    'warmup_steps': 500,
    
    # Multi-objective weights (higher = more important)
    'maxim_weights': {
        'quantity': 1.0,
        'quality': 1.2,   # Higher weight (was degrading)
        'relation': 1.0,
        'manner': 1.2     # Higher weight (was degrading)
    },
    
    # Output
    'output_dir': '/kaggle/working/dpo_optimized',
    'device': device
}

print("Configuration:")
for key, val in CONFIG.items():
    if key != 'device':
        print(f"  {key}: {val}")
```

**⚠️ IMPORTANT:** Update paths in lines 8-9 with your dataset ID!

**Run this cell**

---

### CELL 3: Adaptive Beta Scheduler
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 3: Adaptive Beta Scheduler
# ============================================

class AdaptiveBetaScheduler:
    """Gradually increase beta during training"""
    
    def __init__(self, beta_start=0.03, beta_end=0.08, warmup_steps=500):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def get_beta(self):
        if self.current_step < self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        else:
            beta = self.beta_end
        return beta
    
    def step(self):
        self.current_step += 1
        return self.get_beta()

print("✓ Adaptive beta scheduler defined")
```

**Run this cell**

---

### CELL 4: Multi-Objective DPO Loss
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 4: Multi-Objective DPO Loss
# ============================================

class MultiObjectiveDPOLoss(nn.Module):
    """DPO loss that balances all four Gricean maxims"""
    
    def __init__(self, maxim_weights=None):
        super().__init__()
        self.maxims = ['quantity', 'quality', 'relation', 'manner']
        
        if maxim_weights is None:
            self.maxim_weights = {m: 1.0 for m in self.maxims}
        else:
            self.maxim_weights = maxim_weights
    
    def forward(self, policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                chosen_scores, rejected_scores, beta):
        
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        logits = beta * (policy_logratios - reference_logratios)
        
        maxim_losses = {}
        for maxim in self.maxims:
            chosen_score = chosen_scores.get(maxim, 0.5)
            rejected_score = rejected_scores.get(maxim, 0.5)
            margin = rejected_score - chosen_score
            
            loss = -F.logsigmoid(logits)
            maxim_losses[maxim] = loss * max(margin, 0.01)
        
        total_loss = sum(
            maxim_losses[m] * self.maxim_weights[m]
            for m in self.maxims
        )
        total_loss = total_loss / sum(self.maxim_weights.values())
        
        return total_loss.mean(), maxim_losses

print("✓ Multi-objective DPO loss defined")
```

**Run this cell**

---

### CELL 5: Dataset Class
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 5: Dataset Class
# ============================================

class DPODataset(Dataset):
    """Dataset for DPO training"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = item.get('prompt', item.get('context', ''))
        chosen = item.get('chosen', item.get('chosen_response', ''))
        rejected = item.get('rejected', item.get('rejected_response', ''))
        
        chosen_scores = item.get('chosen_scores', {})
        rejected_scores = item.get('rejected_scores', {})
        
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)['input_ids']
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)['input_ids']
        
        chosen_full = prompt_tokens + chosen_tokens
        rejected_full = prompt_tokens + rejected_tokens
        
        if len(chosen_full) > self.max_length:
            chosen_full = chosen_full[:self.max_length]
        if len(rejected_full) > self.max_length:
            rejected_full = rejected_full[:self.max_length]
        
        return {
            'chosen_input_ids': torch.tensor(chosen_full),
            'rejected_input_ids': torch.tensor(rejected_full),
            'prompt_length': len(prompt_tokens),
            'chosen_scores': chosen_scores,
            'rejected_scores': rejected_scores
        }

def collate_fn(batch):
    max_chosen_len = max(len(item['chosen_input_ids']) for item in batch)
    max_rejected_len = max(len(item['rejected_input_ids']) for item in batch)
    
    chosen_input_ids = []
    rejected_input_ids = []
    chosen_attention_mask = []
    rejected_attention_mask = []
    prompt_lengths = []
    chosen_scores_batch = []
    rejected_scores_batch = []
    
    for item in batch:
        chosen = item['chosen_input_ids']
        chosen_pad = max_chosen_len - len(chosen)
        chosen_input_ids.append(F.pad(chosen, (0, chosen_pad), value=50256))
        chosen_attention_mask.append(torch.cat([
            torch.ones(len(chosen)),
            torch.zeros(chosen_pad)
        ]))
        
        rejected = item['rejected_input_ids']
        rejected_pad = max_rejected_len - len(rejected)
        rejected_input_ids.append(F.pad(rejected, (0, rejected_pad), value=50256))
        rejected_attention_mask.append(torch.cat([
            torch.ones(len(rejected)),
            torch.zeros(rejected_pad)
        ]))
        
        prompt_lengths.append(item['prompt_length'])
        chosen_scores_batch.append(item['chosen_scores'])
        rejected_scores_batch.append(item['rejected_scores'])
    
    return {
        'chosen_input_ids': torch.stack(chosen_input_ids),
        'rejected_input_ids': torch.stack(rejected_input_ids),
        'chosen_attention_mask': torch.stack(chosen_attention_mask),
        'rejected_attention_mask': torch.stack(rejected_attention_mask),
        'prompt_lengths': prompt_lengths,
        'chosen_scores': chosen_scores_batch,
        'rejected_scores': rejected_scores_batch
    }

print("✓ Dataset class defined")
```

**Run this cell**

---

### CELL 6: Load Model and Data
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 6: Load Model and Data
# ============================================

print("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'])
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    CONFIG['base_model'],
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(CONFIG['device'])

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    lora_dropout=CONFIG['lora_dropout'],
    target_modules=CONFIG['lora_target_modules'],
    bias="none"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

reference_model = AutoModelForCausalLM.from_pretrained(
    CONFIG['base_model'],
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(CONFIG['device'])
reference_model.eval()

print("✓ Models loaded")

train_dataset = DPODataset(
    CONFIG['train_data_path'],
    tokenizer,
    CONFIG['max_length']
)

val_dataset = DPODataset(
    CONFIG['val_data_path'],
    tokenizer,
    CONFIG['max_length']
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    collate_fn=collate_fn
)

print(f"✓ Train examples: {len(train_dataset)}")
print(f"✓ Val examples: {len(val_dataset)}")
```

**Run this cell** (Takes 3-5 minutes to download GPT-2)

---

### CELL 7: Training Setup
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 7: Training Setup
# ============================================

criterion = MultiObjectiveDPOLoss(CONFIG['maxim_weights'])

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=0.01
)

beta_scheduler = AdaptiveBetaScheduler(
    beta_start=CONFIG['beta_start'],
    beta_end=CONFIG['beta_end'],
    warmup_steps=CONFIG['warmup_steps']
)

total_steps = len(train_loader) * CONFIG['num_epochs'] // CONFIG['gradient_accumulation_steps']
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)

print("✓ Training setup complete")
print(f"  Total steps: {total_steps}")
print(f"  Beta schedule: {CONFIG['beta_start']} → {CONFIG['beta_end']}")
```

**Run this cell**

---

### CELL 8: Helper Functions
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 8: Helper Functions
# ============================================

def get_log_probs(model, input_ids, attention_mask, prompt_length):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    mask = torch.zeros_like(token_log_probs)
    for i, plen in enumerate(prompt_length):
        mask[i, plen:] = 1
    
    response_log_probs = (token_log_probs * mask).sum(dim=1)
    return response_log_probs

print("✓ Helper functions defined")
```

**Run this cell**

---

### CELL 9: Training Loop
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 9: Training Loop
# ============================================

def train_epoch(model, reference_model, loader, criterion, optimizer, 
                lr_scheduler, beta_scheduler, device):
    model.train()
    reference_model.eval()
    
    total_loss = 0
    maxim_losses_sum = {m: 0 for m in ['quantity', 'quality', 'relation', 'manner']}
    
    progress_bar = tqdm(loader, desc="Training")
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        chosen_ids = batch['chosen_input_ids'].to(device)
        rejected_ids = batch['rejected_input_ids'].to(device)
        chosen_mask = batch['chosen_attention_mask'].to(device)
        rejected_mask = batch['rejected_attention_mask'].to(device)
        prompt_lengths = batch['prompt_lengths']
        
        beta = beta_scheduler.step()
        
        policy_chosen_logps = get_log_probs(model, chosen_ids, chosen_mask, prompt_lengths)
        policy_rejected_logps = get_log_probs(model, rejected_ids, rejected_mask, prompt_lengths)
        
        with torch.no_grad():
            reference_chosen_logps = get_log_probs(reference_model, chosen_ids, chosen_mask, prompt_lengths)
            reference_rejected_logps = get_log_probs(reference_model, rejected_ids, rejected_mask, prompt_lengths)
        
        loss, maxim_losses = criterion(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            batch['chosen_scores'][0],
            batch['rejected_scores'][0],
            beta
        )
        
        loss = loss / CONFIG['gradient_accumulation_steps']
        loss.backward()
        
        for maxim, mloss in maxim_losses.items():
            maxim_losses_sum[maxim] += mloss.item()
        
        if (step + 1) % CONFIG['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * CONFIG['gradient_accumulation_steps']
        progress_bar.set_postfix({
            'loss': total_loss / (step + 1),
            'beta': beta
        })
    
    for maxim in maxim_losses_sum:
        maxim_losses_sum[maxim] /= len(loader)
    
    return total_loss / len(loader), maxim_losses_sum

print("✓ Training loop defined")
```

**Run this cell**

---

### CELL 10: Run Training
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 10: Run Training
# ============================================

print("\n" + "="*60)
print("STARTING DPO OPTIMIZATION")
print("="*60)

history = {
    'train_loss': [],
    'maxim_losses': []
}

for epoch in range(CONFIG['num_epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
    print("-"*60)
    
    train_loss, maxim_losses = train_epoch(
        model, reference_model, train_loader, criterion,
        optimizer, lr_scheduler, beta_scheduler, CONFIG['device']
    )
    
    history['train_loss'].append(train_loss)
    history['maxim_losses'].append(maxim_losses)
    
    print(f"\nTrain Loss: {train_loss:.4f}")
    print("Per-maxim losses:")
    for maxim, loss in maxim_losses.items():
        print(f"  {maxim.capitalize():12s}: {loss:.4f}")
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / f'checkpoint_epoch{epoch+1}')
    print(f"✓ Saved checkpoint")

model.save_pretrained(CONFIG['output_dir'] + '/final_model')
tokenizer.save_pretrained(CONFIG['output_dir'] + '/final_model')

print("\n" + "="*60)
print("DPO OPTIMIZATION COMPLETE!")
print("="*60)

with open(CONFIG['output_dir'] + '/history_optimized.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"✓ Saved training history")
print("\nGenerated files:")
print(f"  - final_model/ (LoRA adapters)")
print(f"  - history_optimized.json")
print("\n📥 Download from /kaggle/working/dpo_optimized/")
print("="*60)
```

**Run this cell** (Takes 3-4 hours - perfect time for a meal! 🍕)

---

## Step 5: Download Results (10 minutes)

### 5.1 Download Files
1. In right sidebar, click **"Output"**
2. Navigate to `dpo_optimized/final_model/` folder
3. Download entire `final_model/` folder (contains LoRA adapters)
4. Download `history_optimized.json`

### 5.2 Save Locally
Place in:
```
c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\dpo_optimized\
```

---

## ✅ Success Criteria

- [ ] All maxim losses decreased
- [ ] Quality and Manner losses improved more
- [ ] Final loss < 0.4
- [ ] Training took 3-4 hours
- [ ] All files downloaded

---

## 🎉 You're Done!

After this, I'll handle the final evaluation locally and show you the results!

**Expected final metrics:**
- Cooperative rate: 70%+
- All 4 maxims improved
- Publication ready! 🎊
