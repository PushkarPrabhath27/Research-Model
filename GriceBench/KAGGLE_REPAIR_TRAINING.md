# 🔧 KAGGLE REPAIR MODEL TRAINING GUIDE
## T5-based Violation Repair | Chapter 10

**What you'll build:** A model that takes violated responses and automatically fixes them!

**Training time:** 4-8 hours on Kaggle GPU T4 x2

---

## STEP 1: Upload Data to Kaggle

### Create a New Dataset

1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload these 4 files from `data_processed/repair_data/`:
   - `repair_train.json` (4.3 MB)
   - `repair_val.json` (535 KB)
   - `repair_test.json` (530 KB)
   - `control_tokens.json` (545 bytes)

4. Name it: **"gricebench-repair-data"**
5. Click **"Create"**

---

## STEP 2: Create New Notebook

1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Settings → **Accelerator** → **GPU T4 x2**
4. Right sidebar → **+ Add data** → Search "gricebench-repair-data" → Add

---

## CELL 1: Setup & Install

```python
print("="*70)
print("🔧 GriceBench Repair Model Training")
print("="*70)

# Check GPU
import torch
if not torch.cuda.is_available():
    raise RuntimeError("❌ NO GPU! Enable GPU T4 x2 in Settings")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# Install packages (already available on Kaggle)
# transformers, torch, tqdm are pre-installed
print("\n✅ All packages ready!")
```

---

## CELL 2: Load Data & Setup

```python
import json
from pathlib import Path

# Data paths
DATA = Path("/kaggle/input/gricebench-repair-data")

# Load data
with open(DATA / "repair_train.json") as f:
    train_data = json.load(f)
with open(DATA / "repair_val.json") as f:
    val_data = json.load(f)
with open(DATA / "control_tokens.json") as f:
    control_tokens = json.load(f)

print(f"✅ Train: {len(train_data):,} examples")
print(f"✅ Val: {len(val_data):,} examples")
print(f"✅ Control tokens: {len(control_tokens['all_tokens'])}")

# Show example
print(f"\n📝 Example input:")
print(train_data[0]['input_text'][:200] + "...")
print(f"\n🎯 Target:")
print(train_data[0]['target_text'][:100] + "...")
```

---

## CELL 3: Model & Dataset

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm.auto import tqdm

DEVICE = torch.device('cuda')

# Configuration
CFG = {
    'model_name': 't5-base',
    'batch_size': 4,
    'lr': 3e-4,
    'epochs': 5,
    'max_input_len': 512,
    'max_target_len': 256,
    'warmup_ratio': 0.1
}

print("📝 Configuration:")
for k, v in CFG.items():
    print(f"   {k}: {v}")

# Dataset
class RepairDataset(Dataset):
    def __init__(self, data, tokenizer, max_input=512, max_target=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        
        # Input
        inp = self.tokenizer(
            ex['input_text'],
            max_length=self.max_input,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Target
        tgt = self.tokenizer(
            ex['target_text'],
            max_length=self.max_target,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Labels (-100 for padding)
        labels = tgt['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inp['input_ids'].squeeze(0),
            'attention_mask': inp['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

# Load model
print(f"\n🤖 Loading {CFG['model_name']}...")
tokenizer = T5Tokenizer.from_pretrained(CFG['model_name'])

# Add control tokens
print(f"   Adding {len(control_tokens['all_tokens'])} control tokens...")
tokenizer.add_tokens(control_tokens['all_tokens'])

model = T5ForConditionalGeneration.from_pretrained(CFG['model_name'])
model.resize_token_embeddings(len(tokenizer))
model.to(DEVICE)

params = sum(p.numel() for p in model.parameters())
print(f"✅ Model: {params:,} parameters")

# Create datasets
train_ds = RepairDataset(train_data, tokenizer, CFG['max_input_len'], CFG['max_target_len'])
val_ds = RepairDataset(val_data, tokenizer, CFG['max_input_len'], CFG['max_target_len'])

train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False)

print(f"✅ {len(train_loader)} train batches, {len(val_loader)} val batches")
```

---

## CELL 4: Training Loop (4-8 hours)

```python
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

print("="*70)
print("🏋️ TRAINING")
print("="*70)

# Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=CFG['lr'])
total_steps = len(train_loader) * CFG['epochs']
warmup_steps = int(total_steps * CFG['warmup_ratio'])
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"Total steps: {total_steps:,} | Warmup: {warmup_steps:,}\n")

# Training
best_val_loss = float('inf')
history = {'train_loss': [], 'val_loss': []}

for epoch in range(CFG['epochs']):
    print(f"\n{'#'*70}")
    print(f"EPOCH {epoch+1}/{CFG['epochs']}")
    print('#'*70)
    
    # Train
    model.train()
    train_loss = 0
    
    progress = tqdm(train_loader, desc=f"Training")
    for batch in progress:
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train = train_loss / len(train_loader)
    history['train_loss'].append(avg_train)
    
    # Validate
    print("\n🔍 Validating...")
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            val_loss += outputs.loss.item()
    
    avg_val = val_loss / len(val_loader)
    history['val_loss'].append(avg_val)
    
    print(f"\n📊 Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
    
    # Save best
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        model.save_pretrained('/kaggle/working/repair_model')
        tokenizer.save_pretrained('/kaggle/working/repair_model')
        print(f"💾 SAVED! Best val loss: {avg_val:.4f}")

print(f"\n{'='*70}")
print(f"✅ DONE! Best loss: {best_val_loss:.4f}")
print('='*70)

# Save history
with open('/kaggle/working/history.json', 'w') as f:
    json.dump(history, f, indent=2)
```

---

## CELL 5: Test Generation

```python
# Test the repair model
print("="*70)
print("🧪 TESTING REPAIR GENERATION")
print("="*70)

def repair_text(text, max_len=256):
    model.eval()
    inputs = tokenizer(text, max_length=512, truncation=True, return_tensors='pt').to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'], 
            max_length=max_len,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Get test examples
for i in range(min(3, len(val_data))):
    ex = val_data[i]
    print(f"\n{'─'*70}")
    print(f"Example {i+1}:")
    print(f"\nViolated: {ex['input_text'][200:400]}...")  # Show middle part
    
    # Generate repair
    repair = repair_text(ex['input_text'])
    print(f"\n🔧 Repair: {repair}")
    print(f"\n✅ Target: {ex['target_text']}")
    print('─'*70)

print("\n✅ Repair model is working!")
```

---

## CELL 6: Download

```python
import shutil

# Zip everything
shutil.make_archive('/kaggle/working/repair_final', 'zip', '/kaggle/working')

print("="*70)
print("📥 DOWNLOAD: repair_final.zip")
print("="*70)
print("Right sidebar → Output → repair_final.zip → ⋮ → Download")
print()
print("Contains:")
print("  - repair_model/ (T5 model + tokenizer)")
print("  - history.json (training metrics)")
print("="*70)
```

---

## ✅ SUCCESS CRITERIA

Your repair model should:
- ✅ Train loss < 0.5
- ✅ Val loss < 0.6
- ✅ Generate fluent, natural repairs
- ✅ Fix violations without changing meaning (except Quality)

---

## 🎯 After Download

1. Extract `repair_final.zip`
2. Move `repair_model/` to `GriceBench/models/repair/`
3. Move `history.json` to `GriceBench/models/repair/`
4. Run evaluation: `python scripts/evaluate_repair.py`

**Ready to train on Kaggle!** 🚀
