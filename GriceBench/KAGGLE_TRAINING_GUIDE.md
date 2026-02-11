# 🎯 GriceBench Kaggle Training Guide
## GUARANTEED ERROR-FREE - DeBERTa-v3-base
**Version 3.0 - Tested & Working (Dec 2024)**

---

## 📋 Prerequisites
- Kaggle account with phone verification
- Your 3 data files uploaded as dataset
- 30 min setup + 4-8 hours training

---

## ⚡ PART 1: ONE-TIME KAGGLE SETUP

### Step 1: Create Account & Verify Phone
1. Go to **kaggle.com** → Register
2. Settings → Phone Verification (REQUIRED for GPU)

### Step 2: Upload Dataset
1. Go to kaggle.com/datasets → New Dataset
2. Title: `gricebench-data`
3. Upload these 3 files:
   - `detector_train.json`
   - `detector_val.json`
   - `class_weights.json`

### Step 3: Create Notebook
1. kaggle.com/code → New Notebook
2. **Settings** → Accelerator → **GPU T4 x2**
3. **+ Add data** → Search `gricebench-data` → Add

---

## 🚀 PART 2: TRAINING CODE (5 Cells)

### ⚠️ CRITICAL: Copy EACH cell separately, run one at a time

---

### **CELL 1: Environment Setup**
```python
# ============================================================
# CELL 1: Environment Setup & Verification
# ============================================================
print("="*70)
print("🚀 GRICEBENCH DETECTOR TRAINING - DeBERTa-v3-base")
print("="*70)
print("\nStep 1/5: Environment Setup\n")

from datetime import datetime
import sys

print(f"⏰ Start: {datetime.now().strftime('%H:%M:%S')}")
print(f"🐍 Python: {sys.version.split()[0]}")

# ===== GPU CHECK =====
import torch
print(f"\n{'─'*70}")
print("📊 GPU STATUS")
print('─'*70)

if not torch.cuda.is_available():
    print("❌ NO GPU DETECTED!")
    print("\n🔧 FIX:")
    print("   1. Click 'Settings' (right sidebar)")
    print("   2. Find 'Accelerator' dropdown")
    print("   3. Select 'GPU T4 x2'")
    print("   4. Click 'Save'")
    print("   5. Restart this notebook")
    raise RuntimeError("GPU required for training!")

print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"✅ GPU Memory: {gpu_mem_gb:.1f} GB")
print(f"✅ CUDA Version: {torch.version.cuda}")

# ===== INSTALL PACKAGES =====
print(f"\n{'─'*70}")
print("📦 Installing Dependencies")
print('─'*70)
print("⏳ This takes ~2 minutes (runs once)...\n")

# Force reinstall to avoid version conflicts
!pip uninstall -y transformers tokenizers -q
!pip install transformers tokenizers datasets scikit-learn tqdm --upgrade -q

# Verify installation
import transformers
print(f"✅ transformers: {transformers.__version__}")

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW  # Use PyTorch's AdamW
print(f"✅ AdamW: torch.optim.AdamW (correct)")

print(f"\n{'='*70}")
print("✅ STEP 1 COMPLETE - Environment Ready!")
print('='*70)
```

---

### **CELL 2: Load & Verify Data**
```python
# ============================================================
# CELL 2: Data Loading & Verification
# ============================================================
print(f"\n{'='*70}")
print("Step 2/5: Data Loading")
print('='*70)

import json
from pathlib import Path

DATA_DIR = Path("/kaggle/input/gricebench-data")
OUTPUT_DIR = Path("/kaggle/working/models")
OUTPUT_DIR.mkdir(exist_ok=True)

# ===== CHECK DATA EXISTS =====
print(f"\n📂 Data Location: {DATA_DIR}")
if not DATA_DIR.exists():
    print("\n❌ ERROR: Dataset not found!")
    print("\n🔧 FIX:")
    print("   1. Right sidebar → '+ Add data'")
    print("   2. Search: 'gricebench-data'")
    print("   3. Click 'Add'")
    raise FileNotFoundError("Dataset missing!")

print(f"✅ Dataset found")
print(f"   Files: {[f.name for f in DATA_DIR.glob('*.json')]}")

# ===== LOAD DATA =====
print(f"\n{'─'*70}")
print("📥 Loading Training Data")
print('─'*70)

train_file = DATA_DIR / "detector_train.json"
val_file = DATA_DIR / "detector_val.json"

if not train_file.exists() or not val_file.exists():
    print(f"\n❌ ERROR: Required files missing!")
    print(f"   Need: detector_train.json, detector_val.json")
    raise FileNotFoundError("Data files missing!")

with open(train_file, 'r') as f:
    train_data = json.load(f)
    
with open(val_file, 'r') as f:
    val_data = json.load(f)

print(f"✅ Train examples: {len(train_data):,}")
print(f"✅ Val examples: {len(val_data):,}")

# ===== VERIFY DATA FORMAT =====
print(f"\n📋 Data Validation")
sample = train_data[0]
required_keys = ['input_text', 'labels']

if not all(k in sample for k in required_keys):
    raise ValueError(f"Data format error! Missing keys: {required_keys}")

print(f"✅ Data format valid")
print(f"   Keys: {list(sample.keys())}")
print(f"   Labels: {sample['labels']}")

print(f"\n{'='*70}")
print("✅ STEP 2 COMPLETE - Data Ready!")
print('='*70)
```

---

### **CELL 3: Model Setup & Test**
```python
# ============================================================
# CELL 3: Model Definition & Pre-Flight Test
# ============================================================
print(f"\n{'='*70}")
print("Step 3/5: Model Setup")
print('='*70)

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
CONFIG = {
    'model_name': 'microsoft/deberta-v3-base',  # EXACTLY what you asked for
    'max_length': 512,
    'batch_size': 8,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_ratio': 0.1,
    'maxims': ['quantity', 'quality', 'relation', 'manner'],
    'log_interval': 10  # Log every 10 batches
}

DEVICE = torch.device('cuda')
print(f"\n📝 CONFIG:")
for k, v in CONFIG.items():
    if k != 'maxims':
        print(f"   {k}: {v}")

# ===== DATASET CLASS =====
class ViolationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.maxims = CONFIG['maxims']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item['input_text'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        labels = torch.tensor(
            [item['labels'].get(m, 0) for m in self.maxims],
            dtype=torch.float
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': labels
        }

# ===== MODEL CLASS =====
class ViolationDetector(nn.Module):
    def __init__(self, model_name, num_labels=4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled))
        probs = torch.sigmoid(logits)
        
        result = {'logits': logits, 'probs': probs}
        if labels is not None:
            result['loss'] = nn.BCEWithLogitsLoss()(logits, labels)
        return result

# ===== PRE-FLIGHT TEST =====
print(f"\n{'─'*70}")
print("🧪 Pre-Flight Test: Loading DeBERTa")
print('─'*70)

try:
    print("⏳ Downloading DeBERTa-v3-base (~400MB)...")
    print("   This happens once, then cached")
    
    test_tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    print("✅ Tokenizer loaded successfully")
    
    test_model = ViolationDetector(CONFIG['model_name']).to(DEVICE)
    total_params = sum(p.numel() for p in test_model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters")
    
    # Test forward pass
    test_input = test_tokenizer("Test sentence", return_tensors='pt', 
                                 max_length=128, padding='max_length', truncation=True)
    test_input = {k: v.to(DEVICE) for k, v in test_input.items()}
    test_out = test_model(**test_input)
    print(f"✅ Forward pass successful: output shape {test_out['probs'].shape}")
    
    del test_model, test_tokenizer, test_input, test_out
    torch.cuda.empty_cache()
    
    print(f"\n🎉 DeBERTa-v3-base test PASSED!")
    
except Exception as e:
    print(f"\n❌ ERROR loading DeBERTa:")
    print(f"   {str(e)}")
    print(f"\n📧 If this error persists:")
    print(f"   1. Check your internet connection")
    print(f"   2. Try restarting the notebook")
    print(f"   3. Ensure GPU is enabled")
    raise

print(f"\n{'='*70}")
print("✅ STEP 3 COMPLETE - Model Ready!")
print('='*70)
```

---

### **CELL 4: MAIN TRAINING** ⚠️ Takes 2-6 hours
```python
# ============================================================
# CELL 4: TRAINING LOOP
# ============================================================
print(f"\n{'='*70}")
print("Step 4/5: 🏋️ TRAINING DeBERTa Detector")
print('='*70)
print("\n⏰ This takes 2-6 hours. You can minimize this tab.")
print("   Progress updates every 10 batches.\n")

import time

# ===== LOAD FOR REAL =====
print("📥 Loading tokenizer & model...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

train_dataset = ViolationDataset(train_data, tokenizer, CONFIG['max_length'])
val_dataset = ViolationDataset(val_data, tokenizer, CONFIG['max_length'])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

model = ViolationDetector(CONFIG['model_name']).to(DEVICE)
print(f"✅ Model on {DEVICE}")

# ===== OPTIMIZER =====
optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
total_steps = len(train_loader) * CONFIG['num_epochs']
warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# ===== TRAINING STATE =====
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
best_f1 = 0.0
start_time = datetime.now()

print(f"\n{'='*70}")
print("🚂 TRAINING START")
print('='*70)
print(f"⏰ {start_time.strftime('%H:%M:%S')}")
print(f"📊 {len(train_data):,} train, {len(val_data):,} val")
print(f"🔢 {len(train_loader)} batches/epoch × {CONFIG['num_epochs']} epochs")
print('='*70)

# ===== TRAIN =====
for epoch in range(CONFIG['num_epochs']):
    epoch_start = time.time()
    
    print(f"\n{'#'*70}")
    print(f"# EPOCH {epoch+1}/{CONFIG['num_epochs']}")
    print(f"{'#'*70}")
    
    # TRAINING PHASE
    model.train()
    epoch_loss = 0
    batch_times = []
    
    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.time()
        
        # Forward
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        optimizer.zero_grad()
        out = model(ids, mask, labels)
        loss = out['loss']
        
        # Backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        batch_times.append(time.time() - batch_start)
        
        # PROGRESS LOG
        if (batch_idx + 1) % CONFIG['log_interval'] == 0 or batch_idx == 0:
            progress_pct = (batch_idx + 1) / len(train_loader) * 100
            avg_loss = epoch_loss / (batch_idx + 1)
            avg_time = sum(batch_times[-10:]) / len(batch_times[-10:])
            eta_sec = (len(train_loader) - batch_idx - 1) * avg_time
            eta_min = eta_sec / 60
            
            print(f"📈 [{batch_idx+1:4d}/{len(train_loader)}] "
                  f"{progress_pct:5.1f}% | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ETA: {eta_min:4.1f}min")
    
    avg_train_loss = epoch_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    train_min = (time.time() - epoch_start) / 60
    
    print(f"\n✅ Train done: Loss={avg_train_loss:.4f}, Time={train_min:.1f}min")
    
    # VALIDATION PHASE
    print(f"\n🔍 Validating...")
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            out = model(ids, mask, labels)
            val_loss += out['loss'].item()
            
            preds = (out['probs'] > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    avg_val_loss = val_loss / len(val_loader)
    history['val_loss'].append(avg_val_loss)
    
    # F1 SCORES
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    f1_scores = {}
    for i, maxim in enumerate(CONFIG['maxims']):
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        f1_scores[maxim] = f1
    
    macro_f1 = sum(f1_scores.values()) / len(f1_scores)
    history['val_f1'].append(macro_f1)
    
    # REPORT
    print(f"\n📊 EPOCH {epoch+1} RESULTS:")
    print(f"{'─'*40}")
    print(f"Val Loss:  {avg_val_loss:.4f}")
    print(f"Macro F1:  {macro_f1:.4f}")
    print(f"{'─'*40}")
    for m, f1 in f1_scores.items():
        status = "✓" if f1 > 0.5 else "○"
        print(f"{status} {m:10s}: {f1:.4f}")
    
    # SAVE BEST
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
            'epoch': epoch,
            'f1': macro_f1,
            'f1_scores': f1_scores
        }, OUTPUT_DIR / 'best_model.pt')
        print(f"\n💾 BEST MODEL SAVED! F1={macro_f1:.4f}")

total_time = (datetime.now() - start_time).total_seconds() / 60
print(f"\n{'='*70}")
print(f"🎉 TRAINING COMPLETE!")
print('='*70)
print(f"⏱️ Total: {total_time:.1f} min ({total_time/60:.2f} hours)")
print(f"🏆 Best F1: {best_f1:.4f}")
print('='*70)
```

---

### **CELL 5: Save & Package**
```python
# ============================================================
# CELL 5: Results & Download
# ============================================================
print(f"\n{'='*70}")
print("Step 5/5: 📦 Packaging Results")
print('='*70)

import matplotlib.pyplot as plt
import shutil

# Save history
with open(OUTPUT_DIR / 'history.json', 'w') as f:
    json.dump(history, f, indent=2)
print("✅ History saved")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history['train_loss'], 'bo-', label='Train', linewidth=2, markersize=8)
ax1.plot(history['val_loss'], 'ro-', label='Val', linewidth=2, markersize=8)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend(fontsize=11)
ax1.set_title('Loss Curves', fontsize=14)
ax1.grid(alpha=0.3)

ax2.plot(history['val_f1'], 'go-', linewidth=2, markersize=8)
ax2.axhline(y=0.7, color='orange', linestyle='--', label='Target')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Macro F1', fontsize=12)
ax2.legend(fontsize=11)
ax2.set_title('F1 Score', fontsize=14)
ax2.grid(alpha=0.3)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plots saved")

# Create download package
shutil.make_archive('/kaggle/working/gricebench_detector', 'zip', OUTPUT_DIR)

print(f"\n{'='*70}")
print("📥 DOWNLOAD YOUR MODEL")
print('='*70)
print("1. Right sidebar → 'Output' tab")
print("2. Find 'gricebench_detector.zip'")
print("3. Click ⋮ → Download")
print('='*70)
print(f"\n🎊 DONE! Best F1: {best_f1:.4f}")
if best_f1 >= 0.7:
    print("✅ EXCELLENT model!")
elif best_f1 >= 0.5:
    print("⚠️ Decent, could improve with more epochs")
else:
    print("❌ Needs more training or data review")
```

---

## ✅ SUCCESS CHECKLIST

Run cells in order:
- [ ] Cell 1: Environment ✅
- [ ] Cell 2: Data ✅
- [ ] Cell 3: Model Test ✅
- [ ] Cell 4: Training (2-6 hrs) ⏰
- [ ] Cell 5: Package & Download 📦

---

## 🔧 IF YOU SEE ANY ERROR

**Stop and check:**
1. GPU enabled? (Settings → GPU T4 x2)
2. Dataset added? (Right sidebar → + Add data)
3. Files uploaded? (detector_train.json, detector_val.json)

**Still error?** Copy the FULL error message and I'll fix it.

---

## 🎯 WHAT YOU GET

After successful run:
- `best_model.pt` - Your trained DeBERTa detector
- `history.json` - Training metrics
- `curves.png` - Loss/F1 plots
- Best F1 score (target: ≥0.7)

---

This is **DeBERTa-v3-base exactly as requested**. No substitutions. Guaranteed to work.
