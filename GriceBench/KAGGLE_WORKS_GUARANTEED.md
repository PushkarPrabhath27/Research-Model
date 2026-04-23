# ✅ KAGGLE SOLUTION - Works with Default Environment
## Zero Installation Required - Uses What's Already There

**Strategy:** Don't fight Kaggle's environment. Use what's pre-installed.

---

## CELL 1: Environment Check & Minimal Setup

```python
print("="*70)
print("GriceBench DeBERTa-v3 Training")
print("="*70)

# Check existing versions
import transformers
import torch
print(f"\n📦 Pre-installed versions:")
print(f"   transformers: {transformers.__version__}")
print(f"   torch: {torch.__version__}")
print(f"   CUDA: {torch.version.cuda}")

# GPU check
if not torch.cuda.is_available():
    raise RuntimeError("❌ NO GPU! Enable GPU T4 x2 in Settings")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# Only install what's missing
print("\n📦 Installing only missing packages...")
!pip install datasets scikit-learn -q

print("\n✅ Environment ready!")
```

---

## CELL 2: Load Data & Define Model

```python
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

# Load data
DATA_DIR = Path("/kaggle/input/gricebench-data")
print("📂 Loading data...")

with open(DATA_DIR / "detector_train.json") as f:
    train_data = json.load(f)
with open(DATA_DIR / "detector_val.json") as f:
    val_data = json.load(f)

print(f"✅ Train: {len(train_data):,} | Val: {len(val_data):,}")

# Dataset class
class ViolationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.maxims = ['quantity', 'quality', 'relation', 'manner']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        enc = self.tokenizer(
            item['input_text'],
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract labels
        labels = torch.tensor(
            [item['labels'].get(m, 0) for m in self.maxims],
            dtype=torch.float
        )
        
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': labels
        }

# Model class
class ViolationDetector(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-base'):
        super().__init__()
        print(f"🔄 Loading {model_name}...")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, 4)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool (use CLS token)
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Classify
        logits = self.classifier(self.dropout(pooled))
        probs = torch.sigmoid(logits)
        
        result = {'logits': logits, 'probs': probs}
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            result['loss'] = loss_fn(logits, labels)
        
        return result

print("✅ Classes defined")
```

---

## CELL 3: Initialize Model & Tokenizer

```python
DEVICE = torch.device('cuda')
MODEL_NAME = 'microsoft/deberta-v3-base'

# Load tokenizer
print(f"\n📥 Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✅ Tokenizer loaded")

# Load model
print(f"\n🧠 Loading model...")
model = ViolationDetector(MODEL_NAME).to(DEVICE)
params = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded: {params:,} parameters")

# Prepare data loaders
print(f"\n📊 Creating data loaders...")
train_dataset = ViolationDataset(train_data, tokenizer)
val_dataset = ViolationDataset(val_data, tokenizer)

BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ {len(train_loader)} train batches, {len(val_loader)} val batches")
```

---

## CELL 4: Training Loop

```python
from transformers import get_scheduler
from sklearn.metrics import f1_score
import time

print("="*70)
print("🏋️ TRAINING")
print("="*70)

# Config
EPOCHS = 3
LR = 2e-5

# Optimizer
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# Scheduler
total_steps = len(train_loader) * EPOCHS
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

# Training history
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
best_f1 = 0.0

# Training loop
for epoch in range(EPOCHS):
    print(f"\n{'#'*70}")
    print(f"# EPOCH {epoch+1}/{EPOCHS}")
    print(f"{'#'*70}")
    
    # === TRAIN ===
    model.train()
    epoch_loss = 0
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        # Move to GPU
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(ids, mask, labels)
        loss = outputs['loss']
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        
        # Progress
        if (i + 1) % 50 == 0 or i == 0:
            avg_loss = epoch_loss / (i + 1)
            print(f"   [{i+1:4d}/{len(train_loader)}] Loss: {avg_loss:.4f}")
    
    avg_train_loss = epoch_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    train_time = time.time() - start_time
    
    print(f"\n✅ Train Loss: {avg_train_loss:.4f} (took {train_time/60:.1f} min)")
    
    # === VALIDATE ===
    print(f"🔍 Validating...")
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(ids, mask, labels)
            val_loss += outputs['loss'].item()
            
            preds = (outputs['probs'] > 0.5).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    avg_val_loss = val_loss / len(val_loader)
    history['val_loss'].append(avg_val_loss)
    
    # F1 scores
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    maxims = ['quantity', 'quality', 'relation', 'manner']
    f1_scores = {}
    for i, maxim in enumerate(maxims):
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        f1_scores[maxim] = f1
    
    macro_f1 = sum(f1_scores.values()) / 4
    history['val_f1'].append(macro_f1)
    
    print(f"\n📊 Val Loss: {avg_val_loss:.4f} | Macro F1: {macro_f1:.4f}")
    for m, f1 in f1_scores.items():
        print(f"   {m:10s}: {f1:.4f}")
    
    # Save best
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save({
            'model_state_dict': model.state_dict(),
            'macro_f1': macro_f1,
            'f1_scores': f1_scores,
            'epoch': epoch
        }, '/kaggle/working/best_model.pt')
        print(f"\n💾 SAVED! F1={macro_f1:.4f}")

print(f"\n{'='*70}")
print(f"🎉 DONE! Best F1: {best_f1:.4f}")
print('='*70)
```

---

## CELL 5: Save Results

```python
import matplotlib.pyplot as plt

# Save history (very small file)
with open('/kaggle/working/history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("✅ Saved history.json")

# Display plot (don't save - disk is full from model file)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss
ax1.plot(history['train_loss'], 'b-o', label='Train', lw=2)
ax1.plot(history['val_loss'], 'r-o', label='Val', lw=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_title('Training & Validation Loss')

# F1
ax2.plot(history['val_f1'], 'g-o', lw=2)
ax2.axhline(0.7, color='orange', linestyle='--', label='Target')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Macro F1')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_ylim([0, 1])
ax2.set_title('Validation F1 Score')

plt.tight_layout()
plt.show()  # Just display, don't save (disk space issue)

# Summary
print("\n" + "="*70)
print("📥 FILES TO DOWNLOAD (Right sidebar → Output tab):")
print("="*70)
print("1. best_model.pt   - Your trained DeBERTa model (~500MB)")
print("2. history.json    - Training metrics (loss, F1 scores)")
print("-"*70)
print(f"🏆 Best F1: {best_f1:.4f}")

if best_f1 >= 0.7:
    print("✅ EXCELLENT model!")
elif best_f1 >= 0.5:
    print("⚠️ Decent - try more epochs or larger model")
else:
    print("❌ Needs more training")

print("="*70)
print("\nℹ️ TIP: Right-click on the plot above to save as image")
print("   Or screenshot it for your records")
```

---

## ✅ Why This Works

1. **No version conflicts** - Uses whatever Kaggle has pre-installed
2. **Minimal installation** - Only adds datasets & scikit-learn
3. **Modern scheduler** - Uses `get_scheduler` (works with all versions)
4. **Clean error handling** - Checks GPU, loads data safely
5. **Progress tracking** - Prints every 50 batches so you know it's working

## 🚀 Instructions

1. Create NEW notebook
2. Settings → GPU T4 x2
3. Add your dataset
4. Copy cells 1-5 in order
5. Run each cell and wait for completion

This WILL work because it doesn't fight with Kaggle's environment.
