# 🚨 KAGGLE EMERGENCY FIX - DeBERTa Setup
## "Nuclear Option" to Fix Environment Errors

**Root Cause Found:** The "MessageFactory" error is caused by a `protobuf` library conflict between TensorFlow and Transformers on Kaggle.

**The Fix:** We must forcefully downgrade `protobuf` and use a specific stable `transformers` version.

---

### Step 1: CREATE A NEW NOTEBOOK
Do not reuse the old one. Start fresh to clear the memory.

---

### Step 2: COPY THIS INTO CELL 1 (The Fixer)

```python
print("="*70)
print("🔧 APPLYING EMERGENCY FIX")
print("="*70)

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 1. UNINSTALL EVERYTHING CONFLICTING
print("\n🗑️ Uninstalling conflicting packages...")
!pip uninstall -y transformers tokenizers protobuf sentencepiece -q

# 2. INSTALL THE "GOLDEN COMBINATION"
print("📦 Installing stable versions...")
# transformers 4.30.2 is the most stable version for DeBERTa on Kaggle
# protobuf 3.20.3 fixes the "MessageFactory" error
!pip install transformers==4.30.2 tokenizers==0.13.3 protobuf==3.20.3 sentencepiece==0.1.99 -q
!pip install datasets scikit-learn tqdm -q

print("\n✅ Environment repaired.")
print("⚠️ IMPORTANT: If you still see errors, click 'Run' > 'Restart Session' and run this cell again.")
```

---

### Step 3: COPY THIS INTO CELL 2 (Model Setup)

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import json
from pathlib import Path

# CONFIG
MODEL_NAME = 'microsoft/deberta-v3-base'
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError("❌ NO GPU! Go to Settings -> Accelerator -> GPU T4 x2")

print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# LOAD DATA
DATA_DIR = Path("/kaggle/input/gricebench-data")
with open(DATA_DIR / "detector_train.json") as f:
    train_data = json.load(f)
with open(DATA_DIR / "detector_val.json") as f:
    val_data = json.load(f)
print(f"✅ Loaded {len(train_data):,} train, {len(val_data):,} val")

# MODEL CLASSES
class ViolationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.maxims = ['quantity', 'quality', 'relation', 'manner']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item['input_text'], max_length=512, padding='max_length', 
            truncation=True, return_tensors='pt'
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

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(0.1)
        self.clf = nn.Linear(self.encoder.config.hidden_size, 4)
    
    def forward(self, ids, mask, labels=None):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        pooled = out.last_hidden_state[:, 0, :]
        logits = self.clf(self.drop(pooled))
        probs = torch.sigmoid(logits)
        res = {'logits': logits, 'probs': probs}
        if labels is not None:
            res['loss'] = nn.BCEWithLogitsLoss()(logits, labels)
        return res

# INITIALIZE
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = Detector().to(DEVICE)
print("✅ Model loaded successfully")
```

---

### Step 4: COPY THIS INTO CELL 3 (Training)

```python
from transformers import get_linear_schedule_with_warmup
import time

print("="*50)
print("🚀 TRAINING STARTED")
print("="*50)

train_ds = ViolationDataset(train_data, tokenizer)
val_ds = ViolationDataset(val_data, tokenizer)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, int(steps*0.1), steps)

for epoch in range(EPOCHS):
    print(f"\n📢 EPOCH {epoch+1}/{EPOCHS}")
    model.train()
    
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        out = model(ids, mask, labels)
        out['loss'].backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if i % 50 == 0:
            print(f"   Batch {i}/{len(train_loader)} | Loss: {out['loss'].item():.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            val_loss += model(ids, mask, labels)['loss'].item()
            
    print(f"   ✅ Validation Loss: {val_loss/len(val_loader):.4f}")
    
    # Save
    torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
    print("   💾 Model saved")

print("\n🎉 DONE!")
```

---

### Why this works:
1. **`protobuf==3.20.3`**: This specific version deletes the "MessageFactory" error.
2. **`transformers==4.30.2`**: Works perfectly with DeBERTa-v3 without needing FSDP.
3. **`sentencepiece`**: Explicitly installed for the tokenizer.

**Please start a NEW notebook to ensure no old files interfere.**
