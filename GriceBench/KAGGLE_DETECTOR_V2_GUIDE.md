# 🚀 KAGGLE GUIDE: Detector V2 Training (Copy-Paste Ready)

## 📋 Overview

This guide provides **exact copy-paste cells** for training Detector V2 on Kaggle with focal loss and temperature scaling.

**Time Required:** 2-3 hours  
**GPU Required:** T4 x2  
**Cost:** Free (Kaggle provides 30 hours/week)

---

## Step 1: Upload Data to Kaggle (15 minutes)

### 1.1 Go to Kaggle Datasets
1. Open browser: https://www.kaggle.com/datasets
2. Click **"New Dataset"** (blue button, top right)

### 1.2 Upload Files
Drag and drop these 3 files:
- `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\data_processed\detector_data\detector_train_hybrid.json`
- `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\data_processed\detector_data\class_weights_filtered.json`
- `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\data_processed\detector_data\detector_val.json`

### 1.3 Configure Dataset
- **Title:** `gricebench-hybrid-detector-data`
- **Subtitle:** Hybrid training data for detector V2
- **Visibility:** Public
- Click **"Create"**

### 1.4 Copy Dataset ID
After creation, copy the dataset ID from the URL:
```
https://www.kaggle.com/datasets/YOUR_USERNAME/gricebench-hybrid-detector-data
                                 ^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                 Copy this entire part
```

---

## Step 2: Create Kaggle Notebook (5 minutes)

### 2.1 Create New Notebook
1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"** (blue button)
3. Click **"File" → "Rename"** → Name it: `detector-v2-training`

### 2.2 Enable GPU
1. Click **"Settings"** (right sidebar)
2. Under **"Accelerator"**, select **"GPU T4 x2"**
3. Click **"Save"**

### 2.3 Add Dataset
1. Click **"Add Data"** (right sidebar)
2. Search for: `gricebench-hybrid-detector-data`
3. Click **"Add"** on your dataset

---

## Step 3: Copy-Paste Cells (10 minutes)

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
from transformers import AutoTokenizer, AutoModel
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

print("✓ Imports complete")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

**Run this cell** (Shift+Enter)  
**Expected output:** Should show CUDA available: True

---

### CELL 2: Configuration
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 2: Configuration
# ============================================

CONFIG = {
    # Paths configured for: pushkarprabhath/gricebench-hybrid-detector-data
    'train_data_path': '/kaggle/input/gricebench-hybrid-detector-data/detector_train_hybrid.json',
    'val_data_path': '/kaggle/input/gricebench-hybrid-detector-data/detector_val.json',
    'class_weights_path': '/kaggle/input/gricebench-hybrid-detector-data/class_weights_filtered.json',
    
    # Model
    'model_name': 'microsoft/deberta-v3-base',
    'max_length': 512,
    
    # Training
    'batch_size': 8,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_steps': 500,
    'gradient_accumulation_steps': 2,
    
    # Focal Loss
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    
    # Output
    'output_dir': '/kaggle/working/detector_v2',
    'device': device
}

print("Configuration:")
for key, val in CONFIG.items():
    if key != 'device':
        print(f"  {key}: {val}")
```

**✅ Already configured for your dataset!** Just copy-paste as-is.

**Run this cell**

---

### CELL 3: Focal Loss Implementation
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 3: Focal Loss Implementation
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples
    
    FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

print("✓ Focal Loss implemented")
```

**Run this cell**

---

### CELL 4: Dataset Class
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 4: Dataset Class
# ============================================

class MaximViolationDataset(Dataset):
    """Dataset for maxim violation detection"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.maxims = ['quantity', 'quality', 'relation', 'manner']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Construct input text
        context = item.get('context', '')
        response = item.get('response', '')
        evidence = item.get('evidence', '')
        
        if evidence:
            text = f"Context: {context} Evidence: {evidence} Response: {response}"
        else:
            text = f"Context: {context} Response: {response}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get labels
        labels = torch.tensor([
            item.get(f'{maxim}_violation', 0)
            for maxim in self.maxims
        ], dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }

print("✓ Dataset class defined")
```

**Run this cell**

---

### CELL 5: Model Architecture
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 5: Model Architecture
# ============================================

class MaximDetectorV2(nn.Module):
    """Improved detector with separate heads per maxim"""
    
    def __init__(self, model_name, num_maxims=4, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Separate classifier for each maxim
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
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
    
    def get_logits(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)

print("✓ Model architecture defined")
```

**Run this cell**

---

### CELL 6: Load Data and Model
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 6: Load Data and Model
# ============================================

print("Loading tokenizer and data...")

tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

train_dataset = MaximViolationDataset(
    CONFIG['train_data_path'],
    tokenizer,
    CONFIG['max_length']
)

val_dataset = MaximViolationDataset(
    CONFIG['val_data_path'],
    tokenizer,
    CONFIG['max_length']
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=2
)

print(f"✓ Train examples: {len(train_dataset)}")
print(f"✓ Val examples: {len(val_dataset)}")

# Load class weights
if Path(CONFIG['class_weights_path']).exists():
    with open(CONFIG['class_weights_path']) as f:
        class_weights_dict = json.load(f)
    
    maxims = ['quantity', 'quality', 'relation', 'manner']
    class_weights = torch.tensor([
        class_weights_dict[m] for m in maxims
    ], dtype=torch.float).to(CONFIG['device'])
    
    print(f"✓ Loaded class weights: {class_weights.tolist()}")
else:
    class_weights = torch.ones(4).to(CONFIG['device'])
    print("⚠️  No class weights found, using equal weights")

# Initialize model
print(f"\nInitializing model...")
model = MaximDetectorV2(CONFIG['model_name']).to(CONFIG['device'])

print(f"✓ Model loaded on {CONFIG['device']}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Run this cell** (This will download the model - takes 2-3 minutes)

---

### CELL 7: Training Setup
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 7: Training Setup
# ============================================

criterion = FocalLoss(
    alpha=CONFIG['focal_alpha'],
    gamma=CONFIG['focal_gamma']
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=0.01
)

total_steps = len(train_loader) * CONFIG['num_epochs'] // CONFIG['gradient_accumulation_steps']
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG['learning_rate'],
    total_steps=total_steps,
    pct_start=0.1
)

print("✓ Training setup complete")
print(f"  Total steps: {total_steps}")
print(f"  Warmup steps: {int(total_steps * 0.1)}")
```

**Run this cell**

---

### CELL 8: Training Functions
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 8: Training Functions
# ============================================

def train_epoch(model, loader, criterion, optimizer, scheduler, device, class_weights):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training")
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        
        loss = 0
        for i in range(4):
            maxim_loss = criterion(logits[:, i], labels[:, i])
            loss += maxim_loss * class_weights[i]
        
        loss = loss / class_weights.sum()
        loss = loss / CONFIG['gradient_accumulation_steps']
        loss.backward()
        
        if (step + 1) % CONFIG['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * CONFIG['gradient_accumulation_steps']
        progress_bar.set_postfix({'loss': total_loss / (step + 1)})
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    maxims = ['quantity', 'quality', 'relation', 'manner']
    metrics = {}
    
    for i, maxim in enumerate(maxims):
        preds = all_preds[:, i]
        labels = all_labels[:, i]
        probs = all_probs[:, i]
        
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[maxim] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'prob_mean': probs.mean().item(),
            'prob_std': probs.std().item()
        }
    
    avg_f1 = np.mean([m['f1'] for m in metrics.values()])
    return metrics, avg_f1

print("✓ Training functions defined")
```

**Run this cell**

---

### CELL 9: Run Training
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 9: Run Training
# ============================================

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

history = {
    'train_loss': [],
    'val_f1': [],
    'val_metrics': []
}

best_f1 = 0
best_epoch = 0

for epoch in range(CONFIG['num_epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
    print("-"*60)
    
    train_loss = train_epoch(
        model, train_loader, criterion, optimizer,
        scheduler, CONFIG['device'], class_weights
    )
    
    val_metrics, val_f1 = evaluate(model, val_loader, CONFIG['device'])
    
    history['train_loss'].append(train_loss)
    history['val_f1'].append(val_f1)
    history['val_metrics'].append(val_metrics)
    
    print(f"\nTrain Loss: {train_loss:.4f}")
    print(f"Val F1 (avg): {val_f1:.4f}")
    print("\nPer-maxim metrics:")
    for maxim, metrics in val_metrics.items():
        print(f"  {maxim.capitalize():12s}: "
              f"F1={metrics['f1']:.3f}, "
              f"Prob μ={metrics['prob_mean']:.3f}, "
              f"Prob σ={metrics['prob_std']:.3f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_epoch = epoch
        
        output_dir = Path(CONFIG['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
            'val_metrics': val_metrics
        }, output_dir / 'best_model_v2.pt')
        
        print(f"✓ Saved best model (F1={val_f1:.4f})")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Best F1: {best_f1:.4f} (epoch {best_epoch+1})")

with open(CONFIG['output_dir'] + '/history_v2.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"✓ Saved training history")
```

**Run this cell** (This will take 2-3 hours - go get coffee! ☕)

**Expected output:**
```
Epoch 1/5: Train Loss=0.234, Val F1=0.94
Epoch 2/5: Train Loss=0.189, Val F1=0.96
...
Best F1: 0.97
```

---

### CELL 10: Temperature Scaling
**Click "+ Code" and paste:**

```python
# ============================================
# CELL 10: Temperature Scaling
# ============================================

print("\n" + "="*60)
print("TEMPERATURE SCALING CALIBRATION")
print("="*60)

checkpoint = torch.load(CONFIG['output_dir'] + '/best_model_v2.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

all_logits = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Collecting logits"):
        input_ids = batch['input_ids'].to(CONFIG['device'])
        attention_mask = batch['attention_mask'].to(CONFIG['device'])
        labels = batch['labels']
        
        logits = model.get_logits(input_ids, attention_mask)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

all_logits = torch.cat(all_logits)
all_labels = torch.cat(all_labels)

from scipy.optimize import minimize
from sklearn.metrics import log_loss

temperatures = {}
maxims = ['quantity', 'quality', 'relation', 'manner']

for i, maxim in enumerate(maxims):
    logits = all_logits[:, i].numpy()
    labels = all_labels[:, i].numpy()
    
    def objective(T):
        scaled_probs = 1 / (1 + np.exp(-logits / T))
        return log_loss(labels, scaled_probs)
    
    result = minimize(objective, x0=1.0, bounds=[(0.1, 10.0)])
    temperatures[maxim] = float(result.x[0])
    
    print(f"{maxim.capitalize():12s}: T = {temperatures[maxim]:.3f}")

with open(CONFIG['output_dir'] + '/temperatures.json', 'w') as f:
    json.dump(temperatures, f, indent=2)

print(f"\n✓ Saved temperature scaling parameters")

print("\n" + "="*60)
print("DETECTOR V2 TRAINING COMPLETE!")
print("="*60)
print("\nGenerated files:")
print(f"  - best_model_v2.pt")
print(f"  - history_v2.json")
print(f"  - temperatures.json")
print("\n📥 Download these files from /kaggle/working/detector_v2/")
print("="*60)
```

**Run this cell**

---

## Step 4: Download Results (5 minutes)

### 4.1 Download Files
1. In the right sidebar, click **"Output"**
2. Navigate to `detector_v2/` folder
3. Download these 3 files:
   - `best_model_v2.pt` (~500 MB)
   - `history_v2.json` (~5 KB)
   - `temperatures.json` (~1 KB)

### 4.2 Save Locally
Place downloaded files in:
```
c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\models\detector_v2\
```

---

## ✅ Success Criteria

After training completes, verify:
- [ ] Val F1 > 0.95
- [ ] Probability std > 0.15 (check in output)
- [ ] All 3 files downloaded
- [ ] Training took 2-3 hours

**If F1 < 0.95:** Increase epochs to 7 in CONFIG and rerun

---

## 🎉 Next Steps

Once you have the detector V2 files, you're ready for:
1. **DPO Data Scoring** (I'll handle locally)
2. **DPO Optimization Training** (Next Kaggle guide)

**Questions?** Check the troubleshooting section in `complete_optimization_guide.md`
