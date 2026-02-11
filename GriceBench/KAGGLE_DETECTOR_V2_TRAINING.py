"""
KAGGLE NOTEBOOK: Detector Retraining with Focal Loss

This notebook retrains the detector on filtered/hybrid data using:
1. Focal Loss (handles class imbalance and hard examples)
2. Class weights (balances maxim distribution)
3. Temperature scaling (post-hoc calibration)

Upload this as a Kaggle notebook and run with GPU.
"""

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

# ============================================
# CELL 2: Configuration
# ============================================

CONFIG = {
    # Paths (update these with your Kaggle dataset IDs)
    'train_data_path': '/kaggle/input/gricebench-hybrid-detector-data/detector_train_hybrid.json',
    'val_data_path': '/kaggle/input/gricebench-detector-data/detector_val.json',
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
    print(f"  {key}: {val}")

# ============================================
# CELL 3: Focal Loss Implementation
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples
    
    FL(p_t) = -alpha * (1-p_t)^gamma * log(p_t)
    
    - alpha: balances positive/negative examples
    - gamma: focuses on hard examples (higher gamma = more focus)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        inputs: logits (before sigmoid)
        targets: binary labels (0 or 1)
        """
        # Binary cross entropy
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Get probabilities
        pt = torch.exp(-BCE_loss)
        
        # Focal loss
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        return F_loss.mean()

print("✓ Focal Loss implemented")

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

# ============================================
# CELL 5: Model Architecture
# ============================================

class MaximDetectorV2(nn.Module):
    """
    Improved detector with:
    - DeBERTa-v3-base encoder
    - Separate classification heads per maxim
    - Dropout for regularization
    """
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
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        
        # Classify each maxim
        logits = torch.cat([
            classifier(pooled)
            for classifier in self.classifiers
        ], dim=1)
        
        return logits
    
    def get_logits(self, input_ids, attention_mask):
        """Get raw logits (for temperature scaling)"""
        return self.forward(input_ids, attention_mask)

print("✓ Model architecture defined")

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

# ============================================
# CELL 7: Training Setup
# ============================================

# Loss function
criterion = FocalLoss(
    alpha=CONFIG['focal_alpha'],
    gamma=CONFIG['focal_gamma']
)

# Optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=0.01
)

# Learning rate scheduler
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

# ============================================
# CELL 8: Training Loop
# ============================================

def train_epoch(model, loader, criterion, optimizer, scheduler, device, class_weights):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(loader, desc="Training")
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        logits = model(input_ids, attention_mask)
        
        # Compute weighted loss per maxim
        loss = 0
        for i in range(4):
            maxim_loss = criterion(logits[:, i], labels[:, i])
            loss += maxim_loss * class_weights[i]
        
        loss = loss / class_weights.sum()
        loss = loss / CONFIG['gradient_accumulation_steps']
        
        # Backward
        loss.backward()
        
        # Update weights
        if (step + 1) % CONFIG['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * CONFIG['gradient_accumulation_steps']
        progress_bar.set_postfix({'loss': total_loss / (step + 1)})
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    """Evaluate model"""
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
    
    # Calculate metrics per maxim
    maxims = ['quantity', 'quality', 'relation', 'manner']
    metrics = {}
    
    for i, maxim in enumerate(maxims):
        preds = all_preds[:, i]
        labels = all_labels[:, i]
        probs = all_probs[:, i]
        
        # F1 score
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Probability statistics
        prob_mean = probs.mean().item()
        prob_std = probs.std().item()
        
        metrics[maxim] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'prob_mean': prob_mean,
            'prob_std': prob_std
        }
    
    # Average F1
    avg_f1 = np.mean([m['f1'] for m in metrics.values()])
    
    return metrics, avg_f1

print("✓ Training functions defined")

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
    
    # Train
    train_loss = train_epoch(
        model, train_loader, criterion, optimizer,
        scheduler, CONFIG['device'], class_weights
    )
    
    # Evaluate
    val_metrics, val_f1 = evaluate(model, val_loader, CONFIG['device'])
    
    # Log
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
    
    # Save best model
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

# Save history
with open(CONFIG['output_dir'] + '/history_v2.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"✓ Saved training history")

# ============================================
# CELL 10: Temperature Scaling Calibration
# ============================================

print("\n" + "="*60)
print("TEMPERATURE SCALING CALIBRATION")
print("="*60)

# Load best model
checkpoint = torch.load(CONFIG['output_dir'] + '/best_model_v2.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get validation logits and labels
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

# Find optimal temperature per maxim
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

# Save temperatures
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
print("\nDownload these files and use for evaluation!")
print("="*60)
