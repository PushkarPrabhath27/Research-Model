#!/usr/bin/env python3
"""
Phase 6 Detector V2 — Research-Grade Notebook Generator
========================================================
Integrates ALL critical fixes from the analysis:
  Fix #1: pos_weight in BCEWithLogitsLoss (prevents model collapse)
  Fix #2: Real-time training health checks (catches problems early)
  Fix #3: True held-out test set (20% reserved before training)
  Fix #4: Prediction distribution / calibration analysis
  Fix #5: Full results JSON with holdout + generalization gap
"""

import json
import os

def create_notebook():
    cells = []

    def add_code(source, execution_count=None):
        cells.append({
            "cell_type": "code",
            "execution_count": execution_count,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": source
        })

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source
        })

    # =========================================================================
    # CELL 0: Markdown Header
    # =========================================================================
    add_md("""# Phase 6: Detector V2 — Research-Grade Training

**Model:** DeBERTa-v3-small (multi-label: Quantity, Quality, Relation, Manner)

**Critical Fixes Applied:**
1. ✅ `pos_weight` in loss function (prevents model collapse to F1=0.0)
2. ✅ Real-time health checks (catches collapse/overfitting during training)
3. ✅ True held-out test set (20% reserved before any training)
4. ✅ Prediction distribution analysis (calibration diagnostics)
5. ✅ Mandatory data assertions (crashes if Phase 4 data missing)
6. ✅ Data leakage check (zero overlap between splits)
7. ✅ Per-generation-method evaluation (injector / mined / adversarial)
8. ✅ Error analysis on worst misclassifications""")

    # =========================================================================
    # CELL 1: Environment Setup
    # =========================================================================
    add_code("""# ============================================================================
# CELL 1: ENVIRONMENT SETUP
# ============================================================================
import subprocess, sys

print("Installing dependencies...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers>=4.35.0', 'accelerate>=0.21.0', 'datasets>=2.14.0',
    'scikit-learn>=1.3.0', 'scipy>=1.11.0'])
print("Dependencies installed.")

import torch
import torch.nn as nn
import os
import gc
import json
import random
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger('Phase6DetectorV2')

# GPU Check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    raise RuntimeError("GPU required for training")

# Progress tracker
class Tracker:
    def __init__(self):
        self.steps = []
        self.start = datetime.now()
    def mark(self, name, status, details=None):
        elapsed = (datetime.now() - self.start).total_seconds()
        self.steps.append({'name': name, 'status': status, 'elapsed': elapsed, 'details': details or {}})
        icon = '✅' if status == 'PASS' else '❌' if status == 'FAIL' else '⏳'
        logger.info(f"{icon} [{elapsed:.0f}s] {name}: {status}")

tracker = Tracker()
tracker.mark('Environment', 'PASS')
""")

    # =========================================================================
    # CELL 2: Configuration
    # =========================================================================
    add_code("""# ============================================================================
# CELL 2: CONFIGURATION
# ============================================================================
@dataclass
class Config:
    # Data
    data_dir: str = '/kaggle/input/gricebench-scientific-fix'
    output_dir: str = '/kaggle/working/detector_v2'
    
    # Model
    model_name: str = 'microsoft/deberta-v3-small'
    num_labels: int = 4
    max_length: int = 512
    
    # Training
    learning_rate: float = 2e-5
    num_epochs: int = 6
    batch_size: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Splits (applied AFTER held-out removal)
    train_ratio: float = 0.80
    val_ratio: float = 0.20
    holdout_ratio: float = 0.20  # Reserved BEFORE splitting
    
    # Verification thresholds
    min_phase4_violations: int = 1000
    
    seed: int = 42

CONFIG = Config()
os.makedirs(CONFIG.output_dir, exist_ok=True)

logger.info(f"Model: {CONFIG.model_name}")
logger.info(f"Data: {CONFIG.data_dir}")
logger.info(f"Holdout ratio: {CONFIG.holdout_ratio}")
logger.info(f"Min Phase 4 violations: {CONFIG.min_phase4_violations}")

tracker.mark('Configuration', 'PASS')
""")

    # =========================================================================
    # CELL 3: Data Structures
    # =========================================================================
    add_code("""# ============================================================================
# CELL 3: DATA STRUCTURES
# ============================================================================
@dataclass
class Example:
    text: str
    labels: List[int]   # [quantity, quality, relation, manner]
    source: str          # 'phase4_violation' or 'phase4_clean'
    example_id: str = ''
    generation_method: str = 'unknown'
    violation_type: str = 'unknown'
    maxim: str = 'unknown'
    
    def __post_init__(self):
        assert len(self.labels) == 4, f"Labels must have 4 elements, got {len(self.labels)}"
        assert self.source in ['phase4_violation', 'phase4_clean'], \\
            f"Source must be 'phase4_violation' or 'phase4_clean', got '{self.source}'"

def normalize_text(text):
    if not text:
        return ''
    text = str(text).strip()
    text = ' '.join(text.split())
    return text

MAXIM_NAMES = ['Quantity', 'Quality', 'Relation', 'Manner']

tracker.mark('Data Structures', 'PASS')
""")

    # =========================================================================
    # CELL 4: Load & VERIFY Phase 4 Data
    # =========================================================================
    add_code("""# ============================================================================
# CELL 4: LOAD & VERIFY PHASE 4 DATA
# ============================================================================
logger.info("=" * 60)
logger.info("LOADING & VERIFYING PHASE 4 DATA")
logger.info("=" * 60)

# ---- Find data file ----
possible_paths = [
    f"{CONFIG.data_dir}/natural_violations.json",
    '/kaggle/input/gricebench-phase4/natural_violations.json',
    '/kaggle/input/datasets/pushkarprabhath/gricebench-scientific-fix/natural_violations.json',
]

phase4_path = None
for path in possible_paths:
    if os.path.exists(path):
        phase4_path = path
        break

if phase4_path is None:
    logger.error("CRITICAL: natural_violations.json NOT FOUND!")
    logger.error("Available files:")
    for root, dirs, files in os.walk('/kaggle/input'):
        for fn in files:
            logger.error(f"  {os.path.join(root, fn)}")
    raise FileNotFoundError(
        "natural_violations.json not found! "
        "Upload Phase 4 output to your Kaggle dataset."
    )

logger.info(f"Found: {phase4_path}")
file_size = os.path.getsize(phase4_path) / 1024
logger.info(f"File size: {file_size:.1f} KB")

# ---- Load raw data ----
with open(phase4_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

logger.info(f"Raw records: {len(raw_data)}")
logger.info(f"Sample keys: {list(raw_data[0].keys())}")

# ---- Process into Examples ----
violations = []
clean_examples = []
errors = []
generation_method_counts = Counter()
violation_type_counts = Counter()

for idx, item in enumerate(raw_data):
    try:
        context = normalize_text(item.get('context', ''))
        gen_method = item.get('generation_method', 'unknown')
        viol_type = item.get('violation_type', 'unknown')
        maxim = item.get('maxim', 'unknown')
        
        # VIOLATION: violated_response
        violated_response = normalize_text(item.get('violated_response', ''))
        if violated_response:
            text = f"{context} [SEP] {violated_response}" if context else violated_response
            
            labels_dict = item.get('labels', {})
            if isinstance(labels_dict, dict):
                labels = [
                    int(labels_dict.get('quantity', 0)),
                    int(labels_dict.get('quality', 0)),
                    int(labels_dict.get('relation', 0)),
                    int(labels_dict.get('manner', 0))
                ]
            else:
                maxim_lower = str(maxim).lower()
                labels = [
                    1 if 'quantity' in maxim_lower else 0,
                    1 if 'quality' in maxim_lower else 0,
                    1 if 'relation' in maxim_lower else 0,
                    1 if 'manner' in maxim_lower else 0
                ]
            
            if sum(labels) > 0 and len(text) > 50:
                violations.append(Example(
                    text=text, labels=labels, source='phase4_violation',
                    example_id=str(item.get('id', f'v_{idx}')),
                    generation_method=gen_method,
                    violation_type=viol_type, maxim=maxim,
                ))
                generation_method_counts[gen_method] += 1
                violation_type_counts[viol_type] += 1
        
        # CLEAN: original_response
        original_response = normalize_text(item.get('original_response', ''))
        if original_response:
            text = f"{context} [SEP] {original_response}" if context else original_response
            if len(text) > 50:
                clean_examples.append(Example(
                    text=text, labels=[0, 0, 0, 0], source='phase4_clean',
                    example_id=f"{item.get('id', idx)}_clean",
                    generation_method='clean', violation_type='none', maxim='none',
                ))
    except Exception as e:
        errors.append(f"Item {idx}: {str(e)}")

# ---- MANDATORY ASSERTIONS ----
print("\\n" + "=" * 60)
print("MANDATORY DATA VERIFICATION")
print("=" * 60)

print(f"\\n  Phase 4 violations: {len(violations)}")
print(f"  Phase 4 clean:      {len(clean_examples)}")
print(f"  Parse errors:       {len(errors)}")

assert len(violations) >= CONFIG.min_phase4_violations, \\
    f"CRITICAL: Only {len(violations)} violations (need >= {CONFIG.min_phase4_violations}). Phase 4 data NOT loaded!"

maxim_counts = Counter()
for ex in violations:
    for i, name in enumerate(MAXIM_NAMES):
        if ex.labels[i] == 1:
            maxim_counts[name] += 1

print(f"\\n  Maxim Distribution:")
for name in MAXIM_NAMES:
    count = maxim_counts.get(name, 0)
    print(f"    {name}: {count}")
    assert count >= 100, f"CRITICAL: {name} has only {count} violations (need >= 100)"

print(f"\\n  Generation Methods:")
for method, count in generation_method_counts.most_common():
    print(f"    {method}: {count} ({100*count/len(violations):.1f}%)")

assert len(clean_examples) >= 500, \\
    f"CRITICAL: Only {len(clean_examples)} clean examples (need >= 500)"

print(f"\\n✅ ALL ASSERTIONS PASSED — Phase 4 data confirmed!")

tracker.mark('Data Verification', 'PASS', {
    'violations': len(violations),
    'clean': len(clean_examples),
    'generation_methods': dict(generation_method_counts),
    'maxim_distribution': dict(maxim_counts),
})
""")

    # =========================================================================
    # CELL 5: Create TRUE Held-Out Test Set (FIX #3)
    # =========================================================================
    add_code("""# ============================================================================
# CELL 5: CREATE TRUE HELD-OUT TEST SET (NEVER IN TRAINING)
# ============================================================================
logger.info("=" * 60)
logger.info("CREATING HELD-OUT TEST SET (COMPLETELY UNSEEN)")
logger.info("=" * 60)

# Use different seed for independence from training
random.seed(999)

# Stratify by generation method for diverse test set
holdout_violations = []
method_groups = defaultdict(list)

for ex in violations:
    method_groups[ex.generation_method].append(ex)

# Take holdout_ratio from each method
for method, examples in method_groups.items():
    n_holdout = max(1, int(len(examples) * CONFIG.holdout_ratio))
    random.shuffle(examples)
    holdout_violations.extend(examples[:n_holdout])
    logger.info(f"  {method}: {n_holdout} held out of {len(examples)}")

# Take holdout_ratio of clean examples
random.shuffle(clean_examples)
n_clean_holdout = int(len(clean_examples) * CONFIG.holdout_ratio)
holdout_clean = clean_examples[:n_clean_holdout]

# Create holdout set
holdout_test_data = holdout_violations + holdout_clean

# Remove from training pool using TEXT (not ID) to handle duplicate texts
# Phase 4 data has ~33 entries with identical text across different IDs
holdout_texts = {ex.text for ex in holdout_test_data}

violations_for_training = [ex for ex in violations if ex.text not in holdout_texts]
clean_for_training = [ex for ex in clean_examples if ex.text not in holdout_texts]

# Verify zero overlap by text
training_texts = {ex.text for ex in violations_for_training + clean_for_training}
overlap = holdout_texts & training_texts

assert len(overlap) == 0, \\
    f"DATA LEAKAGE: {len(overlap)} examples in both holdout and training!"

n_removed_extra = (len(violations) - len(holdout_violations) - len(violations_for_training)) + \\
                  (len(clean_examples) - len(holdout_clean) - len(clean_for_training))
if n_removed_extra > 0:
    logger.info(f"  Removed {n_removed_extra} extra duplicates from training pool")

print(f"\\n  Held-out test: {len(holdout_test_data)} ({len(holdout_violations)} viol + {len(holdout_clean)} clean)")
print(f"  Training pool: {len(violations_for_training) + len(clean_for_training)}")
print(f"  Overlap check: 0 (PASS)")

# Update main variables
violations = violations_for_training
clean_examples = clean_for_training

# Reset seed for consistent training
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

tracker.mark('Held-Out Creation', 'PASS', {
    'holdout_size': len(holdout_test_data),
    'training_pool': len(violations) + len(clean_examples),
})
""")

    # =========================================================================
    # CELL 6: Train/Val Split
    # =========================================================================
    add_code("""# ============================================================================
# CELL 6: TRAIN / VALIDATION SPLIT (from training pool only)
# ============================================================================
logger.info("=" * 60)
logger.info("CREATING TRAIN / VALIDATION SPLIT")
logger.info("=" * 60)

# Combine training pool (held-out already removed!)
all_data = violations + clean_examples
logger.info(f"Training pool (excluding held-out): {len(all_data)}")

random.shuffle(all_data)

# Stratified split by (source, maxim)
groups = defaultdict(list)
for ex in all_data:
    key = (ex.source, ex.maxim)
    groups[key].append(ex)

logger.info(f"Unique (source, maxim) groups: {len(groups)}")

train_data, val_data = [], []

for key, examples in groups.items():
    random.shuffle(examples)
    n = len(examples)
    n_train = max(1, int(n * CONFIG.train_ratio))
    
    train_data.extend(examples[:n_train])
    val_data.extend(examples[n_train:])

random.shuffle(train_data)
random.shuffle(val_data)

# Source distribution per split
def source_dist(data):
    return dict(Counter(ex.source for ex in data))

train_sources = source_dist(train_data)
val_sources = source_dist(val_data)
holdout_sources = source_dist(holdout_test_data)

print(f"\\n  Split Sizes:")
print(f"    Train:   {len(train_data)}  {train_sources}")
print(f"    Val:     {len(val_data)}  {val_sources}")
print(f"    Holdout: {len(holdout_test_data)}  {holdout_sources}")

# Leakage check
train_texts = {ex.text for ex in train_data}
val_texts = {ex.text for ex in val_data}

tv_overlap = len(train_texts & val_texts)
th_overlap = len(train_texts & holdout_texts)

print(f"\\n  Leakage Check:")
print(f"    Train-Val overlap:     {tv_overlap}")
print(f"    Train-Holdout overlap: {th_overlap}")

assert tv_overlap == 0, f"DATA LEAKAGE: {tv_overlap} train-val overlap!"
assert th_overlap == 0, f"DATA LEAKAGE: {th_overlap} train-holdout overlap!"

print(f"  ✅ No data leakage!")

tracker.mark('Data Split', 'PASS', {
    'train': len(train_data),
    'val': len(val_data),
    'holdout': len(holdout_test_data),
    'train_sources': train_sources,
})
""")

    # =========================================================================
    # CELL 7: Dataset & Model (with pos_weight)
    # =========================================================================
    add_code("""# ============================================================================
# CELL 7: DATASET & MODEL (with pos_weight)
# ============================================================================
logger.info("=" * 60)
logger.info("CREATING DATASET & LOADING MODEL")
logger.info("=" * 60)

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name)

class GriceDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex.text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(ex.labels, dtype=torch.float),
        }

train_dataset = GriceDataset(train_data, tokenizer, CONFIG.max_length)
val_dataset = GriceDataset(val_data, tokenizer, CONFIG.max_length)

train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.batch_size * 2, shuffle=False,
                        num_workers=2, pin_memory=True)

logger.info(f"Train batches: {len(train_loader)}")
logger.info(f"Val batches:   {len(val_loader)}")

# ---- Model with pos_weight stored as buffer ----
class GriceDetector(nn.Module):
    \"\"\"Multi-label violation detector with stored pos_weight\"\"\"
    def __init__(self, model_name, num_labels, pos_weight=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels),
        )
        # CRITICAL: Store pos_weight as buffer (persists with model)
        if pos_weight is None:
            pos_weight = torch.ones(num_labels)
        self.register_buffer('pos_weight', pos_weight)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return {'logits': logits}

# ---- FIX #1: Calculate pos_weight from training data ----
logger.info("\\nCalculating pos_weight from training data:")
train_labels_array = np.array([ex.labels for ex in train_data])
pos_counts = train_labels_array.sum(axis=0)
neg_counts = len(train_labels_array) - pos_counts
pos_weight_values = neg_counts / (pos_counts + 1e-6)

# Cap extreme weights
pos_weight_values = np.clip(pos_weight_values, 1.0, 10.0)

pos_weight_tensor = torch.tensor(pos_weight_values, dtype=torch.float32)

for i, name in enumerate(MAXIM_NAMES):
    logger.info(f"  {name}: weight={pos_weight_values[i]:.2f} "
                f"(pos={int(pos_counts[i])}, neg={int(neg_counts[i])})")

model = GriceDetector(CONFIG.model_name, CONFIG.num_labels, pos_weight_tensor).to(device)

# Verify pos_weight
logger.info(f"\\nModel pos_weight: {model.pos_weight.tolist()}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Parameters: {trainable_params:,} / {total_params:,}")

tracker.mark('Model & Data', 'PASS', {
    'params': trainable_params,
    'pos_weight': pos_weight_values.tolist(),
})
""")

    # =========================================================================
    # CELL 8: Training Loop (with health checks)
    # =========================================================================
    add_code("""# ============================================================================
# CELL 8: TRAINING LOOP (with real-time health checks)
# ============================================================================
logger.info("=" * 60)
logger.info("TRAINING")
logger.info("=" * 60)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score

optimizer = AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG.num_epochs * len(train_loader))

# CRITICAL: Use pos_weight from model buffer
criterion = nn.BCEWithLogitsLoss(pos_weight=model.pos_weight)
logger.info(f"Loss: BCEWithLogitsLoss(pos_weight={model.pos_weight.tolist()})")

# ---- Evaluation function ----
def evaluate(model, loader, device, thresholds=None):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs['logits'], labels)
            
            probs = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    if thresholds is None:
        thresholds = [0.5] * CONFIG.num_labels
    
    all_preds = (all_probs >= np.array(thresholds)).astype(int)
    
    per_class = {}
    for i, name in enumerate(MAXIM_NAMES):
        if all_labels[:, i].sum() > 0:
            f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
            prec = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
            rec = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        else:
            f1 = prec = rec = 0.0
        per_class[name] = {'f1': f1, 'precision': prec, 'recall': rec}
    
    macro_f1 = np.mean([v['f1'] for v in per_class.values()])
    avg_loss = total_loss / max(n_batches, 1)
    
    return macro_f1, per_class, avg_loss, all_probs, all_labels

# ---- Threshold optimization ----
def optimize_thresholds(probs, labels):
    best_thresholds = []
    for i in range(CONFIG.num_labels):
        best_f1 = 0
        best_t = 0.5
        for t in np.arange(0.1, 0.95, 0.05):
            preds = (probs[:, i] >= t).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds.append(round(best_t, 2))
    return best_thresholds

# ---- Training ----
training_history = []
best_val_f1 = 0
best_epoch = 0
patience = 0
max_patience = 2
health_alerts = []

train_start = datetime.now()

for epoch in range(1, CONFIG.num_epochs + 1):
    model.train()
    epoch_loss = 0
    n_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs['logits'], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        epoch_loss += loss.item()
        n_batches += 1
        
        if (batch_idx + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_train_loss = epoch_loss / n_batches
    
    # Validate
    val_f1, val_per_class, val_loss, val_probs, val_labels = evaluate(model, val_loader, device)
    
    # Optimize thresholds
    optimal_thresholds = optimize_thresholds(val_probs, val_labels)
    val_f1_opt, val_per_class_opt, _, _, _ = evaluate(model, val_loader, device, optimal_thresholds)
    
    # ================================================================
    # FIX #2: REAL-TIME HEALTH CHECKS
    # ================================================================
    if epoch >= 2:
        # Check 1: Model collapse
        pred_variance = val_probs.var()
        if pred_variance < 0.01:
            alert = f"COLLAPSE ALERT: Pred variance={pred_variance:.4f}"
            logger.warning(f"  ⚠️ {alert}")
            health_alerts.append(f"Epoch {epoch}: {alert}")
        
        # Check 2: Predictions stuck near threshold
        near_threshold = np.sum((val_probs > 0.4) & (val_probs < 0.6)) / val_probs.size
        if near_threshold > 0.7:
            alert = f"THRESHOLD ALERT: {near_threshold:.1%} predictions near 0.5"
            logger.warning(f"  ⚠️ {alert}")
            health_alerts.append(f"Epoch {epoch}: {alert}")
        
        # Check 3: Overfitting
        if len(training_history) >= 1:
            prev_val_loss = training_history[-1]['val_loss']
            if val_loss > prev_val_loss + 0.05:
                alert = f"OVERFITTING: Val loss {prev_val_loss:.3f} -> {val_loss:.3f}"
                logger.warning(f"  ⚠️ {alert}")
                health_alerts.append(f"Epoch {epoch}: {alert}")
        
        # Check 4: Suspiciously high F1 early
        if epoch <= 3 and val_f1_opt > 0.90:
            alert = f"SUSPICIOUS: F1={val_f1_opt:.3f} at epoch {epoch} (too high too early)"
            logger.warning(f"  ⚠️ {alert}")
            health_alerts.append(f"Epoch {epoch}: {alert}")
        
        # Check 5: Per-class collapse
        class_variances = val_probs.var(axis=0)
        collapsed = [MAXIM_NAMES[i] for i in range(4) if class_variances[i] < 0.01]
        if collapsed:
            alert = f"CLASS COLLAPSE: {', '.join(collapsed)} have low variance"
            logger.warning(f"  ⚠️ {alert}")
            health_alerts.append(f"Epoch {epoch}: {alert}")
    
    epoch_result = {
        'epoch': epoch,
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
        'val_macro_f1': val_f1,
        'val_macro_f1_optimized': val_f1_opt,
        'thresholds': dict(zip(MAXIM_NAMES, optimal_thresholds)),
        'per_class': val_per_class_opt,
    }
    training_history.append(epoch_result)
    
    logger.info(f"\\nEpoch {epoch}/{CONFIG.num_epochs}:")
    logger.info(f"  Train loss: {avg_train_loss:.4f}")
    logger.info(f"  Val loss:   {val_loss:.4f}")
    logger.info(f"  Val F1:     {val_f1:.4f} (default) | {val_f1_opt:.4f} (optimized)")
    for name in MAXIM_NAMES:
        sc = val_per_class_opt[name]
        logger.info(f"    {name}: F1={sc['f1']:.3f}, P={sc['precision']:.3f}, R={sc['recall']:.3f}")
    
    # Save best
    if val_f1_opt > best_val_f1:
        best_val_f1 = val_f1_opt
        best_epoch = epoch
        best_thresholds = optimal_thresholds
        torch.save({
            'model_state_dict': model.state_dict(),
            'pos_weight': model.pos_weight,
            'thresholds': best_thresholds,
            'epoch': epoch,
            'val_f1': val_f1_opt,
        }, os.path.join(CONFIG.output_dir, 'best_model.pt'))
        logger.info(f"  ⭐ New best model! F1={val_f1_opt:.4f}")
        patience = 0
    else:
        patience += 1
        logger.info(f"  No improvement ({patience}/{max_patience})")
    
    if patience >= max_patience and epoch >= 3:
        logger.info(f"\\nEarly stopping at epoch {epoch}")
        break

train_time = (datetime.now() - train_start).total_seconds()
logger.info(f"\\nTraining: {train_time:.0f}s ({train_time/60:.1f} min)")
logger.info(f"Best epoch: {best_epoch}, F1={best_val_f1:.4f}")

if health_alerts:
    logger.warning(f"\\n⚠️ Health alerts during training:")
    for alert in health_alerts:
        logger.warning(f"  {alert}")

# Load best checkpoint
checkpoint = torch.load(os.path.join(CONFIG.output_dir, 'best_model.pt'), weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
logger.info("Loaded best model checkpoint")

tracker.mark('Training', 'PASS', {
    'best_epoch': best_epoch,
    'best_f1': best_val_f1,
    'time_seconds': train_time,
    'health_alerts': len(health_alerts),
})
""")

    # =========================================================================
    # CELL 9: In-Distribution Test (from val split)
    # =========================================================================
    add_code("""# ============================================================================
# CELL 9: VALIDATION SET FINAL EVALUATION
# ============================================================================
logger.info("=" * 60)
logger.info("VALIDATION SET FINAL EVALUATION")
logger.info("=" * 60)

val_f1, val_per_class, val_loss, val_probs, val_labels = evaluate(
    model, val_loader, device, best_thresholds
)

print(f"\\n{'='*60}")
print(f"VALIDATION RESULTS (used for threshold optimization)")
print(f"{'='*60}")
print(f"\\nMacro F1: {val_f1:.4f}")
print(f"\\nPer-Maxim:")
for name in MAXIM_NAMES:
    sc = val_per_class[name]
    print(f"  {name}: F1={sc['f1']:.3f}, P={sc['precision']:.3f}, R={sc['recall']:.3f}")

tracker.mark('Val Evaluation', 'PASS', {'val_f1': val_f1})
""")

    # =========================================================================
    # CELL 10: Held-Out Test (FIX #3B - the REAL test)
    # =========================================================================
    add_code("""# ============================================================================
# CELL 10: HELD-OUT TEST (COMPLETELY UNSEEN - THE REAL TEST)
# ============================================================================
logger.info("=" * 60)
logger.info("HELD-OUT TEST (COMPLETELY UNSEEN DATA)")
logger.info("=" * 60)

holdout_dataset = GriceDataset(holdout_test_data, tokenizer, CONFIG.max_length)
holdout_loader = DataLoader(holdout_dataset, batch_size=CONFIG.batch_size * 2,
                            shuffle=False, num_workers=2, pin_memory=True)

holdout_f1, holdout_per_class, holdout_loss, holdout_probs, holdout_labels = evaluate(
    model, holdout_loader, device, best_thresholds
)

print(f"\\n{'='*60}")
print(f"HELD-OUT TEST RESULTS (NEVER SEEN IN TRAINING)")
print(f"{'='*60}")
print(f"\\nMacro F1: {holdout_f1:.4f}")
print(f"Loss: {holdout_loss:.4f}")
print(f"\\nPer-Maxim Performance:")
for name in MAXIM_NAMES:
    sc = holdout_per_class[name]
    flag = " SUSPICIOUS" if sc['f1'] > 0.95 else ""
    print(f"  {name}: F1={sc['f1']:.3f}, P={sc['precision']:.3f}, R={sc['recall']:.3f}{flag}")

# ---- Generalization gap analysis ----
print(f"\\n{'='*60}")
print(f"GENERALIZATION ANALYSIS")
print(f"{'='*60}")
print(f"\\n  Validation F1: {val_f1:.4f}")
print(f"  Held-out F1:   {holdout_f1:.4f}")

gen_gap = abs(val_f1 - holdout_f1)
print(f"  Gap:           {gen_gap:.4f}")

if gen_gap > 0.10:
    print(f"\\n  ⚠️ LARGE GAP (>0.10): Significant overfitting!")
elif gen_gap > 0.05:
    print(f"\\n  ⚠️ MODERATE GAP (>0.05): Some overfitting")
else:
    print(f"\\n  ✅ SMALL GAP (<0.05): Excellent generalization!")

# ---- Per-generation-method eval on holdout ----
print(f"\\n{'='*60}")
print(f"PER-GENERATION-METHOD BREAKDOWN (HOLDOUT)")
print(f"{'='*60}")

holdout_preds = (holdout_probs >= np.array(best_thresholds)).astype(int)
method_examples = defaultdict(list)

for i, ex in enumerate(holdout_test_data):
    if i < len(holdout_preds):
        method_examples[ex.generation_method].append({
            'true': holdout_labels[i] if i < len(holdout_labels) else np.array(ex.labels),
            'pred': holdout_preds[i],
        })

method_results = {}
for method, items in method_examples.items():
    true_arr = np.array([item['true'] for item in items])
    pred_arr = np.array([item['pred'] for item in items])
    
    method_f1s = {}
    for j, name in enumerate(MAXIM_NAMES):
        if true_arr[:, j].sum() > 0:
            method_f1s[name] = f1_score(true_arr[:, j], pred_arr[:, j], zero_division=0)
    
    macro = np.mean(list(method_f1s.values())) if method_f1s else 0
    method_results[method] = {'macro_f1': macro, 'per_class': method_f1s, 'count': len(items)}
    
    print(f"\\n  {method} ({len(items)} examples): Macro F1={macro:.3f}")
    for name, f1 in method_f1s.items():
        print(f"    {name}: {f1:.3f}")

# ---- Health assessment ----
print(f"\\n{'='*60}")
print(f"FINAL HEALTH ASSESSMENT")
print(f"{'='*60}")

if holdout_f1 > 0.95:
    print(f"\\n  CRITICAL: Held-out F1={holdout_f1:.3f} is suspiciously high!")
    print(f"    Possible causes: data leakage, Phase 4 patterns too easy")
    print(f"    DO NOT DEPLOY — investigate first")
elif holdout_f1 > 0.85:
    print(f"\\n  EXCELLENT: Held-out F1={holdout_f1:.3f}")
    print(f"    Ready for Phase 7 evaluation")
elif holdout_f1 > 0.70:
    print(f"\\n  GOOD: Held-out F1={holdout_f1:.3f}")
    print(f"    Acceptable for deployment, consider tuning")
elif holdout_f1 > 0.55:
    print(f"\\n  MODERATE: Held-out F1={holdout_f1:.3f}")
    print(f"    Needs improvement before Phase 7")
else:
    print(f"\\n  LOW: Held-out F1={holdout_f1:.3f}")
    print(f"    Check model health, data quality, hyperparameters")

tracker.mark('Held-Out Evaluation', 'PASS', {
    'holdout_f1': holdout_f1,
    'generalization_gap': gen_gap,
})
""")

    # =========================================================================
    # CELL 11: Error Analysis
    # =========================================================================
    add_code("""# ============================================================================
# CELL 11: ERROR ANALYSIS (on held-out set)
# ============================================================================
logger.info("=" * 60)
logger.info("ERROR ANALYSIS")
logger.info("=" * 60)

errors_list = []
for i in range(min(len(holdout_test_data), len(holdout_preds))):
    ex = holdout_test_data[i]
    pred = holdout_preds[i]
    true = np.array(ex.labels)
    prob = holdout_probs[i]
    
    error_count = int(np.sum(pred != true))
    if error_count > 0:
        errors_list.append({
            'idx': i,
            'text': ex.text[:300],
            'true_labels': true.tolist(),
            'pred_labels': pred.tolist(),
            'probs': prob.tolist(),
            'source': ex.source,
            'generation_method': ex.generation_method,
            'violation_type': ex.violation_type,
            'maxim': ex.maxim,
            'error_count': error_count,
        })

errors_list.sort(key=lambda x: x['error_count'], reverse=True)

print(f"\\n  Total misclassified: {len(errors_list)} / {len(holdout_test_data)} ({100*len(errors_list)/max(len(holdout_test_data),1):.1f}%)")
print(f"  Correctly classified: {len(holdout_test_data) - len(errors_list)}")

# Error type breakdown
error_by_maxim = defaultdict(lambda: {'false_pos': 0, 'false_neg': 0})
for err in errors_list:
    for j, name in enumerate(MAXIM_NAMES):
        if err['true_labels'][j] == 1 and err['pred_labels'][j] == 0:
            error_by_maxim[name]['false_neg'] += 1
        elif err['true_labels'][j] == 0 and err['pred_labels'][j] == 1:
            error_by_maxim[name]['false_pos'] += 1

print(f"\\n  Error Type Breakdown:")
for name in MAXIM_NAMES:
    fp = error_by_maxim[name]['false_pos']
    fn = error_by_maxim[name]['false_neg']
    print(f"    {name}: {fp} FP, {fn} FN")

# Error by generation method
error_by_method = Counter(err['generation_method'] for err in errors_list)
print(f"\\n  Errors by Generation Method:")
for method, count in error_by_method.most_common():
    total = sum(1 for ex in holdout_test_data if ex.generation_method == method)
    print(f"    {method}: {count}/{total} ({100*count/max(total,1):.1f}%)")

# Top 10 worst errors
print(f"\\n  Top 10 Worst Misclassifications:")
for rank, err in enumerate(errors_list[:10], 1):
    true_maxims = ', '.join(MAXIM_NAMES[j] for j in range(4) if err['true_labels'][j]==1) or 'Clean'
    pred_maxims = ', '.join(MAXIM_NAMES[j] for j in range(4) if err['pred_labels'][j]==1) or 'Clean'
    print(f"\\n    #{rank} [{err['generation_method']}] {err['violation_type']}")
    print(f"      Text: {err['text'][:120]}...")
    print(f"      True: {true_maxims}")
    print(f"      Pred: {pred_maxims}")
    print(f"      Probs: [{', '.join(f'{p:.2f}' for p in err['probs'])}]")

tracker.mark('Error Analysis', 'PASS', {'total_errors': len(errors_list)})
""")

    # =========================================================================
    # CELL 12: Prediction Distribution Analysis (FIX #4)
    # =========================================================================
    add_code("""# ============================================================================
# CELL 12: PREDICTION DISTRIBUTION ANALYSIS (CALIBRATION)
# ============================================================================
logger.info("=" * 60)
logger.info("CALIBRATION ANALYSIS")
logger.info("=" * 60)

print(f"\\n{'='*60}")
print(f"MODEL CALIBRATION ANALYSIS")
print(f"{'='*60}")

for i, name in enumerate(MAXIM_NAMES):
    probs_class = holdout_probs[:, i]
    
    print(f"\\n  {name}:")
    print(f"    Mean: {probs_class.mean():.3f}  Std: {probs_class.std():.3f}")
    print(f"    Min:  {probs_class.min():.3f}  Max: {probs_class.max():.3f}")
    
    bins = np.histogram(probs_class, bins=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])[0]
    total = len(probs_class)
    
    print(f"    Distribution:")
    ranges = ['[0.0-0.1)', '[0.1-0.3)', '[0.3-0.5)', '[0.5-0.7)', '[0.7-0.9)', '[0.9-1.0]']
    for r, b in zip(ranges, bins):
        bar = '#' * int(40 * b / max(total, 1))
        print(f"      {r}: {b:4d} ({100*b/total:5.1f}%) {bar}")
    
    if probs_class.std() < 0.10:
        print(f"    ⚠️ LOW VARIANCE — model not confident separating this class")
    
    middle = (bins[2] + bins[3]) / total
    if middle > 0.70:
        print(f"    ⚠️ CLUSTERED — {middle:.1%} predictions around 0.5")

# Overall calibration
all_flat = holdout_probs.flatten()
print(f"\\n  Overall: mean={all_flat.mean():.3f}, std={all_flat.std():.3f}")

if all_flat.std() < 0.15:
    print(f"  ⚠️ Very low overall variance — may be collapsed")
elif all_flat.std() > 0.35:
    print(f"  ⚠️ Very high variance — may be overconfident")
else:
    print(f"  ✅ Healthy variance — good calibration")

tracker.mark('Calibration Analysis', 'PASS')
""")

    # =========================================================================
    # CELL 13: Save Results (FIX #5 — includes holdout)
    # =========================================================================
    add_code("""# ============================================================================
# CELL 13: SAVE COMPREHENSIVE RESULTS
# ============================================================================
logger.info("=" * 60)
logger.info("SAVING RESULTS")
logger.info("=" * 60)

results = {
    'phase': 'Phase 6 - Detector V2 (Research-Grade, Verified)',
    'timestamp': datetime.now().isoformat(),
    'model': CONFIG.model_name,
    'best_epoch': best_epoch,
    'thresholds': dict(zip(MAXIM_NAMES, best_thresholds)),
    'pos_weight': model.pos_weight.tolist(),
    
    'data_verification': {
        'total_violations_loaded': len(violations) + len(holdout_violations),
        'total_clean_loaded': len(clean_examples) + len(holdout_clean),
        'generation_methods': dict(generation_method_counts),
        'maxim_counts': dict(maxim_counts),
        'source_file': phase4_path,
        'assertions_passed': True,
    },
    
    'splits': {
        'train': len(train_data),
        'val': len(val_data),
        'holdout_test': len(holdout_test_data),
        'train_sources': train_sources,
        'val_sources': val_sources,
        'holdout_sources': holdout_sources,
        'leakage_check': 'PASSED',
    },
    
    'validation': {
        'macro_f1': val_f1,
        'per_class': {name: val_per_class[name] for name in MAXIM_NAMES},
    },
    
    'holdout_test': {
        'macro_f1': holdout_f1,
        'loss': holdout_loss,
        'per_class': {name: holdout_per_class[name] for name in MAXIM_NAMES},
        'generalization_gap': gen_gap,
    },
    
    'holdout_per_method': {method: {
        'macro_f1': info['macro_f1'],
        'count': info['count'],
        'per_class': info['per_class'],
    } for method, info in method_results.items()},
    
    'error_analysis': {
        'total_errors': len(errors_list),
        'error_rate': round(100 * len(errors_list) / max(len(holdout_test_data),1), 2),
        'error_by_maxim': {k: dict(v) for k, v in error_by_maxim.items()},
        'error_by_method': dict(error_by_method),
        'top_10_errors': errors_list[:10],
    },
    
    'training_history': training_history,
    'health_alerts': health_alerts,
    
    'gpu': {
        'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'peak_vram_gb': float(torch.cuda.max_memory_allocated(0) / 1e9) if torch.cuda.is_available() else 0,
    },
    'training_time_seconds': train_time,
}

# Save results
results_path = os.path.join(CONFIG.output_dir, 'detector_v2_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
logger.info(f"Results: {results_path}")

# Save thresholds
thresholds_path = os.path.join(CONFIG.output_dir, 'optimal_thresholds.json')
with open(thresholds_path, 'w') as f:
    json.dump({
        'thresholds': dict(zip(MAXIM_NAMES, best_thresholds)),
        'macro_f1': holdout_f1,
        'generalization_gap': gen_gap,
    }, f, indent=2)
logger.info(f"Thresholds: {thresholds_path}")

# Copy to /kaggle/working for easy download
import shutil
for fname in ['detector_v2_results.json', 'optimal_thresholds.json', 'best_model.pt']:
    src = os.path.join(CONFIG.output_dir, fname)
    if os.path.exists(src):
        dst = os.path.join('/kaggle/working', fname)
        shutil.copy2(src, dst)
        logger.info(f"Copied: {dst}")

tracker.mark('Save Results', 'PASS')
""")

    # =========================================================================
    # CELL 14: Final Summary
    # =========================================================================
    add_code("""# ============================================================================
# CELL 14: FINAL SUMMARY
# ============================================================================
print("\\n" + "=" * 60)
print("PHASE 6 DETECTOR V2 — FINAL SUMMARY")
print("=" * 60)

print(f"\\n  Model: {CONFIG.model_name}")
print(f"  Data: {len(violations) + len(holdout_violations)} violations + "
      f"{len(clean_examples) + len(holdout_clean)} clean from Phase 4")
print(f"  pos_weight: {model.pos_weight.tolist()}")

print(f"\\n  Training:")
print(f"    Best epoch: {best_epoch}")
print(f"    Val F1: {best_val_f1:.4f}")
print(f"    Time: {train_time:.0f}s ({train_time/60:.1f} min)")

print(f"\\n  HELD-OUT TEST (ultimate metric):")
print(f"    Macro F1: {holdout_f1:.4f}")
for name in MAXIM_NAMES:
    sc = holdout_per_class[name]
    print(f"      {name}: F1={sc['f1']:.3f}")

print(f"\\n  Generalization gap: {gen_gap:.4f}")

print(f"\\n  Per-Method (holdout):")
for method, info in sorted(method_results.items(), key=lambda x: -x[1]['macro_f1']):
    print(f"    {method}: F1={info['macro_f1']:.3f} ({info['count']} examples)")

print(f"\\n  Errors: {len(errors_list)}/{len(holdout_test_data)} ({100*len(errors_list)/max(len(holdout_test_data),1):.1f}%)")

if health_alerts:
    print(f"\\n  ⚠️ Health alerts: {len(health_alerts)}")
    for a in health_alerts:
        print(f"    • {a}")
else:
    print(f"\\n  ✅ No health alerts during training")

print(f"\\n  Output Files:")
print(f"    /kaggle/working/detector_v2_results.json")
print(f"    /kaggle/working/optimal_thresholds.json")
print(f"    /kaggle/working/best_model.pt")

print(f"\\n{'='*60}")
print(f"PHASE 6 V2 COMPLETE")
print(f"{'='*60}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

tracker.mark('Complete', 'PASS')
print("\\nExecution log:")
for step in tracker.steps:
    print(f"  {step['status']}: {step['name']} ({step['elapsed']:.0f}s)")
""")

    # =========================================================================
    # Build Notebook JSON
    # =========================================================================
    notebook = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            },
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [
                    {
                        "sourceId": 0,
                        "sourceType": "datasetVersion",
                        "datasetSlug": "gricebench-scientific-fix"
                    }
                ],
                "isInternetEnabled": True,
                "isGpuEnabled": True
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": cells
    }

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'KAGGLE_PHASE6_DETECTOR_V2.ipynb'
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
    md_cells = sum(1 for c in cells if c['cell_type'] == 'markdown')
    print(f"Notebook created: {output_path}")
    print(f"Total cells: {len(cells)} ({code_cells} code, {md_cells} markdown)")

    return output_path

if __name__ == '__main__':
    create_notebook()
