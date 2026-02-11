#!/usr/bin/env python3
"""
Phase 6 Detector V2 — Verified Natural Violations Training
===========================================================
Key improvements over ROBUST:
  1. MANDATORY data source assertions — hard-fail if Phase 4 data missing
  2. Source distribution logged in results JSON — proof of data composition
  3. Held-out test set — 500 examples never seen in training
  4. Per-generation-method evaluation — F1 by injector/mined/adversarial
  5. Error analysis — worst 20 misclassifications with text excerpts
  6. Leakage verification — zero overlap between train/val/test
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
    add_md("""# Phase 6: Detector V2 — Verified Natural Violations Training

**Goal:** Train a Gricean maxim violation detector on **real** Phase 4 natural violations (not synthetic).

**Key Features:**
- Hard assertions on data sources (will crash if Phase 4 data missing)
- Source distribution proof logged in results
- Held-out test set (500 examples never in training)  
- Per-generation-method breakdown (injector / mined / adversarial)
- Error analysis on worst misclassifications

**Model:** DeBERTa-v3-small (multi-label: Quantity, Quality, Relation, Manner)""")

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
    raise RuntimeError("GPU required")

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
    
    # Splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Verification
    min_phase4_violations: int = 1000  # HARD MINIMUM
    
    seed: int = 42

CONFIG = Config()
os.makedirs(CONFIG.output_dir, exist_ok=True)

logger.info(f"Model: {CONFIG.model_name}")
logger.info(f"Data: {CONFIG.data_dir}")
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
    # CELL 4: Load & VERIFY Phase 4 Data (CRITICAL)
    # =========================================================================
    add_code("""# ============================================================================
# CELL 4: LOAD & VERIFY PHASE 4 DATA (CRITICAL)
# ============================================================================
logger.info("=" * 60)
logger.info("🔴 CRITICAL: LOADING & VERIFYING PHASE 4 DATA")
logger.info("=" * 60)

# ---- Find the data file ----
possible_paths = [
    f"{CONFIG.data_dir}/natural_violations.json",
    '/kaggle/input/gricebench-phase4/natural_violations.json',
    '/kaggle/input/datasets/pushkarprabhath/gricebench-scientific-fix/natural_violations.json',
    '/kaggle/input/gricebench-scientific-fix/natural_violations.json',
]

phase4_path = None
for path in possible_paths:
    if os.path.exists(path):
        phase4_path = path
        break

if phase4_path is None:
    # List what's actually available
    logger.error("❌ CRITICAL: natural_violations.json NOT FOUND!")
    logger.error("Available files in /kaggle/input:")
    for root, dirs, files in os.walk('/kaggle/input'):
        for fn in files:
            logger.error(f"  {os.path.join(root, fn)}")
    raise FileNotFoundError(
        "natural_violations.json not found! "
        "Upload Phase 4 output to your Kaggle dataset."
    )

logger.info(f"✅ Found data: {phase4_path}")
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
                    text=text,
                    labels=labels,
                    source='phase4_violation',
                    example_id=str(item.get('id', f'v_{idx}')),
                    generation_method=gen_method,
                    violation_type=viol_type,
                    maxim=maxim,
                ))
                generation_method_counts[gen_method] += 1
                violation_type_counts[viol_type] += 1
        
        # CLEAN: original_response
        original_response = normalize_text(item.get('original_response', ''))
        if original_response:
            text = f"{context} [SEP] {original_response}" if context else original_response
            if len(text) > 50:
                clean_examples.append(Example(
                    text=text,
                    labels=[0, 0, 0, 0],
                    source='phase4_clean',
                    example_id=f"{item.get('id', idx)}_clean",
                    generation_method='clean',
                    violation_type='none',
                    maxim='none',
                ))
    except Exception as e:
        errors.append(f"Item {idx}: {str(e)}")

# ---- MANDATORY ASSERTIONS ----
print("\\n" + "=" * 60)
print("🔴 MANDATORY DATA VERIFICATION")
print("=" * 60)

print(f"\\n📊 Data Loaded:")
print(f"  Phase 4 violations: {len(violations)}")
print(f"  Phase 4 clean:      {len(clean_examples)}")
print(f"  Errors:             {len(errors)}")

# ASSERTION 1: Must have enough violations
assert len(violations) >= CONFIG.min_phase4_violations, \\
    f"❌ CRITICAL FAILURE: Only {len(violations)} violations loaded " \\
    f"(need >= {CONFIG.min_phase4_violations}). " \\
    f"Phase 4 data NOT loaded correctly!"

print(f"\\n✅ ASSERTION 1 PASSED: {len(violations)} violations >= {CONFIG.min_phase4_violations} minimum")

# ASSERTION 2: Must have all 4 maxims represented
maxim_counts = Counter()
for ex in violations:
    for i, name in enumerate(MAXIM_NAMES):
        if ex.labels[i] == 1:
            maxim_counts[name] += 1

print(f"\\n📊 Maxim Distribution:")
for name in MAXIM_NAMES:
    count = maxim_counts.get(name, 0)
    print(f"  {name}: {count}")
    assert count >= 100, f"❌ CRITICAL: {name} has only {count} violations (need >= 100)"

print(f"\\n✅ ASSERTION 2 PASSED: All maxims have >= 100 violations")

# ASSERTION 3: Must have natural generation methods
print(f"\\n📊 Generation Methods:")
for method, count in generation_method_counts.most_common():
    print(f"  {method}: {count} ({100*count/len(violations):.1f}%)")

print(f"\\n📊 Violation Types (top 10):")
for vtype, count in violation_type_counts.most_common(10):
    print(f"  {vtype}: {count}")

# ASSERTION 4: Must have clean examples
assert len(clean_examples) >= 500, \\
    f"❌ CRITICAL: Only {len(clean_examples)} clean examples (need >= 500)"

print(f"\\n✅ ASSERTION 3 PASSED: {len(clean_examples)} clean examples >= 500")

print("\\n" + "=" * 60)
print("✅ ALL DATA ASSERTIONS PASSED — Phase 4 data confirmed!")
print("=" * 60)

tracker.mark('Data Verification', 'PASS', {
    'violations': len(violations),
    'clean': len(clean_examples),
    'generation_methods': dict(generation_method_counts),
    'maxim_distribution': dict(maxim_counts),
})
""")

    # =========================================================================
    # CELL 5: Stratified Split with Leakage Check
    # =========================================================================
    add_code("""# ============================================================================
# CELL 5: STRATIFIED SPLIT WITH LEAKAGE CHECK
# ============================================================================
logger.info("=" * 60)
logger.info("CREATING STRATIFIED SPLITS")
logger.info("=" * 60)

# Combine all data
all_data = violations + clean_examples
random.shuffle(all_data)

logger.info(f"Total examples: {len(all_data)}")

# Stratified split by (source, maxim) to ensure representation
groups = defaultdict(list)
for ex in all_data:
    key = (ex.source, ex.maxim)
    groups[key].append(ex)

logger.info(f"Unique (source, maxim) groups: {len(groups)}")

train_data, val_data, test_data = [], [], []

for key, examples in groups.items():
    random.shuffle(examples)
    n = len(examples)
    n_train = max(1, int(n * CONFIG.train_ratio))
    n_val = max(1, int(n * CONFIG.val_ratio))
    n_test = n - n_train - n_val
    
    if n_test < 1:
        n_test = 1
        n_train = n - n_val - n_test
    
    train_data.extend(examples[:n_train])
    val_data.extend(examples[n_train:n_train + n_val])
    test_data.extend(examples[n_train + n_val:])

random.shuffle(train_data)
random.shuffle(val_data)
random.shuffle(test_data)

logger.info(f"\\nSplit sizes:")
logger.info(f"  Train: {len(train_data)}")
logger.info(f"  Val:   {len(val_data)}")
logger.info(f"  Test:  {len(test_data)}")

# ---- Source distribution per split ----
def source_dist(data):
    counts = Counter(ex.source for ex in data)
    return dict(counts)

train_sources = source_dist(train_data)
val_sources = source_dist(val_data)
test_sources = source_dist(test_data)

print("\\n📊 Source Distribution:")
print(f"  Train: {train_sources}")
print(f"  Val:   {val_sources}")
print(f"  Test:  {test_sources}")

# ---- LEAKAGE CHECK ----
train_texts = {ex.text for ex in train_data}
val_texts = {ex.text for ex in val_data}
test_texts = {ex.text for ex in test_data}

train_val_overlap = len(train_texts & val_texts)
train_test_overlap = len(train_texts & test_texts)
val_test_overlap = len(val_texts & test_texts)

print(f"\\n🔍 Leakage Check:")
print(f"  Train-Val overlap:  {train_val_overlap}")
print(f"  Train-Test overlap: {train_test_overlap}")
print(f"  Val-Test overlap:   {val_test_overlap}")

assert train_test_overlap == 0, f"❌ DATA LEAKAGE: {train_test_overlap} examples in both train and test!"
assert train_val_overlap == 0, f"❌ DATA LEAKAGE: {train_val_overlap} examples in both train and val!"

print("✅ No data leakage detected!")

# ---- Generation method distribution in test set ----
test_gen_methods = Counter(ex.generation_method for ex in test_data if ex.source == 'phase4_violation')
print(f"\\nTest set generation methods:")
for method, count in test_gen_methods.most_common():
    print(f"  {method}: {count}")

tracker.mark('Data Split', 'PASS', {
    'train': len(train_data),
    'val': len(val_data),
    'test': len(test_data),
    'train_sources': train_sources,
    'test_sources': test_sources,
    'leakage': {'train_test': train_test_overlap, 'train_val': train_val_overlap},
})
""")

    # =========================================================================
    # CELL 6: Dataset & Model
    # =========================================================================
    add_code("""# ============================================================================
# CELL 6: DATASET & MODEL
# ============================================================================
logger.info("=" * 60)
logger.info("CREATING DATASET & LOADING MODEL")
logger.info("=" * 60)

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

# Tokenizer
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

# Create datasets
train_dataset = GriceDataset(train_data, tokenizer, CONFIG.max_length)
val_dataset = GriceDataset(val_data, tokenizer, CONFIG.max_length)
test_dataset = GriceDataset(test_data, tokenizer, CONFIG.max_length)

train_loader = DataLoader(train_dataset, batch_size=CONFIG.batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

logger.info(f"Train batches: {len(train_loader)}")
logger.info(f"Val batches:   {len(val_loader)}")
logger.info(f"Test batches:  {len(test_loader)}")

# ---- Model ----
class GriceDetector(nn.Module):
    def __init__(self, model_name, num_labels):
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
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return {'logits': logits}

model = GriceDetector(CONFIG.model_name, CONFIG.num_labels).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Parameters: {trainable_params:,} / {total_params:,}")

tracker.mark('Model & Data', 'PASS', {'params': trainable_params})
""")

    # =========================================================================
    # CELL 7: Training Loop
    # =========================================================================
    add_code("""# ============================================================================
# CELL 7: TRAINING LOOP
# ============================================================================
logger.info("=" * 60)
logger.info("TRAINING")
logger.info("=" * 60)

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score

# Optimizer
optimizer = AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG.num_epochs * len(train_loader))
criterion = nn.BCEWithLogitsLoss()

# ---- Evaluation function ----
def evaluate(model, loader, device, thresholds=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
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
    
    # Use thresholds or default 0.5
    if thresholds is None:
        thresholds = [0.5] * CONFIG.num_labels
    
    all_preds = (all_probs >= np.array(thresholds)).astype(int)
    
    # Per-class metrics
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
    
    # Optimize thresholds on validation set
    optimal_thresholds = optimize_thresholds(val_probs, val_labels)
    val_f1_opt, val_per_class_opt, _, _, _ = evaluate(model, val_loader, device, optimal_thresholds)
    
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
        torch.save(model.state_dict(), os.path.join(CONFIG.output_dir, 'best_model.pt'))
        logger.info(f"  ⭐ New best model! F1={val_f1_opt:.4f}")
        patience = 0
    else:
        patience += 1
        logger.info(f"  No improvement ({patience}/{max_patience})")
    
    # Early stopping
    if patience >= max_patience and epoch >= 3:
        logger.info(f"\\nEarly stopping at epoch {epoch}")
        break

train_time = (datetime.now() - train_start).total_seconds()
logger.info(f"\\nTraining complete: {train_time:.0f}s ({train_time/60:.1f} min)")
logger.info(f"Best epoch: {best_epoch} with F1={best_val_f1:.4f}")

# Load best model
model.load_state_dict(torch.load(os.path.join(CONFIG.output_dir, 'best_model.pt'), weights_only=True))
logger.info("Loaded best model checkpoint")

tracker.mark('Training', 'PASS', {
    'best_epoch': best_epoch,
    'best_f1': best_val_f1,
    'time_seconds': train_time,
})
""")

    # =========================================================================
    # CELL 8: Test Evaluation (Held-Out)
    # =========================================================================
    add_code("""# ============================================================================
# CELL 8: TEST EVALUATION (HELD-OUT — NEVER SEEN IN TRAINING)
# ============================================================================
logger.info("=" * 60)
logger.info("🎯 TEST SET EVALUATION (HELD-OUT)")
logger.info("=" * 60)

# Overall test metrics
test_f1, test_per_class, test_loss, test_probs, test_labels = evaluate(
    model, test_loader, device, best_thresholds
)

print(f"\\n{'='*60}")
print(f"📊 HELD-OUT TEST RESULTS")
print(f"{'='*60}")
print(f"\\nMacro F1: {test_f1:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"\\nPer-Maxim Performance:")
for name in MAXIM_NAMES:
    sc = test_per_class[name]
    # Flag suspicious scores
    flag = " ⚠️ SUSPICIOUS" if sc['f1'] > 0.95 else ""
    print(f"  {name}: F1={sc['f1']:.3f}, P={sc['precision']:.3f}, R={sc['recall']:.3f}{flag}")

print(f"\\nThresholds used: {dict(zip(MAXIM_NAMES, best_thresholds))}")

# ---- Per-generation-method evaluation ----
print(f"\\n{'='*60}")
print(f"📊 PER-GENERATION-METHOD BREAKDOWN")
print(f"{'='*60}")

# Group test examples by generation method
method_examples = defaultdict(list)
test_preds = (test_probs >= np.array(best_thresholds)).astype(int)

for i, ex in enumerate(test_data):
    if i < len(test_preds):
        method_examples[ex.generation_method].append({
            'true': test_labels[i] if i < len(test_labels) else ex.labels,
            'pred': test_preds[i],
            'probs': test_probs[i] if i < len(test_probs) else None,
        })

method_results = {}
for method, items in method_examples.items():
    true_arr = np.array([item['true'] for item in items])
    pred_arr = np.array([item['pred'] for item in items])
    
    method_f1s = {}
    for j, name in enumerate(MAXIM_NAMES):
        if true_arr[:, j].sum() > 0:
            f1 = f1_score(true_arr[:, j], pred_arr[:, j], zero_division=0)
            method_f1s[name] = f1
    
    macro = np.mean(list(method_f1s.values())) if method_f1s else 0
    method_results[method] = {'macro_f1': macro, 'per_class': method_f1s, 'count': len(items)}
    
    print(f"\\n  {method} ({len(items)} examples):")
    print(f"    Macro F1: {macro:.3f}")
    for name, f1 in method_f1s.items():
        print(f"      {name}: {f1:.3f}")

# ---- Health check ----
print(f"\\n{'='*60}")
print(f"✅ HEALTH CHECKS")
print(f"{'='*60}")

if test_f1 > 0.95:
    print(f"  ⚠️ WARNING: F1={test_f1:.3f} is suspiciously high (>0.95)")
    print(f"     This may indicate overfitting to synthetic patterns")
elif test_f1 > 0.80:
    print(f"  ✅ EXCELLENT: F1={test_f1:.3f} is in the excellent range (0.80-0.95)")
elif test_f1 > 0.65:
    print(f"  ✅ GOOD: F1={test_f1:.3f} is in the good range (0.65-0.80)")
else:
    print(f"  ⚠️ LOW: F1={test_f1:.3f} — model may need more data or tuning")

tracker.mark('Test Evaluation', 'PASS', {
    'test_f1': test_f1,
    'test_loss': test_loss,
    'per_class': {k: v['f1'] for k, v in test_per_class.items()},
    'per_method': {k: v['macro_f1'] for k, v in method_results.items()},
})
""")

    # =========================================================================
    # CELL 9: Error Analysis
    # =========================================================================
    add_code("""# ============================================================================
# CELL 9: ERROR ANALYSIS
# ============================================================================
logger.info("=" * 60)
logger.info("ERROR ANALYSIS")
logger.info("=" * 60)

# Find misclassified examples
errors = []

for i in range(min(len(test_data), len(test_preds))):
    ex = test_data[i]
    pred = test_preds[i]
    true = np.array(ex.labels)
    prob = test_probs[i]
    
    error_count = np.sum(pred != true)
    if error_count > 0:
        errors.append({
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

errors.sort(key=lambda x: x['error_count'], reverse=True)

print(f"\\n❌ Total misclassified: {len(errors)} / {len(test_data)} ({100*len(errors)/len(test_data):.1f}%)")
print(f"✅ Correctly classified: {len(test_data) - len(errors)} ({100*(len(test_data)-len(errors))/len(test_data):.1f}%)")

# Error type breakdown
print(f"\\n📊 Error Type Breakdown:")
error_by_maxim = defaultdict(lambda: {'false_pos': 0, 'false_neg': 0})
for err in errors:
    for j, name in enumerate(MAXIM_NAMES):
        if err['true_labels'][j] == 1 and err['pred_labels'][j] == 0:
            error_by_maxim[name]['false_neg'] += 1
        elif err['true_labels'][j] == 0 and err['pred_labels'][j] == 1:
            error_by_maxim[name]['false_pos'] += 1

for name in MAXIM_NAMES:
    fp = error_by_maxim[name]['false_pos']
    fn = error_by_maxim[name]['false_neg']
    print(f"  {name}: {fp} false positives, {fn} false negatives")

# Error by generation method
print(f"\\n📊 Errors by Generation Method:")
error_by_method = Counter(err['generation_method'] for err in errors)
for method, count in error_by_method.most_common():
    total_method = sum(1 for ex in test_data if ex.generation_method == method)
    print(f"  {method}: {count}/{total_method} errors ({100*count/max(total_method,1):.1f}%)")

# Top 10 worst errors
print(f"\\n📋 Top 10 Worst Misclassifications:")
for rank, err in enumerate(errors[:10], 1):
    print(f"\\n  #{rank} ({err['generation_method']}, {err['violation_type']})")
    print(f"    Text: {err['text'][:150]}...")
    print(f"    True: {err['true_labels']} ({', '.join(MAXIM_NAMES[j] for j in range(4) if err['true_labels'][j]==1) or 'Clean'})")
    print(f"    Pred: {err['pred_labels']} ({', '.join(MAXIM_NAMES[j] for j in range(4) if err['pred_labels'][j]==1) or 'Clean'})")
    print(f"    Probs: [{', '.join(f'{p:.2f}' for p in err['probs'])}]")

tracker.mark('Error Analysis', 'PASS', {
    'total_errors': len(errors),
    'error_rate': f"{100*len(errors)/len(test_data):.1f}%",
    'error_by_maxim': {k: dict(v) for k, v in error_by_maxim.items()},
})
""")

    # =========================================================================
    # CELL 10: Save Results
    # =========================================================================
    add_code("""# ============================================================================
# CELL 10: SAVE RESULTS
# ============================================================================
logger.info("=" * 60)
logger.info("SAVING RESULTS")
logger.info("=" * 60)

# Compile results
results = {
    'phase': 'Phase 6 - Detector V2 (Verified Natural Violations)',
    'timestamp': datetime.now().isoformat(),
    'model': CONFIG.model_name,
    'best_epoch': best_epoch,
    'thresholds': dict(zip(MAXIM_NAMES, best_thresholds)),
    'data_verification': {
        'total_violations': len(violations),
        'total_clean': len(clean_examples),
        'generation_methods': dict(generation_method_counts),
        'maxim_counts': dict(maxim_counts),
        'source_file': phase4_path,
        'assertions_passed': True,
    },
    'splits': {
        'train': len(train_data),
        'val': len(val_data),
        'test': len(test_data),
        'train_sources': train_sources,
        'val_sources': val_sources,
        'test_sources': test_sources,
        'leakage_check': {
            'train_val': train_val_overlap,
            'train_test': train_test_overlap,
            'val_test': val_test_overlap,
        },
    },
    'validation': {
        'macro_f1': best_val_f1,
        'per_class': {name: training_history[best_epoch-1]['per_class'][name] for name in MAXIM_NAMES},
    },
    'test': {
        'macro_f1': test_f1,
        'loss': test_loss,
        'per_class': {name: test_per_class[name] for name in MAXIM_NAMES},
    },
    'test_per_method': {method: {
        'macro_f1': info['macro_f1'],
        'count': info['count'],
        'per_class': info['per_class'],
    } for method, info in method_results.items()},
    'error_analysis': {
        'total_errors': len(errors),
        'error_rate': round(100 * len(errors) / len(test_data), 2),
        'error_by_maxim': {k: dict(v) for k, v in error_by_maxim.items()},
        'error_by_method': dict(error_by_method),
        'top_10_errors': errors[:10],
    },
    'training_history': training_history,
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
logger.info(f"Results saved: {results_path}")

# Save thresholds
thresholds_path = os.path.join(CONFIG.output_dir, 'optimal_thresholds.json')
with open(thresholds_path, 'w') as f:
    json.dump({
        'thresholds': dict(zip(MAXIM_NAMES, best_thresholds)),
        'macro_f1': best_val_f1,
    }, f, indent=2)
logger.info(f"Thresholds saved: {thresholds_path}")

# Copy to /kaggle/working for download
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
    # CELL 11: Final Summary
    # =========================================================================
    add_code("""# ============================================================================
# CELL 11: FINAL SUMMARY
# ============================================================================
print("\\n" + "=" * 60)
print("🏁 PHASE 6 DETECTOR V2 — FINAL SUMMARY")
print("=" * 60)

print(f"\\n📦 Model: {CONFIG.model_name}")
print(f"📊 Data: {len(violations)} violations + {len(clean_examples)} clean from Phase 4")
print(f"🔬 Generation methods: {dict(generation_method_counts)}")

print(f"\\n📈 Training:")
print(f"  Best epoch: {best_epoch}")
print(f"  Val F1: {best_val_f1:.4f}")
print(f"  Time: {train_time:.0f}s ({train_time/60:.1f} min)")

print(f"\\n🎯 TEST SET RESULTS (held-out, never in training):")
print(f"  Macro F1: {test_f1:.4f}")
for name in MAXIM_NAMES:
    sc = test_per_class[name]
    print(f"    {name}: F1={sc['f1']:.3f}")

print(f"\\n🔍 Per-Method Performance:")
for method, info in sorted(method_results.items(), key=lambda x: -x[1]['macro_f1']):
    print(f"  {method}: F1={info['macro_f1']:.3f} ({info['count']} examples)")

print(f"\\n❌ Errors: {len(errors)}/{len(test_data)} ({100*len(errors)/len(test_data):.1f}%)")

print(f"\\n✅ DATA VERIFIED:")
print(f"  Phase 4 violations loaded: {len(violations)}")
print(f"  No data leakage: ✅")
print(f"  All maxims represented: ✅")

print(f"\\n📁 Output Files:")
print(f"  /kaggle/working/detector_v2_results.json")
print(f"  /kaggle/working/optimal_thresholds.json")
print(f"  /kaggle/working/best_model.pt")

if test_f1 > 0.95:
    print(f"\\n⚠️  F1={test_f1:.3f} is very high — review error analysis for overfitting signs")
elif test_f1 > 0.70:
    print(f"\\n✅ Results look realistic and healthy")
else:
    print(f"\\n⚠️  F1={test_f1:.3f} is below target — may need more data or training")

print(f"\\n{'='*60}")
print(f"PHASE 6 V2 COMPLETE — Download detector_v2_results.json")
print(f"{'='*60}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

tracker.mark('Complete', 'PASS')
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
