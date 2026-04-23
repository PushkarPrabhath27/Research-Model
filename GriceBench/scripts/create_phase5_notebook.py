#!/usr/bin/env python3
"""Generate Phase 5 DPO Training Notebook for Kaggle."""
import json

def md(source):
    return {"cell_type": "markdown", "source": source, "metadata": {}}

def code(source):
    return {"cell_type": "code", "source": source, "metadata": {"trusted": True}, "outputs": [], "execution_count": None}

cells = []

# ============================================================
# CELL 0: MARKDOWN HEADER
# ============================================================
cells.append(md(
"""# Phase 5: DPO Training with Human-Annotated Preferences

**Version**: 3.0 (Production-Grade)

## Architecture
- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct (QLoRA 4-bit)
- **Method**: Direct Preference Optimization (DPO)
- **Data**: 500 human-annotated preference pairs (301 usable)
- **GPU Target**: 90%+ utilization on T4

## Key Design Decisions
- **QLoRA**: 4-bit quantized base + LoRA adapters = fits T4 easily with large batches
- **Gradient Accumulation**: Effective batch size of 32 for stable training
- **Cosine LR Schedule**: With warmup for smooth convergence
- **Per-sample logging**: Track every preference pair's contribution

## Execution Order
Run ALL cells sequentially. Do NOT skip cells."""
))

# ============================================================
# CELL 1: ENVIRONMENT SETUP
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 1: Environment Setup & Dependencies
# ============================================================================
import subprocess
import sys
import time

cell_start = time.time()

# Install required packages
packages = [
    "trl>=0.12.0",
    "peft>=0.14.0",
    "bitsandbytes>=0.45.0",
    "accelerate>=1.2.0",
    "datasets>=3.2.0",
    "transformers>=4.47.0",
    "scipy",
]

print("Installing dependencies...")
for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("Dependencies installed.\\n")

# Core imports
import os
import json
import random
import logging
import gc
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# HuggingFace imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import DPOConfig, DPOTrainer
from datasets import Dataset as HFDataset

# Logging setup
class ColoredFormatter(logging.Formatter):
    COLORS = {'DEBUG': '\\033[36m', 'INFO': '\\033[32m', 'WARNING': '\\033[33m', 'ERROR': '\\033[31m'}
    RESET = '\\033[0m'
    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

logger = logging.getLogger('Phase5DPO')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(ColoredFormatter('%(levelname)s | %(message)s'))
    logger.addHandler(ch)

    os.makedirs('/kaggle/working/logs', exist_ok=True)
    fh = logging.FileHandler(f'/kaggle/working/logs/dpo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)

# Checkpoint tracker
class CheckpointTracker:
    def __init__(self):
        self.checkpoints = {}
        self.t0 = time.time()

    def mark(self, name, status='PASS', details=None):
        elapsed = time.time() - self.t0
        self.checkpoints[name] = {'status': status, 'time': elapsed, 'details': details or {}}
        icon = '✅' if status == 'PASS' else '❌' if status == 'FAIL' else '⚠️'
        logger.info(f"{icon} CHECKPOINT [{name}]: {status} ({elapsed:.1f}s)")
        if details:
            for k, v in details.items():
                logger.info(f"   {k}: {v}")

    def summary(self):
        logger.info("=" * 60)
        logger.info("CHECKPOINT SUMMARY")
        logger.info("=" * 60)
        for name, data in self.checkpoints.items():
            icon = '✅' if data['status'] == 'PASS' else '❌'
            logger.info(f"{icon} {name}: {data['status']} ({data['time']:.1f}s)")

tracker = CheckpointTracker()

# Seeds
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # Faster convolutions

set_seeds(42)

# GPU check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"VRAM: {gpu_mem:.1f} GB")
    # Enable TF32 for faster matmuls on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    raise RuntimeError("GPU required for DPO training!")

tracker.mark('Environment Setup', 'PASS', {'device': str(device), 'gpu': gpu_name})
print(f"\\nCELL 1 COMPLETE ({time.time()-cell_start:.1f}s): Environment ready")
"""
))

# ============================================================
# CELL 2: CONFIGURATION
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 2: Configuration
# ============================================================================

@dataclass
class DPOConfig_Custom:
    # Paths
    data_path: str = '/kaggle/input/datasets/pushkarprabhath/gricebench-dpo-annotations/tier1_hard_pairs_FULLY_ANNOTATED.json'
    output_dir: str = '/kaggle/working/dpo_output'

    # Model
    model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct'
    max_length: int = 512
    max_prompt_length: int = 384

    # QLoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    use_4bit: bool = True

    # Training - optimized for T4 GPU at 90% utilization
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8   # effective batch = 32
    learning_rate: float = 5e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    beta: float = 0.1                      # DPO temperature parameter

    # Precision
    fp16: bool = False
    bf16: bool = True                      # Better for DPO stability

    # Data
    val_ratio: float = 0.15
    min_preference_strength: str = 'slight'  # Include both 'slight' and 'much'

    # Logging
    logging_steps: int = 5
    eval_steps: int = 25
    save_steps: int = 50

    def __post_init__(self):
        self.effective_batch = self.per_device_batch_size * self.gradient_accumulation_steps
        os.makedirs(self.output_dir, exist_ok=True)

CONFIG = DPOConfig_Custom()

# Check if bf16 is supported, fallback to fp16
if CONFIG.bf16 and not torch.cuda.is_bf16_supported():
    logger.warning("bf16 not supported on this GPU, falling back to fp16")
    CONFIG.bf16 = False
    CONFIG.fp16 = True

logger.info("Configuration:")
for k, v in vars(CONFIG).items():
    logger.info(f"  {k}: {v}")

tracker.mark('Configuration', 'PASS')
print("\\nCELL 2 COMPLETE: Configuration set")
"""
))

# ============================================================
# CELL 3: LOAD & VALIDATE ANNOTATED DATA
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 3: Load & Validate Annotated Data
# ============================================================================

logger.info("=" * 60)
logger.info("LOADING ANNOTATED DATA")
logger.info("=" * 60)

# Try multiple paths (Kaggle dataset or local)
possible_paths = [
    CONFIG.data_path,
    '/kaggle/input/datasets/pushkarprabhath/gricebench-dpo-annotations/tier1_hard_pairs_FULLY_ANNOTATED.json',
    '/kaggle/input/gricebench-dpo-annotations/tier1_hard_pairs_FULLY_ANNOTATED.json',
    '/kaggle/input/gricebench-dpo/tier1_hard_pairs_FULLY_ANNOTATED.json',
]

raw_data = None
for path in possible_paths:
    if os.path.exists(path):
        logger.info(f"Found data at: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        break

if raw_data is None:
    # List available files for debugging
    logger.error("Data file not found! Available files:")
    for root_dir in ['/kaggle/input']:
        if os.path.exists(root_dir):
            for dirpath, dirnames, filenames in os.walk(root_dir):
                for fn in filenames:
                    if fn.endswith('.json'):
                        logger.error(f"  {os.path.join(dirpath, fn)}")
    raise FileNotFoundError("tier1_hard_pairs_FULLY_ANNOTATED.json not found in any expected location")

logger.info(f"Total records loaded: {len(raw_data)}")

# Validate structure
required_keys = {'id', 'context', 'response_A', 'response_B', 'preference', 'reason', 'annotated'}
sample = raw_data[0]
missing = required_keys - set(sample.keys())
if missing:
    raise ValueError(f"Missing required keys: {missing}")

# Check all are annotated
annotated_count = sum(1 for d in raw_data if d.get('annotated', False))
logger.info(f"Annotated: {annotated_count}/{len(raw_data)}")
assert annotated_count == len(raw_data), f"Not all records annotated! Only {annotated_count}/{len(raw_data)}"

# Preference distribution
pref_counts = Counter(d['preference'] for d in raw_data)
logger.info("\\nPreference Distribution:")
for pref, count in pref_counts.most_common():
    pct = 100 * count / len(raw_data)
    logger.info(f"  {pref}: {count} ({pct:.1f}%)")

# Reason analysis
reason_counts = Counter(d.get('reason', '') for d in raw_data)
logger.info(f"\\nUnique reasons: {len(reason_counts)}")
logger.info("Top 5 reasons:")
for reason, count in reason_counts.most_common(5):
    logger.info(f"  [{count}x] {reason[:80]}...")

# Maxim mentions in reasons
maxim_names = ['quantity', 'quality', 'relation', 'manner']
maxim_mentions = {m: 0 for m in maxim_names}
for d in raw_data:
    reason_lower = d.get('reason', '').lower()
    for m in maxim_names:
        if m in reason_lower:
            maxim_mentions[m] += 1

logger.info("\\nMaxim Mentions in Reasons:")
for m, count in sorted(maxim_mentions.items(), key=lambda x: -x[1]):
    logger.info(f"  {m.capitalize()}: {count}")

tracker.mark('Data Loaded', 'PASS', {
    'total': len(raw_data),
    'annotated': annotated_count,
    'preferences': dict(pref_counts)
})
print(f"\\nCELL 3 COMPLETE: {len(raw_data)} annotated pairs loaded")
"""
))

# ============================================================
# CELL 4: CONVERT TO DPO FORMAT
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 4: Convert to DPO Format (Chosen / Rejected)
# ============================================================================

logger.info("=" * 60)
logger.info("CONVERTING TO DPO FORMAT")
logger.info("=" * 60)

PREFERENCE_MAP = {
    'A_much': ('A', 'strong'),
    'A_slight': ('A', 'weak'),
    'B_much': ('B', 'strong'),
    'B_slight': ('B', 'weak'),
    'equal': (None, 'none'),
}

dpo_examples = []
skipped = 0
errors = []

for item in raw_data:
    try:
        pref = item['preference']
        winner, strength = PREFERENCE_MAP.get(pref, (None, 'none'))

        if winner is None:
            skipped += 1
            continue

        context = item['context'].strip()
        resp_a = item['response_A'].strip()
        resp_b = item['response_B'].strip()
        reason = item.get('reason', '').strip()

        # Skip if responses are too short or empty
        if len(resp_a) < 10 or len(resp_b) < 10:
            skipped += 1
            continue

        # Skip if responses are nearly identical
        if resp_a[:50] == resp_b[:50] and pref != 'equal':
            skipped += 1
            continue

        # Build prompt from context
        prompt = f"Continue the following conversation naturally, following Gricean maxims (be relevant, truthful, clear, and appropriately informative):\\n\\n{context}\\n\\nResponse:"

        # Determine chosen and rejected
        if winner == 'A':
            chosen = resp_a
            rejected = resp_b
        else:
            chosen = resp_b
            rejected = resp_a

        dpo_examples.append({
            'prompt': prompt,
            'chosen': chosen,
            'rejected': rejected,
            'strength': strength,
            'reason': reason,
            'id': item['id'],
        })

    except Exception as e:
        errors.append(f"{item.get('id', '?')}: {str(e)}")

logger.info(f"\\nConversion Results:")
logger.info(f"  DPO pairs created: {len(dpo_examples)}")
logger.info(f"  Skipped (equal/invalid): {skipped}")
logger.info(f"  Errors: {len(errors)}")

if errors:
    logger.warning("Sample errors:")
    for e in errors[:3]:
        logger.warning(f"  {e}")

# Strength distribution
strength_counts = Counter(ex['strength'] for ex in dpo_examples)
logger.info(f"\\nStrength Distribution:")
for s, c in strength_counts.most_common():
    logger.info(f"  {s}: {c}")

# Sample a DPO example
if dpo_examples:
    sample = dpo_examples[0]
    logger.info(f"\\nSample DPO Pair:")
    logger.info(f"  Prompt: {sample['prompt'][:120]}...")
    logger.info(f"  Chosen: {sample['chosen'][:80]}...")
    logger.info(f"  Rejected: {sample['rejected'][:80]}...")
    logger.info(f"  Reason: {sample['reason']}")

tracker.mark('DPO Conversion', 'PASS', {
    'dpo_pairs': len(dpo_examples),
    'skipped': skipped
})
print(f"\\nCELL 4 COMPLETE: {len(dpo_examples)} DPO pairs ready")
"""
))

# ============================================================
# CELL 5: TRAIN/VAL SPLIT
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 5: Stratified Train/Val Split
# ============================================================================

logger.info("=" * 60)
logger.info("CREATING TRAIN/VAL SPLIT")
logger.info("=" * 60)

random.seed(42)

# Stratify by strength (strong vs weak preferences)
strong = [ex for ex in dpo_examples if ex['strength'] == 'strong']
weak = [ex for ex in dpo_examples if ex['strength'] == 'weak']

random.shuffle(strong)
random.shuffle(weak)

# Split each group
val_strong_n = max(1, int(len(strong) * CONFIG.val_ratio))
val_weak_n = max(1, int(len(weak) * CONFIG.val_ratio)) if weak else 0

val_data = strong[:val_strong_n] + weak[:val_weak_n]
train_data = strong[val_strong_n:] + weak[val_weak_n:]

random.shuffle(train_data)
random.shuffle(val_data)

logger.info(f"Split Results:")
logger.info(f"  Train: {len(train_data)}")
logger.info(f"  Val: {len(val_data)}")
logger.info(f"  Train strong: {sum(1 for x in train_data if x['strength']=='strong')}")
logger.info(f"  Train weak: {sum(1 for x in train_data if x['strength']=='weak')}")
logger.info(f"  Val strong: {sum(1 for x in val_data if x['strength']=='strong')}")
logger.info(f"  Val weak: {sum(1 for x in val_data if x['strength']=='weak')}")

# Convert to HuggingFace Dataset format
def to_hf_dataset(examples):
    return HFDataset.from_dict({
        'prompt': [ex['prompt'] for ex in examples],
        'chosen': [ex['chosen'] for ex in examples],
        'rejected': [ex['rejected'] for ex in examples],
    })

train_dataset = to_hf_dataset(train_data)
val_dataset = to_hf_dataset(val_data)

logger.info(f"\\nHF Dataset columns: {train_dataset.column_names}")
logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Val dataset size: {len(val_dataset)}")

tracker.mark('Train/Val Split', 'PASS', {
    'train': len(train_data),
    'val': len(val_data)
})
print(f"\\nCELL 5 COMPLETE: Train={len(train_data)}, Val={len(val_data)}")
"""
))

# ============================================================
# CELL 6: LOAD MODEL + TOKENIZER WITH QLoRA
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 6: Load Model & Tokenizer with QLoRA (4-bit)
# ============================================================================

logger.info("=" * 60)
logger.info("LOADING MODEL WITH QLoRA")
logger.info("=" * 60)

cell_start = time.time()

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer
logger.info(f"Loading tokenizer: {CONFIG.model_name}")
tokenizer = AutoTokenizer.from_pretrained(
    CONFIG.model_name,
    trust_remote_code=True,
    padding_side='left',
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

logger.info(f"Vocab size: {tokenizer.vocab_size}")
logger.info(f"Pad token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")

# Load model in 4-bit
logger.info(f"\\nLoading model: {CONFIG.model_name} (4-bit quantized)")
model = AutoModelForCausalLM.from_pretrained(
    CONFIG.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if CONFIG.bf16 else torch.float16,
    attn_implementation="eager",  # Ensure compatibility
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# LoRA config - target all linear layers for maximum expressivity
lora_config = LoraConfig(
    r=CONFIG.lora_r,
    lora_alpha=CONFIG.lora_alpha,
    lora_dropout=CONFIG.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules="all-linear",
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Model stats
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
train_pct = 100.0 * trainable_params / total_params

logger.info(f"\\nModel Parameters:")
logger.info(f"  Total: {total_params:,}")
logger.info(f"  Trainable (LoRA): {trainable_params:,}")
logger.info(f"  Trainable %: {train_pct:.2f}%")

# VRAM usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"\\nVRAM Usage (after model load):")
    logger.info(f"  Allocated: {allocated:.2f} GB")
    logger.info(f"  Reserved: {reserved:.2f} GB")
    logger.info(f"  Total: {total_vram:.1f} GB")
    logger.info(f"  Utilization: {100*allocated/total_vram:.1f}%")

tracker.mark('Model Loaded', 'PASS', {
    'total_params': f"{total_params:,}",
    'trainable_params': f"{trainable_params:,}",
    'trainable_pct': f"{train_pct:.2f}%",
    'vram_gb': f"{allocated:.2f}",
})

load_time = time.time() - cell_start
print(f"\\nCELL 6 COMPLETE ({load_time:.1f}s): Model loaded with QLoRA")
"""
))

# ============================================================
# CELL 7: DPO TRAINING SETUP
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 7: DPO Training Setup
# ============================================================================

logger.info("=" * 60)
logger.info("SETTING UP DPO TRAINER")
logger.info("=" * 60)

# Calculate training steps
steps_per_epoch = len(train_dataset) // (CONFIG.per_device_batch_size * CONFIG.gradient_accumulation_steps)
total_steps = steps_per_epoch * CONFIG.num_epochs
warmup_steps = int(total_steps * CONFIG.warmup_ratio)

logger.info(f"Training Plan:")
logger.info(f"  Train examples: {len(train_dataset)}")
logger.info(f"  Batch size: {CONFIG.per_device_batch_size}")
logger.info(f"  Gradient accumulation: {CONFIG.gradient_accumulation_steps}")
logger.info(f"  Effective batch: {CONFIG.effective_batch}")
logger.info(f"  Steps per epoch: {steps_per_epoch}")
logger.info(f"  Total steps: {total_steps}")
logger.info(f"  Warmup steps: {warmup_steps}")
logger.info(f"  DPO beta: {CONFIG.beta}")

# DPO Training Arguments
training_args = DPOConfig(
    output_dir=CONFIG.output_dir,

    # Batch & accumulation
    per_device_train_batch_size=CONFIG.per_device_batch_size,
    per_device_eval_batch_size=CONFIG.per_device_batch_size,
    gradient_accumulation_steps=CONFIG.gradient_accumulation_steps,

    # Learning rate
    learning_rate=CONFIG.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,
    weight_decay=CONFIG.weight_decay,
    max_grad_norm=CONFIG.max_grad_norm,

    # Epochs
    num_train_epochs=CONFIG.num_epochs,

    # Precision
    fp16=CONFIG.fp16,
    bf16=CONFIG.bf16,

    # DPO specific
    beta=CONFIG.beta,
    max_length=CONFIG.max_length,
    max_prompt_length=CONFIG.max_prompt_length,
    loss_type="sigmoid",  # Standard DPO loss

    # Logging & saving
    logging_steps=CONFIG.logging_steps,
    eval_strategy="steps",
    eval_steps=CONFIG.eval_steps,
    save_strategy="steps",
    save_steps=CONFIG.save_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Performance
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    optim="paged_adamw_8bit",      # Memory-efficient optimizer
    remove_unused_columns=False,

    # Reporting
    report_to="none",
    run_name="gricebench_dpo_phase5",
)

# Disable tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
)

logger.info("\\nDPO Trainer created successfully")

# VRAM after trainer setup
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1e9
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"VRAM after trainer setup: {allocated:.2f} GB ({100*allocated/total_vram:.1f}%)")

tracker.mark('DPO Trainer Setup', 'PASS', {
    'total_steps': total_steps,
    'effective_batch': CONFIG.effective_batch,
})
print("\\nCELL 7 COMPLETE: DPO Trainer ready")
"""
))

# ============================================================
# CELL 8: TRAINING
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 8: DPO Training Loop
# ============================================================================

logger.info("=" * 60)
logger.info("STARTING DPO TRAINING")
logger.info("=" * 60)

train_start = time.time()

# GPU monitoring callback
class GPUMonitor:
    def __init__(self):
        self.peak_util = 0
        self.readings = []

    def log(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            util = 100 * allocated / total
            self.readings.append(util)
            self.peak_util = max(self.peak_util, util)
            return util
        return 0

gpu_monitor = GPUMonitor()

# Pre-training eval
logger.info("\\nPre-training evaluation...")
try:
    pre_eval = dpo_trainer.evaluate()
    logger.info(f"Pre-training eval loss: {pre_eval.get('eval_loss', 'N/A')}")
except Exception as e:
    logger.warning(f"Pre-training eval failed (normal for first run): {e}")

# Train!
logger.info("\\n" + "=" * 60)
logger.info("TRAINING IN PROGRESS...")
logger.info("=" * 60)

train_result = dpo_trainer.train()

train_time = time.time() - train_start

# Log GPU utilization
gpu_util = gpu_monitor.log()
logger.info(f"\\nGPU VRAM utilization: {gpu_util:.1f}%")

# Training metrics
logger.info(f"\\n{'='*60}")
logger.info(f"TRAINING COMPLETE")
logger.info(f"{'='*60}")
logger.info(f"Total training time: {train_time:.1f}s ({train_time/60:.1f}m)")
logger.info(f"Train loss: {train_result.training_loss:.4f}")

# Log training metrics
metrics = train_result.metrics
for key, value in metrics.items():
    logger.info(f"  {key}: {value}")

# VRAM peak
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated(0) / 1e9
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    peak_pct = 100 * peak_mem / total_vram
    logger.info(f"\\nPeak VRAM: {peak_mem:.2f} GB ({peak_pct:.1f}%)")

tracker.mark('DPO Training', 'PASS', {
    'train_loss': f"{train_result.training_loss:.4f}",
    'time': f"{train_time:.1f}s",
    'peak_vram_pct': f"{peak_pct:.1f}%"
})
print(f"\\nCELL 8 COMPLETE: Training done in {train_time:.1f}s, loss={train_result.training_loss:.4f}")
"""
))

# ============================================================
# CELL 9: EVALUATION
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 9: Post-Training Evaluation
# ============================================================================

logger.info("=" * 60)
logger.info("POST-TRAINING EVALUATION")
logger.info("=" * 60)

# Evaluate on validation set
eval_results = dpo_trainer.evaluate()

logger.info("\\nValidation Metrics:")
for key, value in eval_results.items():
    logger.info(f"  {key}: {value}")

# Generate sample responses to compare
logger.info("\\n" + "=" * 60)
logger.info("SAMPLE GENERATION COMPARISON")
logger.info("=" * 60)

model.eval()

# Take 5 samples from validation set
n_samples = min(5, len(val_data))
sample_indices = random.sample(range(len(val_data)), n_samples)

generation_results = []

for idx in sample_indices:
    sample = val_data[idx]
    prompt = sample['prompt']

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=CONFIG.max_prompt_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    result = {
        'prompt_snippet': prompt[:100],
        'generated': generated[:200],
        'original_chosen': sample['chosen'][:200],
        'original_rejected': sample['rejected'][:200],
        'reason': sample['reason'],
    }
    generation_results.append(result)

    logger.info(f"\\n--- Sample {idx} ---")
    logger.info(f"Context: {prompt[:100]}...")
    logger.info(f"Generated: {generated[:150]}...")
    logger.info(f"Chosen was: {sample['chosen'][:100]}...")
    logger.info(f"Rejected was: {sample['rejected'][:100]}...")
    logger.info(f"Reason: {sample['reason']}")

tracker.mark('Post-Training Eval', 'PASS', {
    'eval_loss': f"{eval_results.get('eval_loss', 'N/A')}",
    'samples_generated': n_samples,
})
print(f"\\nCELL 9 COMPLETE: Evaluation done, eval_loss={eval_results.get('eval_loss', 'N/A')}")
"""
))

# ============================================================
# CELL 10: PREFERENCE ACCURACY
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 10: Compute Preference Accuracy (Reward Margin)
# ============================================================================

logger.info("=" * 60)
logger.info("COMPUTING PREFERENCE ACCURACY")
logger.info("=" * 60)

# For each validation pair, check if model assigns higher probability to chosen vs rejected
correct = 0
total = 0
margins = []

model.eval()

for i, sample in enumerate(val_data):
    try:
        prompt = sample['prompt']
        chosen = sample['chosen']
        rejected = sample['rejected']

        # Tokenize prompt + chosen
        chosen_input = tokenizer(
            prompt + chosen,
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG.max_length,
        )
        chosen_input = {k: v.to(device) for k, v in chosen_input.items()}

        # Tokenize prompt + rejected
        rejected_input = tokenizer(
            prompt + rejected,
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG.max_length,
        )
        rejected_input = {k: v.to(device) for k, v in rejected_input.items()}

        with torch.no_grad():
            chosen_out = model(**chosen_input)
            rejected_out = model(**rejected_input)

            # Get mean log probability (normalized by length)
            chosen_logprob = -chosen_out.loss.item() if hasattr(chosen_out, 'loss') and chosen_out.loss is not None else 0
            rejected_logprob = -rejected_out.loss.item() if hasattr(rejected_out, 'loss') and rejected_out.loss is not None else 0

            # Model prefers chosen if its loss is lower (higher log prob)
            chosen_loss = chosen_out.loss.item() if chosen_out.loss is not None else float('inf')
            rejected_loss = rejected_out.loss.item() if rejected_out.loss is not None else float('inf')

            margin = rejected_loss - chosen_loss  # positive = model prefers chosen
            margins.append(margin)

            if chosen_loss < rejected_loss:
                correct += 1
            total += 1

    except Exception as e:
        logger.debug(f"Sample {i} error: {e}")
        continue

    if (i + 1) % 10 == 0:
        logger.info(f"  Processed {i+1}/{len(val_data)} (accuracy so far: {100*correct/total:.1f}%)")

accuracy = 100 * correct / total if total > 0 else 0
avg_margin = np.mean(margins) if margins else 0

logger.info(f"\\nPreference Accuracy Results:")
logger.info(f"  Correct: {correct}/{total}")
logger.info(f"  Accuracy: {accuracy:.1f}%")
logger.info(f"  Average margin: {avg_margin:.4f}")
logger.info(f"  Margin std: {np.std(margins):.4f}" if margins else "  No margins")

# Interpret results
if accuracy > 70:
    logger.info("\\n  ✅ STRONG: Model clearly prefers chosen responses")
elif accuracy > 55:
    logger.info("\\n  ⚠️ MODERATE: Model shows some preference learning")
elif accuracy > 50:
    logger.info("\\n  ⚠️ WEAK: Model barely distinguishes preferences")
else:
    logger.info("\\n  ❌ FAILED: Model does not prefer chosen responses")

tracker.mark('Preference Accuracy', 'PASS' if accuracy > 55 else 'WARN', {
    'accuracy': f"{accuracy:.1f}%",
    'avg_margin': f"{avg_margin:.4f}",
})
print(f"\\nCELL 10 COMPLETE: Preference accuracy = {accuracy:.1f}%")
"""
))

# ============================================================
# CELL 11: SAVE MODEL & RESULTS
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 11: Save Model, Adapter, and Results
# ============================================================================

logger.info("=" * 60)
logger.info("SAVING OUTPUTS")
logger.info("=" * 60)

# Save LoRA adapter (small, easy to download)
adapter_path = f"{CONFIG.output_dir}/dpo_adapter"
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
logger.info(f"LoRA adapter saved to: {adapter_path}")

# Calculate adapter size
adapter_size = 0
for root, dirs, files in os.walk(adapter_path):
    for f in files:
        adapter_size += os.path.getsize(os.path.join(root, f))
logger.info(f"Adapter size: {adapter_size/1e6:.1f} MB")

# Save comprehensive results
results = {
    'phase': 'Phase 5 - DPO Training',
    'timestamp': datetime.now().isoformat(),
    'model': CONFIG.model_name,
    'method': 'DPO with QLoRA (4-bit)',
    'data': {
        'total_annotated': len(raw_data),
        'dpo_pairs': len(dpo_examples),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'preference_distribution': dict(pref_counts),
    },
    'hyperparameters': {
        'lora_r': CONFIG.lora_r,
        'lora_alpha': CONFIG.lora_alpha,
        'learning_rate': CONFIG.learning_rate,
        'effective_batch_size': CONFIG.effective_batch,
        'num_epochs': CONFIG.num_epochs,
        'beta': CONFIG.beta,
        'max_length': CONFIG.max_length,
    },
    'training': {
        'train_loss': float(train_result.training_loss),
        'training_time_seconds': float(train_time),
    },
    'evaluation': {
        'eval_loss': float(eval_results.get('eval_loss', 0)),
        'preference_accuracy': float(accuracy),
        'avg_margin': float(avg_margin),
    },
    'generation_samples': generation_results,
    'gpu': {
        'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'peak_vram_gb': float(torch.cuda.max_memory_allocated(0) / 1e9) if torch.cuda.is_available() else 0,
        'total_vram_gb': float(torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0,
    },
}

results_path = f"{CONFIG.output_dir}/dpo_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
logger.info(f"Results saved to: {results_path}")

# Also save to /kaggle/working for easy download
import shutil
shutil.copy2(results_path, '/kaggle/working/dpo_results.json')

# Save training history
history_path = f"{CONFIG.output_dir}/training_history.json"
if hasattr(dpo_trainer, 'state') and dpo_trainer.state.log_history:
    with open(history_path, 'w') as f:
        json.dump(dpo_trainer.state.log_history, f, indent=2, default=str)
    shutil.copy2(history_path, '/kaggle/working/training_history.json')
    logger.info(f"Training history saved: {len(dpo_trainer.state.log_history)} entries")

tracker.mark('Outputs Saved', 'PASS', {
    'adapter_size_mb': f"{adapter_size/1e6:.1f}",
    'results_path': results_path,
})
print(f"\\nCELL 11 COMPLETE: All outputs saved")
"""
))

# ============================================================
# CELL 12: FINAL SUMMARY
# ============================================================
cells.append(code(
"""# ============================================================================
# CELL 12: Final Summary
# ============================================================================

print("\\n" + "=" * 70)
print("PHASE 5: DPO TRAINING COMPLETE")
print("=" * 70)

# Checkpoint summary
tracker.summary()

print("\\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\\n📊 Data:")
print(f"  Annotated pairs: {len(raw_data)}")
print(f"  DPO training pairs: {len(dpo_examples)}")
print(f"  Train / Val: {len(train_data)} / {len(val_data)}")

print(f"\\n🏋️ Training:")
print(f"  Model: {CONFIG.model_name}")
print(f"  Method: QLoRA (r={CONFIG.lora_r}, alpha={CONFIG.lora_alpha})")
print(f"  Final train loss: {train_result.training_loss:.4f}")
print(f"  Training time: {train_time:.1f}s ({train_time/60:.1f} min)")

print(f"\\n📈 Evaluation:")
print(f"  Eval loss: {eval_results.get('eval_loss', 'N/A')}")
print(f"  Preference accuracy: {accuracy:.1f}%")
print(f"  Average margin: {avg_margin:.4f}")

if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\\n🖥️ GPU:")
    print(f"  Peak VRAM: {peak:.2f} / {total:.1f} GB ({100*peak/total:.1f}%)")

print(f"\\n📁 Output Files:")
print(f"  /kaggle/working/dpo_results.json")
print(f"  /kaggle/working/training_history.json")
print(f"  {CONFIG.output_dir}/dpo_adapter/")

print("\\n" + "=" * 70)
print("✅ DOWNLOAD: /kaggle/working/dpo_results.json")
print("✅ DOWNLOAD: /kaggle/working/training_history.json")
print("=" * 70)

# Cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""
))

# ============================================================
# ASSEMBLE NOTEBOOK
# ============================================================

notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
            "mimetype": "text/x-python",
        },
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [
                {
                    "sourceId": 14639592,
                    "sourceType": "datasetVersion",
                    "datasetId": 9329674
                }
            ],
            "dockerImageVersionId": 31260,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells,
}

output_path = r'c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\KAGGLE_PHASE5_DPO_ANNOTATED.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook created: {output_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
