#!/usr/bin/env python3
"""
Phase 5 DPO Training V2 — Optimized Notebook Generator
=======================================================
Fixes from V1:
  1. beta: 0.1 → 0.3 (regularize reward margins)
  2. epochs: 5 → 3 (prevent overfitting)
  3. batch_size: 4 → 8, grad_accum: 8 → 4 (better GPU utilization)
  4. max_length: 512 → 256 (data max is 114 tokens)
  5. EarlyStoppingCallback + load_best_model_at_end
  6. DPODiagnosticCallback for real-time health monitoring
  7. Fixed preference accuracy computation (Cell 10 bug)
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
    add_md("""# Phase 5: DPO Training V2 — Optimized
    
**Fixes applied:**
- β = 0.3 (was 0.1) → prevents reward margin explosion
- 3 epochs (was 5) → stops before overfitting  
- Batch size 8 (was 4) → better GPU utilization
- max_length 256 (was 512) → data max is 114 tokens
- Early stopping + best checkpoint loading
- Real-time diagnostic callback

**Model:** Qwen/Qwen2.5-1.5B-Instruct with QLoRA (4-bit)  
**Data:** 301 annotated DPO pairs from GriceBench""")

    # =========================================================================
    # CELL 1: Environment Setup & GPU Check
    # =========================================================================
    add_code("""# ============================================================
# CELL 1: ENVIRONMENT SETUP
# ============================================================
import subprocess, sys

print("Installing dependencies...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'trl>=0.7.0', 'peft>=0.5.0', 'bitsandbytes>=0.41.0',
    'accelerate>=0.21.0', 'datasets>=2.14.0', 'transformers>=4.35.0',
    'scipy'])
print("Dependencies installed.")

import torch
import os
import gc
import json
import random
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger('Phase5DPO_V2')

# GPU check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Device: {device}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"GPU: {gpu_name}")
    logger.info(f"VRAM: {gpu_mem:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    raise RuntimeError("GPU required for DPO training")

# Progress tracker
class ProgressTracker:
    def __init__(self):
        self.steps = []
        self.start = datetime.now()
    def mark(self, name, status, details=None):
        elapsed = (datetime.now() - self.start).total_seconds()
        self.steps.append({'name': name, 'status': status, 'elapsed': elapsed, 'details': details or {}})
        icon = '✅' if status == 'PASS' else '❌' if status == 'FAIL' else '⏳'
        logger.info(f"{icon} [{elapsed:.0f}s] {name}: {status}")
    def summary(self):
        passed = sum(1 for s in self.steps if s['status'] == 'PASS')
        total = len(self.steps)
        logger.info(f"\\nProgress: {passed}/{total} steps passed")
        return self.steps

tracker = ProgressTracker()
tracker.mark('Environment Setup', 'PASS', {'gpu': gpu_name, 'vram_gb': f'{gpu_mem:.1f}'})
""")

    # =========================================================================
    # CELL 2: Configuration
    # =========================================================================
    add_code("""# ============================================================
# CELL 2: CONFIGURATION — V2 OPTIMIZED
# ============================================================
@dataclass
class DPOConfig_V2:
    # Paths
    data_path: str = '/kaggle/input/datasets/pushkarprabhath/gricebench-dpo-annotations/tier1_hard_pairs_FULLY_ANNOTATED.json'
    output_dir: str = '/kaggle/working/dpo_output_v2'

    # Model
    model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct'

    # QLoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
                                   'gate_proj', 'up_proj', 'down_proj')

    # === V2 CHANGES ===
    # DPO
    beta: float = 0.3              # Was 0.1 → regularize margins
    max_length: int = 256          # Was 512 → data max is 114 tokens
    max_prompt_length: int = 192   # Was 384

    # Training
    learning_rate: float = 5e-5
    num_epochs: int = 3            # Was 5 → stop before overfitting
    per_device_batch: int = 8      # Was 4 → better GPU utilization
    gradient_accumulation: int = 4 # Was 8 → same effective batch (32)
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Evaluation & Early Stopping
    eval_steps: int = 4            # Was 25 → eval every ~half epoch
    early_stopping_patience: int = 3
    
    # Seed
    seed: int = 42

CONFIG = DPOConfig_V2()

logger.info("\\n" + "="*60)
logger.info("DPO V2 CONFIGURATION (OPTIMIZED)")
logger.info("="*60)
logger.info(f"  Model: {CONFIG.model_name}")
logger.info(f"  Beta: {CONFIG.beta} (was 0.1)")
logger.info(f"  Epochs: {CONFIG.num_epochs} (was 5)")
logger.info(f"  Batch: {CONFIG.per_device_batch} × {CONFIG.gradient_accumulation} = {CONFIG.per_device_batch * CONFIG.gradient_accumulation} effective")
logger.info(f"  Max length: {CONFIG.max_length} (was 512)")
logger.info(f"  Eval every: {CONFIG.eval_steps} steps")
logger.info(f"  Early stopping patience: {CONFIG.early_stopping_patience}")
logger.info(f"  LoRA: r={CONFIG.lora_r}, alpha={CONFIG.lora_alpha}")

os.makedirs(CONFIG.output_dir, exist_ok=True)
tracker.mark('Configuration', 'PASS', {'beta': CONFIG.beta, 'epochs': CONFIG.num_epochs})
""")

    # =========================================================================
    # CELL 3: Load & Validate Annotated Data
    # =========================================================================
    add_code("""# ============================================================
# CELL 3: LOAD ANNOTATED DATA
# ============================================================
logger.info("\\n" + "="*60)
logger.info("LOADING ANNOTATED DATA")
logger.info("="*60)

# Try multiple possible paths
possible_paths = [
    CONFIG.data_path,
    '/kaggle/input/datasets/pushkarprabhath/gricebench-dpo-annotations/tier1_hard_pairs_FULLY_ANNOTATED.json',
    '/kaggle/input/gricebench-dpo-annotations/tier1_hard_pairs_FULLY_ANNOTATED.json',
    '/kaggle/input/gricebench-dpo/tier1_hard_pairs_FULLY_ANNOTATED.json',
]

raw_data = None
for path in possible_paths:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        logger.info(f"Loaded from: {path}")
        break

if raw_data is None:
    logger.error("Data file not found! Available files:")
    for root, dirs, files in os.walk('/kaggle/input'):
        for fn in files:
            if fn.endswith('.json'):
                logger.error(f"  {os.path.join(root, fn)}")
    raise FileNotFoundError("tier1_hard_pairs_FULLY_ANNOTATED.json not found in any expected location")

logger.info(f"Total records loaded: {len(raw_data)}")

# Validate structure
required_keys = {'context', 'response_A', 'response_B', 'preference', 'reason'}
for i, entry in enumerate(raw_data):
    missing = required_keys - set(entry.keys())
    assert not missing, f"Entry {i} missing keys: {missing}"

# Preference distribution
pref_counts = Counter(d['preference'] for d in raw_data)
logger.info(f"\\nPreference distribution:")
for pref, count in sorted(pref_counts.items(), key=lambda x: -x[1]):
    logger.info(f"  {pref}: {count} ({100*count/len(raw_data):.1f}%)")

tracker.mark('Data Loading', 'PASS', {'total': len(raw_data), 'preferences': dict(pref_counts)})
""")

    # =========================================================================
    # CELL 4: Convert to DPO Format + Stratified Split
    # =========================================================================
    add_code("""# ============================================================
# CELL 4: CONVERT TO DPO FORMAT + STRATIFIED SPLIT
# ============================================================
logger.info("\\n" + "="*60)
logger.info("CONVERTING TO DPO FORMAT")
logger.info("="*60)

SYSTEM_PROMPT = (
    "You are a helpful conversational AI assistant that follows Gricean maxims. "
    "Be relevant (stay on topic), truthful (say only what you believe is true), "
    "clear (avoid ambiguity and be well-formatted), and appropriately informative "
    "(not too much, not too little)."
)

def create_prompt(context):
    return f"Continue the following conversation naturally, following Gricean maxims (be relevant, truthful, clear, appropriately informative):\\n\\n{context}\\n\\nResponse:"

dpo_pairs = []
skipped = 0

for entry in raw_data:
    pref = entry['preference']
    if pref == 'equal':
        skipped += 1
        continue
    
    prompt = create_prompt(entry['context'])
    
    if pref.startswith('A'):
        chosen = entry['response_A']
        rejected = entry['response_B']
    else:
        chosen = entry['response_B']
        rejected = entry['response_A']
    
    # Preference strength weight
    strength = 1.0 if 'much' in pref else 0.6
    
    dpo_pairs.append({
        'prompt': prompt,
        'chosen': chosen,
        'rejected': rejected,
        'reason': entry.get('reason', ''),
        'preference': pref,
        'strength': strength,
    })

logger.info(f"DPO pairs created: {len(dpo_pairs)}")
logger.info(f"Skipped (equal): {skipped}")

# Stratified split: keep proportion of strong/weak in train/val
strong_pairs = [p for p in dpo_pairs if p['strength'] == 1.0]
weak_pairs = [p for p in dpo_pairs if p['strength'] < 1.0]

random.shuffle(strong_pairs)
random.shuffle(weak_pairs)

val_strong_n = max(1, int(len(strong_pairs) * 0.15))
val_weak_n = max(1, int(len(weak_pairs) * 0.15)) if weak_pairs else 0

val_pairs = strong_pairs[:val_strong_n] + weak_pairs[:val_weak_n]
train_pairs = strong_pairs[val_strong_n:] + weak_pairs[val_weak_n:]

random.shuffle(train_pairs)
random.shuffle(val_pairs)

logger.info(f"\\nTrain: {len(train_pairs)} ({len([p for p in train_pairs if p['strength']==1.0])} strong, {len([p for p in train_pairs if p['strength']<1.0])} weak)")
logger.info(f"Val:   {len(val_pairs)} ({len([p for p in val_pairs if p['strength']==1.0])} strong, {len([p for p in val_pairs if p['strength']<1.0])} weak)")

steps_per_epoch = len(train_pairs) // (CONFIG.per_device_batch * CONFIG.gradient_accumulation)
total_steps = steps_per_epoch * CONFIG.num_epochs
logger.info(f"\\nSteps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
logger.info(f"Eval every {CONFIG.eval_steps} steps = ~every {CONFIG.eval_steps/steps_per_epoch:.1f} epochs")

# Convert to HuggingFace Dataset
from datasets import Dataset as HFDataset

train_dataset = HFDataset.from_list([{'prompt': p['prompt'], 'chosen': p['chosen'], 'rejected': p['rejected']} for p in train_pairs])
val_dataset = HFDataset.from_list([{'prompt': p['prompt'], 'chosen': p['chosen'], 'rejected': p['rejected']} for p in val_pairs])

logger.info(f"\\nTrain dataset: {len(train_dataset)} examples")
logger.info(f"Val dataset:   {len(val_dataset)} examples")

tracker.mark('DPO Conversion', 'PASS', {'train': len(train_pairs), 'val': len(val_pairs), 'steps': total_steps})
""")

    # =========================================================================
    # CELL 5: Load Model with QLoRA
    # =========================================================================
    add_code("""# ============================================================
# CELL 5: LOAD MODEL WITH QLoRA (4-bit)
# ============================================================
logger.info("\\n" + "="*60)
logger.info("LOADING MODEL + QLoRA")
logger.info("="*60)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

logger.info(f"Loading {CONFIG.model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    CONFIG.model_name,
    quantization_config=bnb_config,
    device_map='auto',
    trust_remote_code=True,
    attn_implementation='eager',
)

tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'

logger.info(f"Model loaded: {model.__class__.__name__}")
logger.info(f"Vocab size: {len(tokenizer)}")

# Prepare for QLoRA
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=CONFIG.lora_r,
    lora_alpha=CONFIG.lora_alpha,
    lora_dropout=CONFIG.lora_dropout,
    target_modules=list(CONFIG.lora_target_modules),
    bias='none',
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

trainable, total = 0, 0
for p in model.parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()

logger.info(f"\\nParameters:")
logger.info(f"  Total: {total:,}")
logger.info(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

# VRAM after model load
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"\\nVRAM Usage (after model load):")
    logger.info(f"  Allocated: {allocated:.2f} GB")
    logger.info(f"  Reserved: {reserved:.2f} GB")
    logger.info(f"  Available: {total_vram - allocated:.2f} GB")

tracker.mark('Model Loading', 'PASS', {'trainable_params': trainable, 'total_params': total})
""")

    # =========================================================================
    # CELL 6: Setup DPO Trainer with Callbacks
    # =========================================================================
    add_code("""# ============================================================
# CELL 6: DPO TRAINER SETUP (V2 — WITH CALLBACKS)
# ============================================================
logger.info("\\n" + "="*60)
logger.info("SETTING UP DPO TRAINER V2")
logger.info("="*60)

from trl import DPOConfig as TRLDPOConfig, DPOTrainer
from transformers import TrainerCallback, EarlyStoppingCallback

# ---- Diagnostic Callback ----
class DPODiagnosticCallback(TrainerCallback):
    \"\"\"Real-time training health monitor.\"\"\"
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.reward_margins = []
        self.reward_accs = []
        self.step_data = []  # all logged data
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        
        self.step_data.append(dict(logs))
        
        if 'loss' in logs:
            self.train_losses.append(logs['loss'])
        
        if 'eval_loss' in logs:
            self.eval_losses.append(logs['eval_loss'])
        
        if 'rewards/margins' in logs:
            self.reward_margins.append(logs['rewards/margins'])
        
        if 'eval_rewards/margins' in logs:
            margin = logs['eval_rewards/margins']
            if margin > 6.0:
                logger.warning(f"⚠️  Eval reward margin = {margin:.2f} — approaching over-separation!")
        
        if 'eval_rewards/accuracies' in logs:
            self.reward_accs.append(logs['eval_rewards/accuracies'])
        
        # Overfitting check
        if len(self.eval_losses) >= 3:
            if self.eval_losses[-1] > self.eval_losses[-2] > self.eval_losses[-3]:
                logger.warning("⚠️  Eval loss increasing for 3 consecutive checks — likely overfitting")
    
    def on_train_end(self, args, state, control, **kwargs):
        print("\\n" + "="*60)
        print("📊 DPO DIAGNOSTIC REPORT")
        print("="*60)
        
        if self.train_losses:
            print(f"\\nTrain loss: {self.train_losses[0]:.3f} → {self.train_losses[-1]:.3f}")
        if self.eval_losses:
            best_idx = int(np.argmin(self.eval_losses))
            print(f"Eval loss:  best={min(self.eval_losses):.3f} (check #{best_idx+1}) | final={self.eval_losses[-1]:.3f}")
        if self.reward_accs:
            print(f"Val accuracy: {self.reward_accs[0]:.1%} → {self.reward_accs[-1]:.1%}")
        
        # Health checks
        print("\\n✅ Health Checks:")
        
        # 1. Margin check
        if self.reward_margins:
            final_margin = self.reward_margins[-1]
            if 1.0 <= final_margin <= 5.0:
                print(f"  ✓ Train reward margin healthy ({final_margin:.2f})")
            elif final_margin > 5.0:
                print(f"  ✗ Train reward margin high ({final_margin:.2f}) — increase beta next time")
            else:
                print(f"  ✗ Train reward margin low ({final_margin:.2f}) — decrease beta next time")
        
        # 2. Overfitting check
        if self.eval_losses:
            best_idx = int(np.argmin(self.eval_losses))
            if best_idx == len(self.eval_losses) - 1:
                print("  ✓ No overfitting detected — final checkpoint is best")
            else:
                print(f"  ⚠ Best eval was check #{best_idx+1}, final is #{len(self.eval_losses)}")
                print(f"    → load_best_model_at_end will use the optimal checkpoint")
        
        # 3. Accuracy check
        if self.reward_accs:
            final_acc = self.reward_accs[-1]
            if 0.70 <= final_acc <= 0.92:
                print(f"  ✓ Val accuracy in healthy range ({final_acc:.1%})")
            elif final_acc > 0.92:
                print(f"  ⚠ Val accuracy very high ({final_acc:.1%}) — possible data leakage")
            else:
                print(f"  ✗ Val accuracy low ({final_acc:.1%}) — model may not be learning")
        
        print("\\n" + "="*60)

diagnostic_cb = DPODiagnosticCallback()

# ---- Training Arguments ----
training_args = TRLDPOConfig(
    output_dir=CONFIG.output_dir,
    
    # === V2 OPTIMIZED TRAINING SCHEDULE ===
    num_train_epochs=CONFIG.num_epochs,      # 3 (was 5)
    per_device_train_batch_size=CONFIG.per_device_batch,  # 8 (was 4)
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=CONFIG.gradient_accumulation,  # 4 (was 8)
    
    # DPO hyperparameters
    beta=CONFIG.beta,                         # 0.3 (was 0.1)
    loss_type='sigmoid',
    
    # Learning rate
    learning_rate=CONFIG.learning_rate,
    lr_scheduler_type='cosine',
    warmup_ratio=CONFIG.warmup_ratio,
    
    # === V2 — EARLY STOPPING + BEST CHECKPOINT ===
    eval_strategy='steps',
    eval_steps=CONFIG.eval_steps,             # 4 (was 25)
    save_strategy='steps',
    save_steps=CONFIG.eval_steps,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    save_total_limit=3,
    
    # Regularization
    weight_decay=CONFIG.weight_decay,
    max_grad_norm=CONFIG.max_grad_norm,
    
    # Efficiency
    bf16=True,
    gradient_checkpointing=True,
    optim='paged_adamw_8bit',
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    
    # === V2 — SEQUENCE LENGTHS ===
    max_length=CONFIG.max_length,             # 256 (was 512)
    max_prompt_length=CONFIG.max_prompt_length,  # 192 (was 384)
    
    # Logging
    logging_steps=1,
    report_to='none',
    seed=CONFIG.seed,
    
    # Disable find_unused_parameters for efficiency
    ddp_find_unused_parameters=False,
)

# ---- Create DPO Trainer ----
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    callbacks=[
        diagnostic_cb,
        EarlyStoppingCallback(
            early_stopping_patience=CONFIG.early_stopping_patience,
            early_stopping_threshold=0.01
        ),
    ],
)

# VRAM after trainer setup
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1e9
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"VRAM after trainer setup: {allocated:.2f} GB ({100*allocated/total_vram:.1f}%)")

tracker.mark('DPO Trainer Setup', 'PASS', {
    'beta': CONFIG.beta,
    'effective_batch': CONFIG.per_device_batch * CONFIG.gradient_accumulation,
    'max_length': CONFIG.max_length,
})
""")

    # =========================================================================
    # CELL 7: Training
    # =========================================================================
    add_code("""# ============================================================
# CELL 7: TRAIN (V2 — WITH MONITORING)
# ============================================================
logger.info("\\n" + "="*60)
logger.info("STARTING DPO TRAINING V2")
logger.info("="*60)
logger.info(f"Expected: ~{steps_per_epoch * CONFIG.num_epochs} steps, {CONFIG.num_epochs} epochs")
logger.info(f"Early stopping will halt if no improvement for {CONFIG.early_stopping_patience} evals")

# GPU utilization monitor
class GPUMonitor:
    def __init__(self):
        self.readings = []
        self.peak_util = 0
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

train_start = datetime.now()
logger.info(f"Training started at: {train_start.strftime('%H:%M:%S')}")

# Run training
try:
    train_result = trainer.train()
    train_time = (datetime.now() - train_start).total_seconds()
    
    logger.info(f"\\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Time: {train_time:.0f}s ({train_time/60:.1f} min)")
    logger.info(f"Final train loss: {train_result.training_loss:.4f}")
    
    # Check if early stopping triggered
    if hasattr(trainer.state, 'best_metric'):
        logger.info(f"Best eval_loss: {trainer.state.best_metric:.4f}")
        logger.info(f"Best checkpoint: step {trainer.state.best_model_checkpoint}")
    
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise

# VRAM peak
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated(0) / 1e9
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    peak_pct = 100 * peak_mem / total_vram
    logger.info(f"\\nPeak VRAM: {peak_mem:.2f} GB ({peak_pct:.1f}%)")

tracker.mark('DPO Training', 'PASS', {
    'train_loss': train_result.training_loss,
    'time_seconds': train_time,
    'peak_vram_pct': f'{peak_pct:.1f}%' if torch.cuda.is_available() else 'N/A',
})
""")

    # =========================================================================
    # CELL 8: Evaluation
    # =========================================================================
    add_code("""# ============================================================
# CELL 8: EVALUATION
# ============================================================
logger.info("\\n" + "="*60)
logger.info("EVALUATING BEST CHECKPOINT")
logger.info("="*60)

eval_results = trainer.evaluate()

logger.info(f"\\nEval Results:")
for key, value in sorted(eval_results.items()):
    if isinstance(value, float):
        logger.info(f"  {key}: {value:.4f}")
    else:
        logger.info(f"  {key}: {value}")

# Extract key metrics
eval_loss = eval_results.get('eval_loss', float('nan'))
eval_reward_acc = eval_results.get('eval_rewards/accuracies', float('nan'))
eval_margin = eval_results.get('eval_rewards/margins', float('nan'))

logger.info(f"\\n📊 KEY METRICS:")
logger.info(f"  Eval loss:       {eval_loss:.4f}")
logger.info(f"  Reward accuracy: {eval_reward_acc:.1%}")
logger.info(f"  Reward margin:   {eval_margin:.2f}")

# Health verdict
if eval_margin < 5.0:
    logger.info(f"  ✅ Margin healthy (< 5.0)")
else:
    logger.warning(f"  ⚠️ Margin still high (≥ 5.0)")

if eval_reward_acc > 0.75:
    logger.info(f"  ✅ Accuracy above threshold (> 75%)")
else:
    logger.warning(f"  ⚠️ Accuracy below expected (< 75%)")

tracker.mark('Evaluation', 'PASS', {
    'eval_loss': eval_loss,
    'reward_accuracy': eval_reward_acc,
    'reward_margin': eval_margin,
})
""")

    # =========================================================================
    # CELL 9: Preference Accuracy (FIXED — V2)
    # =========================================================================
    add_code("""# ============================================================
# CELL 9: PREFERENCE ACCURACY (FIXED — V2)
# ============================================================
# V1 BUG: Used model.forward() which doesn't compute DPO preferences.
# V2 FIX: Use the DPO trainer's built-in reward computation, which is
#          already reported as eval_rewards/accuracies. We also perform
#          a manual verification here for transparency.

logger.info("\\n" + "="*60)
logger.info("PREFERENCE ACCURACY VERIFICATION")
logger.info("="*60)

# The DPO trainer already computed this correctly
# eval_rewards/accuracies = fraction where chosen_reward > rejected_reward
# This IS the preference accuracy metric.

logger.info(f"\\nFrom DPO Trainer evaluation:")
logger.info(f"  Preference accuracy: {eval_reward_acc:.1%}")
logger.info(f"  (Model assigns higher reward to chosen response {eval_reward_acc:.1%} of the time)")

# Additional: manual spot-check with generation
logger.info(f"\\nRunning generation spot-check on 5 validation samples...")

model.eval()
check_correct = 0
check_total = 0

for i in range(min(5, len(val_pairs))):
    pair = val_pairs[i]
    
    # Tokenize prompt + chosen and prompt + rejected
    chosen_text = f"{pair['prompt']}\\n{pair['chosen']}"
    rejected_text = f"{pair['prompt']}\\n{pair['rejected']}"
    
    with torch.no_grad():
        chosen_ids = tokenizer(chosen_text, return_tensors='pt', truncation=True, max_length=CONFIG.max_length).to(device)
        rejected_ids = tokenizer(rejected_text, return_tensors='pt', truncation=True, max_length=CONFIG.max_length).to(device)
        
        # Get per-token log probs
        chosen_out = model(**chosen_ids)
        rejected_out = model(**rejected_ids)
        
        # Average log prob (normalized by length)
        chosen_logprob = -torch.nn.functional.cross_entropy(
            chosen_out.logits[:, :-1, :].reshape(-1, chosen_out.logits.size(-1)),
            chosen_ids['input_ids'][:, 1:].reshape(-1),
            reduction='mean'
        ).item()
        
        rejected_logprob = -torch.nn.functional.cross_entropy(
            rejected_out.logits[:, :-1, :].reshape(-1, rejected_out.logits.size(-1)),
            rejected_ids['input_ids'][:, 1:].reshape(-1),
            reduction='mean'
        ).item()
    
    correct = chosen_logprob > rejected_logprob
    check_correct += int(correct)
    check_total += 1
    
    icon = "✅" if correct else "❌"
    logger.info(f"  Sample {i+1}: chosen={chosen_logprob:.3f} vs rejected={rejected_logprob:.3f} {icon}")

logger.info(f"\\nSpot-check: {check_correct}/{check_total} correct ({100*check_correct/check_total:.0f}%)")
logger.info(f"Official preference accuracy (from DPO eval): {eval_reward_acc:.1%}")

tracker.mark('Preference Accuracy', 'PASS', {
    'dpo_eval_accuracy': eval_reward_acc,
    'spot_check': f'{check_correct}/{check_total}',
})
""")

    # =========================================================================
    # CELL 10: Sample Generation
    # =========================================================================
    add_code("""# ============================================================
# CELL 10: SAMPLE GENERATION
# ============================================================
logger.info("\\n" + "="*60)
logger.info("GENERATING SAMPLE RESPONSES")
logger.info("="*60)

generation_samples = []
model.eval()

for i in range(min(5, len(val_pairs))):
    pair = val_pairs[i]
    prompt = pair['prompt']
    
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=CONFIG.max_prompt_length).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    sample = {
        'prompt_snippet': prompt[:100],
        'generated': generated[:200],
        'original_chosen': pair['chosen'][:150],
        'original_rejected': pair['rejected'][:150],
        'reason': pair.get('reason', 'N/A') if isinstance(pair, dict) and 'reason' in pair else val_pairs[i].get('reason', 'N/A'),
    }
    generation_samples.append(sample)
    
    logger.info(f"\\n--- Sample {i+1} ---")
    logger.info(f"Generated: {generated[:200]}")

tracker.mark('Sample Generation', 'PASS', {'n_samples': len(generation_samples)})
""")

    # =========================================================================
    # CELL 11: Save Results
    # =========================================================================
    add_code("""# ============================================================
# CELL 11: SAVE RESULTS
# ============================================================
logger.info("\\n" + "="*60)
logger.info("SAVING RESULTS")
logger.info("="*60)

# Compile results
results = {
    'phase': 'Phase 5 - DPO Training V2 (Optimized)',
    'timestamp': datetime.now().isoformat(),
    'model': CONFIG.model_name,
    'method': 'DPO with QLoRA (4-bit) — V2 Optimized',
    'version_changes': {
        'beta': '0.1 → 0.3 (regularize margins)',
        'epochs': '5 → 3 (prevent overfitting)',
        'batch_size': '4 → 8 (GPU utilization)',
        'max_length': '512 → 256 (data-informed)',
        'new_features': ['EarlyStoppingCallback', 'DPODiagnosticCallback', 'load_best_model_at_end'],
    },
    'data': {
        'total_annotated': len(raw_data),
        'dpo_pairs': len(dpo_pairs),
        'train_size': len(train_pairs),
        'val_size': len(val_pairs),
        'preference_distribution': dict(pref_counts),
    },
    'hyperparameters': {
        'lora_r': CONFIG.lora_r,
        'lora_alpha': CONFIG.lora_alpha,
        'learning_rate': CONFIG.learning_rate,
        'effective_batch_size': CONFIG.per_device_batch * CONFIG.gradient_accumulation,
        'num_epochs': CONFIG.num_epochs,
        'beta': CONFIG.beta,
        'max_length': CONFIG.max_length,
        'early_stopping_patience': CONFIG.early_stopping_patience,
    },
    'training': {
        'train_loss': train_result.training_loss,
        'training_time_seconds': train_time,
        'best_checkpoint': str(getattr(trainer.state, 'best_model_checkpoint', 'N/A')),
        'best_eval_loss': getattr(trainer.state, 'best_metric', None),
    },
    'evaluation': {
        'eval_loss': eval_loss,
        'preference_accuracy': eval_reward_acc,
        'reward_margin': eval_margin,
    },
    'generation_samples': generation_samples,
    'gpu': {
        'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'peak_vram_gb': float(torch.cuda.max_memory_allocated(0) / 1e9) if torch.cuda.is_available() else 0,
        'total_vram_gb': float(torch.cuda.get_device_properties(0).total_memory / 1e9) if torch.cuda.is_available() else 0,
    },
    'diagnostic_data': {
        'train_losses': diagnostic_cb.train_losses,
        'eval_losses': diagnostic_cb.eval_losses,
        'reward_margins': diagnostic_cb.reward_margins,
        'reward_accuracies': diagnostic_cb.reward_accs,
    },
}

# Save results
results_path = os.path.join(CONFIG.output_dir, 'dpo_results_v2.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
logger.info(f"Results saved: {results_path}")

# Save training history
history_path = os.path.join(CONFIG.output_dir, 'training_history_v2.json')
with open(history_path, 'w') as f:
    json.dump(trainer.state.log_history, f, indent=2)
logger.info(f"Training history saved: {history_path}")

# Save LoRA adapter
adapter_path = os.path.join(CONFIG.output_dir, 'lora_adapter')
model.save_pretrained(adapter_path)
tokenizer.save_pretrained(adapter_path)
logger.info(f"LoRA adapter saved: {adapter_path}")

# Copy to /kaggle/working for easy download
import shutil
for fname in ['dpo_results_v2.json', 'training_history_v2.json']:
    src = os.path.join(CONFIG.output_dir, fname)
    dst = os.path.join('/kaggle/working', fname)
    shutil.copy2(src, dst)
    logger.info(f"Copied: {dst}")

tracker.mark('Save Results', 'PASS', {'results_file': results_path})
""")

    # =========================================================================
    # CELL 12: Final Summary
    # =========================================================================
    add_code("""# ============================================================
# CELL 12: FINAL SUMMARY
# ============================================================
print("\\n" + "="*60)
print("🏁 PHASE 5 DPO TRAINING V2 — FINAL SUMMARY")
print("="*60)

print(f"\\n📦 Model: {CONFIG.model_name}")
print(f"📊 Data: {len(dpo_pairs)} DPO pairs ({len(train_pairs)} train, {len(val_pairs)} val)")

print(f"\\n⚡ V2 Improvements:")
print(f"  β: 0.1 → {CONFIG.beta}")
print(f"  Epochs: 5 → {CONFIG.num_epochs}")
print(f"  Batch: 4 → {CONFIG.per_device_batch}")
print(f"  Max length: 512 → {CONFIG.max_length}")

print(f"\\n📈 Training Results:")
print(f"  Train loss: {train_result.training_loss:.4f}")
print(f"  Best eval loss: {getattr(trainer.state, 'best_metric', 'N/A')}")
print(f"  Time: {train_time:.0f}s ({train_time/60:.1f} min)")

print(f"\\n🎯 Key Metrics:")
print(f"  Preference accuracy: {eval_reward_acc:.1%}")
print(f"  Reward margin: {eval_margin:.2f}")
if eval_margin < 5.0:
    print(f"  ✅ Margin healthy (target: < 5.0)")
if eval_reward_acc > 0.75:
    print(f"  ✅ Accuracy above threshold (target: > 75%)")

if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\\n🖥️ GPU:")
    print(f"  Peak VRAM: {peak:.2f} / {total:.1f} GB ({100*peak/total:.1f}%)")

print(f"\\n📁 Output Files:")
print(f"  /kaggle/working/dpo_results_v2.json")
print(f"  /kaggle/working/training_history_v2.json")
print(f"  {CONFIG.output_dir}/lora_adapter/")

print(f"\\n{'='*60}")
print(f"✅ PHASE 5 V2 COMPLETE — Download dpo_results_v2.json and training_history_v2.json")
print(f"{'='*60}")

# Cleanup
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

tracker.summary()
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
                        "datasetSlug": "gricebench-dpo-annotations"
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

    # Write notebook
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'KAGGLE_PHASE5_DPO_V2.ipynb'
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
