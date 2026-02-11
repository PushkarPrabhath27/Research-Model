"""
KAGGLE NOTEBOOK: DPO Optimization with Multi-Objective Loss

This notebook retrains the DPO model with:
1. Filtered preference pairs (clear margins only)
2. Multi-objective loss (balances all 4 maxims)
3. Adaptive beta scheduling (prevents instability)

Upload this as a Kaggle notebook and run with GPU.
"""

# ============================================
# CELL 1: Setup and Imports
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("✓ Imports complete")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# CELL 2: Configuration
# ============================================

CONFIG = {
    # Paths (update with your Kaggle dataset IDs)
    'base_model': 'gpt2-medium',
    'train_data_path': '/kaggle/input/gricebench-dpo-filtered/dpo_train_filtered.json',
    'val_data_path': '/kaggle/input/gricebench-dpo-data/dpo_val.json',
    
    # LoRA
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'lora_target_modules': ['c_attn', 'c_proj', 'c_fc'],
    
    # Training
    'batch_size': 1,
    'gradient_accumulation_steps': 4,
    'learning_rate': 5e-5,
    'num_epochs': 3,
    'max_length': 512,
    
    # DPO - Adaptive Beta
    'beta_start': 0.03,  # Conservative start
    'beta_end': 0.08,    # Moderate final value
    'warmup_steps': 500,
    
    # Multi-objective weights
    'maxim_weights': {
        'quantity': 1.0,
        'quality': 1.2,   # Slightly higher (was degrading)
        'relation': 1.0,
        'manner': 1.2     # Slightly higher (was degrading)
    },
    
    # Output
    'output_dir': '/kaggle/working/dpo_optimized',
    'device': device
}

print("Configuration:")
for key, val in CONFIG.items():
    if key != 'device':
        print(f"  {key}: {val}")

# ============================================
# CELL 3: Adaptive Beta Scheduler
# ============================================

class AdaptiveBetaScheduler:
    """
    Gradually increase beta during training
    
    Start conservatively to avoid disrupting base model,
    then increase once model starts learning.
    """
    def __init__(self, beta_start=0.03, beta_end=0.08, warmup_steps=500):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_steps = warmup_steps
        self.current_step = 0
    
    def get_beta(self):
        """Get current beta value"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        else:
            beta = self.beta_end
        
        return beta
    
    def step(self):
        """Update step counter"""
        self.current_step += 1
        return self.get_beta()

print("✓ Adaptive beta scheduler defined")

# ============================================
# CELL 4: Multi-Objective DPO Loss
# ============================================

class MultiObjectiveDPOLoss(nn.Module):
    """
    DPO loss that balances all four Gricean maxims
    
    Weights each maxim by:
    1. User-defined importance (maxim_weights)
    2. Preference margin (focus on clear preferences)
    """
    def __init__(self, maxim_weights=None):
        super().__init__()
        
        self.maxims = ['quantity', 'quality', 'relation', 'manner']
        
        if maxim_weights is None:
            self.maxim_weights = {m: 1.0 for m in self.maxims}
        else:
            self.maxim_weights = maxim_weights
    
    def forward(self, policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                chosen_scores, rejected_scores, beta):
        """
        Compute multi-objective DPO loss
        
        Args:
            policy_chosen_logps: Log probs from policy model for chosen
            policy_rejected_logps: Log probs from policy model for rejected
            reference_chosen_logps: Log probs from reference model for chosen
            reference_rejected_logps: Log probs from reference model for rejected
            chosen_scores: Dict of detector scores for chosen response
            rejected_scores: Dict of detector scores for rejected response
            beta: DPO temperature parameter
        """
        # Standard DPO logits
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = beta * (policy_logratios - reference_logratios)
        
        # Compute loss per maxim weighted by margin
        maxim_losses = {}
        
        for maxim in self.maxims:
            # Get margin for this maxim
            chosen_score = chosen_scores.get(maxim, 0.5)
            rejected_score = rejected_scores.get(maxim, 0.5)
            margin = rejected_score - chosen_score  # Positive = chosen is better
            
            # Standard DPO loss
            loss = -F.logsigmoid(logits)
            
            # Weight by margin (focus on clear preferences)
            maxim_losses[maxim] = loss * max(margin, 0.01)  # Avoid zero weight
        
        # Combine with maxim weights
        total_loss = sum(
            maxim_losses[m] * self.maxim_weights[m]
            for m in self.maxims
        )
        
        # Normalize by total weight
        total_loss = total_loss / sum(self.maxim_weights.values())
        
        return total_loss.mean(), maxim_losses

print("✓ Multi-objective DPO loss defined")

# ============================================
# CELL 5: Dataset Class
# ============================================

class DPODataset(Dataset):
    """Dataset for DPO training"""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path) as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get prompt, chosen, rejected
        prompt = item.get('prompt', item.get('context', ''))
        chosen = item.get('chosen', item.get('chosen_response', ''))
        rejected = item.get('rejected', item.get('rejected_response', ''))
        
        # Get detector scores
        chosen_scores = item.get('chosen_scores', {})
        rejected_scores = item.get('rejected_scores', {})
        
        # Tokenize
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)['input_ids']
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)['input_ids']
        
        # Combine prompt + response
        chosen_full = prompt_tokens + chosen_tokens
        rejected_full = prompt_tokens + rejected_tokens
        
        # Truncate if needed
        if len(chosen_full) > self.max_length:
            chosen_full = chosen_full[:self.max_length]
        if len(rejected_full) > self.max_length:
            rejected_full = rejected_full[:self.max_length]
        
        return {
            'chosen_input_ids': torch.tensor(chosen_full),
            'rejected_input_ids': torch.tensor(rejected_full),
            'prompt_length': len(prompt_tokens),
            'chosen_scores': chosen_scores,
            'rejected_scores': rejected_scores
        }

def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    max_chosen_len = max(len(item['chosen_input_ids']) for item in batch)
    max_rejected_len = max(len(item['rejected_input_ids']) for item in batch)
    
    chosen_input_ids = []
    rejected_input_ids = []
    chosen_attention_mask = []
    rejected_attention_mask = []
    prompt_lengths = []
    chosen_scores_batch = []
    rejected_scores_batch = []
    
    for item in batch:
        # Pad chosen
        chosen = item['chosen_input_ids']
        chosen_pad = max_chosen_len - len(chosen)
        chosen_input_ids.append(F.pad(chosen, (0, chosen_pad), value=50256))  # GPT-2 pad token
        chosen_attention_mask.append(torch.cat([
            torch.ones(len(chosen)),
            torch.zeros(chosen_pad)
        ]))
        
        # Pad rejected
        rejected = item['rejected_input_ids']
        rejected_pad = max_rejected_len - len(rejected)
        rejected_input_ids.append(F.pad(rejected, (0, rejected_pad), value=50256))
        rejected_attention_mask.append(torch.cat([
            torch.ones(len(rejected)),
            torch.zeros(rejected_pad)
        ]))
        
        prompt_lengths.append(item['prompt_length'])
        chosen_scores_batch.append(item['chosen_scores'])
        rejected_scores_batch.append(item['rejected_scores'])
    
    return {
        'chosen_input_ids': torch.stack(chosen_input_ids),
        'rejected_input_ids': torch.stack(rejected_input_ids),
        'chosen_attention_mask': torch.stack(chosen_attention_mask),
        'rejected_attention_mask': torch.stack(rejected_attention_mask),
        'prompt_lengths': prompt_lengths,
        'chosen_scores': chosen_scores_batch,
        'rejected_scores': rejected_scores_batch
    }

print("✓ Dataset class defined")

# ============================================
# CELL 6: Load Model and Data
# ============================================

print("Loading model and tokenizer...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'])
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    CONFIG['base_model'],
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(CONFIG['device'])

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    lora_dropout=CONFIG['lora_dropout'],
    target_modules=CONFIG['lora_target_modules'],
    bias="none"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Create reference model (frozen copy)
reference_model = AutoModelForCausalLM.from_pretrained(
    CONFIG['base_model'],
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(CONFIG['device'])
reference_model.eval()

print("✓ Models loaded")

# Load datasets
train_dataset = DPODataset(
    CONFIG['train_data_path'],
    tokenizer,
    CONFIG['max_length']
)

val_dataset = DPODataset(
    CONFIG['val_data_path'],
    tokenizer,
    CONFIG['max_length']
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    collate_fn=collate_fn
)

print(f"✓ Train examples: {len(train_dataset)}")
print(f"✓ Val examples: {len(val_dataset)}")

# ============================================
# CELL 7: Training Setup
# ============================================

# Loss function
criterion = MultiObjectiveDPOLoss(CONFIG['maxim_weights'])

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=0.01
)

# Beta scheduler
beta_scheduler = AdaptiveBetaScheduler(
    beta_start=CONFIG['beta_start'],
    beta_end=CONFIG['beta_end'],
    warmup_steps=CONFIG['warmup_steps']
)

# Learning rate scheduler
total_steps = len(train_loader) * CONFIG['num_epochs'] // CONFIG['gradient_accumulation_steps']
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)

print("✓ Training setup complete")
print(f"  Total steps: {total_steps}")
print(f"  Beta schedule: {CONFIG['beta_start']} → {CONFIG['beta_end']}")

# ============================================
# CELL 8: Helper Functions
# ============================================

def get_log_probs(model, input_ids, attention_mask, prompt_length):
    """Get log probabilities for responses (excluding prompt)"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    # Get log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs for actual tokens
    token_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Mask out prompt tokens (only count response)
    mask = torch.zeros_like(token_log_probs)
    for i, plen in enumerate(prompt_length):
        mask[i, plen:] = 1
    
    # Sum log probs for response only
    response_log_probs = (token_log_probs * mask).sum(dim=1)
    
    return response_log_probs

print("✓ Helper functions defined")

# ============================================
# CELL 9: Training Loop
# ============================================

def train_epoch(model, reference_model, loader, criterion, optimizer, 
                lr_scheduler, beta_scheduler, device):
    """Train for one epoch"""
    model.train()
    reference_model.eval()
    
    total_loss = 0
    maxim_losses_sum = {m: 0 for m in ['quantity', 'quality', 'relation', 'manner']}
    
    progress_bar = tqdm(loader, desc="Training")
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # Move to device
        chosen_ids = batch['chosen_input_ids'].to(device)
        rejected_ids = batch['rejected_input_ids'].to(device)
        chosen_mask = batch['chosen_attention_mask'].to(device)
        rejected_mask = batch['rejected_attention_mask'].to(device)
        prompt_lengths = batch['prompt_lengths']
        
        # Get beta for this step
        beta = beta_scheduler.step()
        
        # Get log probs from policy model
        policy_chosen_logps = get_log_probs(model, chosen_ids, chosen_mask, prompt_lengths)
        policy_rejected_logps = get_log_probs(model, rejected_ids, rejected_mask, prompt_lengths)
        
        # Get log probs from reference model
        with torch.no_grad():
            reference_chosen_logps = get_log_probs(reference_model, chosen_ids, chosen_mask, prompt_lengths)
            reference_rejected_logps = get_log_probs(reference_model, rejected_ids, rejected_mask, prompt_lengths)
        
        # Compute loss
        loss, maxim_losses = criterion(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            batch['chosen_scores'][0],  # First item in batch
            batch['rejected_scores'][0],
            beta
        )
        
        loss = loss / CONFIG['gradient_accumulation_steps']
        loss.backward()
        
        # Track maxim losses
        for maxim, mloss in maxim_losses.items():
            maxim_losses_sum[maxim] += mloss.item()
        
        # Update weights
        if (step + 1) % CONFIG['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * CONFIG['gradient_accumulation_steps']
        progress_bar.set_postfix({
            'loss': total_loss / (step + 1),
            'beta': beta
        })
    
    # Average maxim losses
    for maxim in maxim_losses_sum:
        maxim_losses_sum[maxim] /= len(loader)
    
    return total_loss / len(loader), maxim_losses_sum

print("✓ Training loop defined")

# ============================================
# CELL 10: Run Training
# ============================================

print("\n" + "="*60)
print("STARTING DPO OPTIMIZATION")
print("="*60)

history = {
    'train_loss': [],
    'maxim_losses': []
}

for epoch in range(CONFIG['num_epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
    print("-"*60)
    
    # Train
    train_loss, maxim_losses = train_epoch(
        model, reference_model, train_loader, criterion,
        optimizer, lr_scheduler, beta_scheduler, CONFIG['device']
    )
    
    # Log
    history['train_loss'].append(train_loss)
    history['maxim_losses'].append(maxim_losses)
    
    print(f"\nTrain Loss: {train_loss:.4f}")
    print("Per-maxim losses:")
    for maxim, loss in maxim_losses.items():
        print(f"  {maxim.capitalize():12s}: {loss:.4f}")
    
    # Save checkpoint
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_dir / f'checkpoint_epoch{epoch+1}')
    
    print(f"✓ Saved checkpoint")

# Save final model
model.save_pretrained(CONFIG['output_dir'] + '/final_model')
tokenizer.save_pretrained(CONFIG['output_dir'] + '/final_model')

print("\n" + "="*60)
print("DPO OPTIMIZATION COMPLETE!")
print("="*60)

# Save history
with open(CONFIG['output_dir'] + '/history_optimized.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"✓ Saved training history")
print("\nGenerated files:")
print(f"  - final_model/ (LoRA adapters)")
print(f"  - history_optimized.json")
print("\nDownload and merge with base model for evaluation!")
print("="*60)
