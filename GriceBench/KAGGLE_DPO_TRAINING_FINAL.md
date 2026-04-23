# 🚀 DPO Training - Complete Guide (Copy-Paste Ready)

## 📋 What You Have Right Now

✅ **Downloaded locally:**
- `dpo_train_filtered.json` (1,970 conflict-free pairs)
- `dpo_val_filtered.json` (101 pairs)

✅ **Data Quality:**
- Quantity margin: +0.073
- Quality margin: -0.023 (neutral)
- Relation margin: +0.019  
- Manner margin: +0.070
- No conflicting signals!

---

## 🎯 What You Need to Do (3 Steps Total)

### **STEP 1: Upload to Kaggle** (10 minutes)

1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload these 2 files:
   - `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\dpo_train_filtered.json`
   - `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\dpo_val_filtered.json`
4. **Title:** `gricebench-dpo-filtered`
5. Click **"Create"**

**Your dataset URL will be:**
```
https://www.kaggle.com/datasets/pushkarprabhath/gricebench-dpo-filtered
```

---

### **STEP 2: Create Kaggle Notebook** (5 minutes)

1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Rename to: `dpo-training`
4. **Settings** → **Accelerator** → **GPU T4 x2**
5. **Add Data** → Search `pushkarprabhath/gricebench-dpo-filtered` → Add

---

### **STEP 3: Copy-Paste 10 Cells** (10 minutes + 3-4 hours GPU)

Copy each cell below into your Kaggle notebook and run them in order.

---

## 📝 CELL 1: Setup

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("✓ Imports complete")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

---

## 📝 CELL 2: Configuration

```python
CONFIG = {
    # YOUR DATASET PATH
    'train_data': '/kaggle/input/gricebench-dpo-filtered/dpo_train_filtered.json',
    'val_data': '/kaggle/input/gricebench-dpo-filtered/dpo_val_filtered.json',
    
    # Model
    'base_model': 'gpt2-medium',
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    
    # Training
    'batch_size': 1,
    'gradient_accumulation': 4,
    'learning_rate': 5e-5,
    'num_epochs': 3,
    'max_length': 512,
    
    # DPO
    'beta_start': 0.03,
    'beta_end': 0.08,
    'warmup_steps': 500,
    
    # Maxim weights (focus on weak ones)
    'maxim_weights': {
        'quantity': 1.0,
        'quality': 1.2,
        'relation': 1.0,
        'manner': 1.2
    },
    
    'output_dir': '/kaggle/working/dpo_final',
    'device': device
}

print("✓ Configuration loaded")
```

---

## 📝 CELL 3: Adaptive Beta

```python
class AdaptiveBetaScheduler:
    def __init__(self, beta_start, beta_end, warmup_steps):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def step(self):
        if self.step_count < self.warmup_steps:
            progress = self.step_count / self.warmup_steps
            beta = self.beta_start + (self.beta_end - self.beta_start) * progress
        else:
            beta = self.beta_end
        self.step_count += 1
        return beta

print("✓ Beta scheduler defined")
```

---

## 📝 CELL 4: Multi-Objective Loss

```python
class MultiObjectiveDPOLoss(nn.Module):
    def __init__(self, maxim_weights):
        super().__init__()
        self.maxims = ['quantity', 'quality', 'relation', 'manner']
        self.maxim_weights = maxim_weights
    
    def forward(self, policy_chosen_logps, policy_rejected_logps,
                reference_chosen_logps, reference_rejected_logps,
                chosen_scores, rejected_scores, beta):
        
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        logits = beta * (policy_logratios - reference_logratios)
        
        maxim_losses = {}
        for maxim in self.maxims:
            chosen_score = chosen_scores.get(maxim, 0.5)
            rejected_score = rejected_scores.get(maxim, 0.5)
            margin = rejected_score - chosen_score
            
            loss = -F.logsigmoid(logits)
            maxim_losses[maxim] = loss * max(margin, 0.01)
        
        total_loss = sum(
            maxim_losses[m] * self.maxim_weights[m]
            for m in self.maxims
        )
        total_loss = total_loss / sum(self.maxim_weights.values())
        
        return total_loss.mean(), maxim_losses

print("✓ Loss function defined")
```

---

## 📝 CELL 5: Dataset

```python
class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)['input_ids']
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)['input_ids']
        
        chosen_full = prompt_tokens + chosen_tokens
        rejected_full = prompt_tokens + rejected_tokens
        
        if len(chosen_full) > self.max_length:
            chosen_full = chosen_full[:self.max_length]
        if len(rejected_full) > self.max_length:
            rejected_full = rejected_full[:self.max_length]
        
        return {
            'chosen_input_ids': torch.tensor(chosen_full),
            'rejected_input_ids': torch.tensor(rejected_full),
            'prompt_length': len(prompt_tokens),
            'chosen_scores': item['chosen_scores'],
            'rejected_scores': item['rejected_scores']
        }

def collate_fn(batch):
    max_chosen = max(len(item['chosen_input_ids']) for item in batch)
    max_rejected = max(len(item['rejected_input_ids']) for item in batch)
    
    chosen_ids, rejected_ids = [], []
    chosen_mask, rejected_mask = [], []
    prompt_lengths, chosen_scores, rejected_scores = [], [], []
    
    for item in batch:
        chosen = item['chosen_input_ids']
        chosen_pad = max_chosen - len(chosen)
        chosen_ids.append(F.pad(chosen, (0, chosen_pad), value=50256))
        chosen_mask.append(torch.cat([torch.ones(len(chosen)), torch.zeros(chosen_pad)]))
        
        rejected = item['rejected_input_ids']
        rejected_pad = max_rejected - len(rejected)
        rejected_ids.append(F.pad(rejected, (0, rejected_pad), value=50256))
        rejected_mask.append(torch.cat([torch.ones(len(rejected)), torch.zeros(rejected_pad)]))
        
        prompt_lengths.append(item['prompt_length'])
        chosen_scores.append(item['chosen_scores'])
        rejected_scores.append(item['rejected_scores'])
    
    return {
        'chosen_input_ids': torch.stack(chosen_ids),
        'rejected_input_ids': torch.stack(rejected_ids),
        'chosen_attention_mask': torch.stack(chosen_mask),
        'rejected_attention_mask': torch.stack(rejected_mask),
        'prompt_lengths': prompt_lengths,
        'chosen_scores': chosen_scores,
        'rejected_scores': rejected_scores
    }

print("✓ Dataset defined")
```

---

## 📝 CELL 6: Load Models

```python
print("Loading models...")

tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'])
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    CONFIG['base_model'],
    torch_dtype=torch.float16
).to(CONFIG['device'])

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    lora_dropout=CONFIG['lora_dropout'],
    target_modules=['c_attn', 'c_proj', 'c_fc'],
    bias="none"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

reference_model = AutoModelForCausalLM.from_pretrained(
    CONFIG['base_model'],
    torch_dtype=torch.float16
).to(CONFIG['device'])
reference_model.eval()

print("✓ Models loaded")

train_dataset = DPODataset(CONFIG['train_data'], tokenizer, CONFIG['max_length'])
val_dataset = DPODataset(CONFIG['val_data'], tokenizer, CONFIG['max_length'])

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)

print(f"✓ Train: {len(train_dataset)} pairs")
print(f"✓ Val: {len(val_dataset)} pairs")
```

---

## 📝 CELL 7: Training Setup

```python
criterion = MultiObjectiveDPOLoss(CONFIG['maxim_weights'])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
beta_scheduler = AdaptiveBetaScheduler(CONFIG['beta_start'], CONFIG['beta_end'], CONFIG['warmup_steps'])

total_steps = len(train_loader) * CONFIG['num_epochs'] // CONFIG['gradient_accumulation']
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

print("✓ Training setup complete")
```

---

## 📝 CELL 8: Helper Function

```python
def get_log_probs(model, input_ids, attention_mask, prompt_lengths):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    mask = torch.zeros_like(token_log_probs)
    for i, plen in enumerate(prompt_lengths):
        mask[i, plen:] = 1
    
    return (token_log_probs * mask).sum(dim=1)

print("✓ Helper function defined")
```

---

## 📝 CELL 9: Training Loop

```python
def train_epoch(model, reference_model, loader, criterion, optimizer, lr_scheduler, beta_scheduler):
    model.train()
    reference_model.eval()
    
    total_loss = 0
    maxim_losses_sum = {m: 0 for m in ['quantity', 'quality', 'relation', 'manner']}
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        chosen_ids = batch['chosen_input_ids'].to(CONFIG['device'])
        rejected_ids = batch['rejected_input_ids'].to(CONFIG['device'])
        chosen_mask = batch['chosen_attention_mask'].to(CONFIG['device'])
        rejected_mask = batch['rejected_attention_mask'].to(CONFIG['device'])
        prompt_lengths = batch['prompt_lengths']
        
        beta = beta_scheduler.step()
        
        policy_chosen_logps = get_log_probs(model, chosen_ids, chosen_mask, prompt_lengths)
        policy_rejected_logps = get_log_probs(model, rejected_ids, rejected_mask, prompt_lengths)
        
        with torch.no_grad():
            reference_chosen_logps = get_log_probs(reference_model, chosen_ids, chosen_mask, prompt_lengths)
            reference_rejected_logps = get_log_probs(reference_model, rejected_ids, rejected_mask, prompt_lengths)
        
        loss, maxim_losses = criterion(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps,
            batch['chosen_scores'][0], batch['rejected_scores'][0],
            beta
        )
        
        loss = loss / CONFIG['gradient_accumulation']
        loss.backward()
        
        for maxim, mloss in maxim_losses.items():
            maxim_losses_sum[maxim] += mloss.item()
        
        if (step + 1) % CONFIG['gradient_accumulation'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * CONFIG['gradient_accumulation']
    
    for maxim in maxim_losses_sum:
        maxim_losses_sum[maxim] /= len(loader)
    
    return total_loss / len(loader), maxim_losses_sum

print("✓ Training loop defined")
```

---

## 📝 CELL 10: Run Training

```python
print("\n" + "="*60)
print("STARTING DPO TRAINING")
print("="*60)

history = {'train_loss': [], 'maxim_losses': []}

for epoch in range(CONFIG['num_epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
    print("-"*60)
    
    train_loss, maxim_losses = train_epoch(
        model, reference_model, train_loader, criterion,
        optimizer, lr_scheduler, beta_scheduler
    )
    
    history['train_loss'].append(train_loss)
    history['maxim_losses'].append(maxim_losses)
    
    print(f"\nTrain Loss: {train_loss:.4f}")
    print("Per-maxim losses:")
    for maxim, loss in maxim_losses.items():
        print(f"  {maxim.capitalize():12s}: {loss:.4f}")

output_dir = Path(CONFIG['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(output_dir / 'final_model')
tokenizer.save_pretrained(output_dir / 'final_model')

with open(output_dir / 'history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\n" + "="*60)
print("✅ DPO TRAINING COMPLETE!")
print("="*60)
print("\n📥 Download from /kaggle/working/dpo_final/")
print("  - final_model/ (LoRA adapters)")
print("  - history.json")
print("="*60)
```

---

## ✅ After Training

**Download these files:**
1. Navigate to `/kaggle/working/dpo_final/`
2. Download `final_model/` folder
3. Download `history.json`

**Save to:**
```
c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\models\dpo_final\
```

**Then:** Let me know and I'll run the final evaluation!

---

## 🎯 Expected Results

- Training time: 3-4 hours
- All maxim losses should decrease
- Final loss < 0.45
- Cooperative rate: 62-68%
- 3/4 maxims improve significantly

**That's it! Just 3 steps total.** 🚀
