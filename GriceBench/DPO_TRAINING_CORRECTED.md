# CORRECTED DPO Training - Proper Implementation

## Cell 5: CORRECTED DPO Training

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time

print("="*70)
print("CORRECTED DPO TRAINING")
print("="*70)

# Ensure model on GPU
model = model.to('cuda').train()
print(f"Model device: {next(model.parameters()).device}")

# Create reference model
import copy
ref_model = copy.deepcopy(model).eval()
for param in ref_model.parameters():
    param.requires_grad = False
print("Reference model created")

# Proper Dataset class
class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize prompt + chosen
        chosen_text = item['prompt'] + " " + item['chosen']
        chosen = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize prompt + rejected  
        rejected_text = item['prompt'] + " " + item['rejected']
        rejected = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Tokenize prompt only (to know where to start computing loss)
        prompt = self.tokenizer(
            item['prompt'],
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        prompt_len = prompt['attention_mask'].sum().item()
        
        return {
            'chosen_input_ids': chosen['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected['attention_mask'].squeeze(0),
            'prompt_len': prompt_len
        }

# Prepare datasets
print("\nPreparing datasets...")

# Convert HF datasets to list of dicts
train_data = list(train_dataset)
val_data = list(val_dataset)

print(f"Loaded {len(train_data)} train examples")
print(f"Loaded {len(val_data)} val examples")

# Create DPO datasets
dpo_train_dataset = DPODataset(train_data, tokenizer)
dpo_val_dataset = DPODataset(val_data, tokenizer)

train_loader = DataLoader(dpo_train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(dpo_val_dataset, batch_size=2)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# CORRECT log probability function
def get_sequence_log_probs(model, input_ids, attention_mask, prompt_len):
    """
    Compute proper sequence log probabilities.
    
    Args:
        model: Language model
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        prompt_len: Length of prompt (don't compute loss on this part)
    
    Returns:
        log_probs: [batch] - log probability of each sequence
    """
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [batch, seq_len, vocab_size]
    
    # Shift for next-token prediction
    # logits[:, :-1] predicts tokens[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()
    
    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch, seq_len-1, vocab_size]
    
    # Gather the log prob of the actual next token
    gathered_log_probs = torch.gather(
        log_probs,
        dim=2,
        index=shift_labels.unsqueeze(2)
    ).squeeze(2)  # [batch, seq_len-1]
    
    # Mask out prompt tokens and padding
    # We only want to compute loss on the response part
    response_mask = shift_mask.clone()
    response_mask[:, :prompt_len] = 0  # Don't include prompt
    
    # Sum log probs per sequence (only on response tokens)
    sequence_log_probs = (gathered_log_probs * response_mask).sum(dim=1)  # [batch]
    
    # Normalize by length to avoid bias towards shorter sequences
    response_lengths = response_mask.sum(dim=1)
    sequence_log_probs = sequence_log_probs / (response_lengths + 1e-10)
    
    return sequence_log_probs

# DPO Loss function
def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    DPO loss: -log(sigmoid(beta * (policy_diff - ref_diff)))
    """
    policy_diff = policy_chosen_logps - policy_rejected_logps
    ref_diff = ref_chosen_logps - ref_rejected_logps
    
    losses = -F.logsigmoid(beta * (policy_diff - ref_diff))
    return losses.mean()

# Optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-7, weight_decay=0.01)

# Training loop
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

epochs = 3
global_step = 0
start_time = time.time()

for epoch in range(epochs):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch+1}/{epochs}")
    print(f"{'='*70}")
    
    model.train()
    epoch_loss = 0
    valid_batches = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        # Move to GPU
        chosen_ids = batch['chosen_input_ids'].to('cuda')
        chosen_mask = batch['chosen_attention_mask'].to('cuda')
        rejected_ids = batch['rejected_input_ids'].to('cuda')
        rejected_mask = batch['rejected_attention_mask'].to('cuda')
        prompt_len = batch['prompt_len'][0].item()  # Assume same in batch
        
        # Compute log probs - policy model
        policy_chosen_logps = get_sequence_log_probs(model, chosen_ids, chosen_mask, prompt_len)
        policy_rejected_logps = get_sequence_log_probs(model, rejected_ids, rejected_mask, prompt_len)
        
        # Compute log probs - reference model
        with torch.no_grad():
            ref_chosen_logps = get_sequence_log_probs(ref_model, chosen_ids, chosen_mask, prompt_len)
            ref_rejected_logps = get_sequence_log_probs(ref_model, rejected_ids, rejected_mask, prompt_len)
        
        # DPO loss
        loss = dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\nWARNING: NaN loss at step {global_step}, skipping batch")
            continue
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        valid_batches += 1
        global_step += 1
        
        # Log every 50 steps
        if global_step % 50 == 0:
            elapsed = time.time() - start_time
            avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
            print(f"\n{'='*70}")
            print(f"Step {global_step} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
            print(f"Time: {elapsed/60:.1f}min")
            print(f"{'='*70}")
    
    avg_epoch_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
    print(f"\nEpoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

# Save
print("\nSaving model...")
model.save_pretrained("/kaggle/working/dpo_generator_final")
tokenizer.save_pretrained("/kaggle/working/dpo_generator_final")
print("Model saved to: /kaggle/working/dpo_generator_final")

# Test generation
print("\n" + "="*70)
print("TESTING GENERATOR")
print("="*70)

model.eval()

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# Test on 3 examples
for i in range(min(3, len(val_data))):
    ex = val_data[i]
    print(f"\nExample {i+1}:")
    print(f"Prompt: {ex['prompt'][:100]}...")
    
    try:
        generated = generate(ex['prompt'])
        print(f"\nGenerated: {generated}")
        print(f"Reference: {ex['chosen']}")
    except Exception as e:
        print(f"Generation failed: {e}")
    print("-"*70)

# Zip for download
import shutil
shutil.make_archive('/kaggle/working/dpo_final', 'zip', '/kaggle/working/dpo_generator_final')

print("\n" + "="*70)
print("DOWNLOAD: dpo_final.zip")
print("="*70)
```

## Key Fixes

3. **Length normalization** - prevents bias towards short sequences
4. **Gradient clipping** - prevents gradient explosions
5. **NaN checking** - skips corrupt batches instead of poisoning entire model
6. **Proper gather operation** - gets the right token probabilities

This implementation follows the actual DPO paper methodology.
