# 🤖 KAGGLE DPO GENERATOR TRAINING GUIDE
## Chapter 13: Training Generator with Maxim Feedback

**What you'll build:** A dialogue generator that learns to avoid Gricean violations!

**Training time:** 4-6 hours on Kaggle GPU T4 x2

---

## STEP 1: Upload DPO Data to Kaggle

### Create a New Dataset

1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload these 2 files from `data_processed/dpo_data/`:
   - `dpo_train.json` (3.2 MB)
   - `dpo_val.json` (358 KB)

4. Name it: **"gricebench-dpo-data"**
5. Click **"Create"**

---

## STEP 2: Create New Notebook

1. Go to: https://www.kaggle.com/code
2. Click **"New Notebook"**
3. Settings → **Accelerator** → **GPU T4 x2**
4. Right sidebar → **+ Add data** → Search "gricebench-dpo-data" → Add

---

## CELL 1: Setup & Install

```python
print("="*70)
print("Chapter 13: DPO Generator Training")
print("="*70)

# Check GPU
import torch
if not torch.cuda.is_available():
    raise RuntimeError("NO GPU! Enable GPU T4 x2 in Settings")

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Install latest stable versions (all have pre-built wheels)
print("\nInstalling TRL and dependencies...")
print("Using latest versions with pre-built wheels...")
!pip install transformers trl peft accelerate --upgrade

# Verify installation
print("\n" + "="*70)
print("VERIFYING INSTALLATION")
print("="*70)

try:
    import trl
    import transformers
    import peft
    print(f"TRL version: {trl.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"PEFT version: {peft.__version__}")
    print("\n✓ All libraries installed successfully!")
    print("IMPORTANT: Restart kernel now, then run all cells from Cell 1")
except ImportError as e:
    print(f"\n✗ ERROR: {e}")
    print("Installation may have failed - check output above")
```

---

## CELL 2: Load DPO Data

```python
import json
from pathlib import Path
from datasets import Dataset

# Data paths
DATA_DIR = Path("/kaggle/input/gricebench-dpo-data")

# Load
with open(DATA_DIR / "dpo_train.json") as f:
    train_data = json.load(f)

with open(DATA_DIR / "dpo_val.json") as f:
    val_data = json.load(f)

print(f"Train: {len(train_data):,} examples")
print(f"Val: {len(val_data):,} examples")

# Convert to HF Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Sample
print("\nSample DPO Example:")
print(f"Prompt: {train_data[0]['prompt'][:150]}...")
print(f"\nChosen (cooperative): {train_data[0]['chosen'][:100]}...")
print(f"\nRejected (violated): {train_data[0]['rejected'][:100]}...")
```

---

## CELL 3: Load GPT-2 Base Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use GPT-2 Medium (355M params)
MODEL_NAME = "gpt2-medium"

print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load model WITHOUT device_map (single GPU)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)

# Move to GPU manually
model = model.to('cuda')

params = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {params:,} parameters")
print(f"Model device: {next(model.parameters()).device}")
```

---

## CELL 4: Configure Training (Version-Agnostic)

```python
import inspect

# Check TRL version and API
import trl
print(f"Using TRL version: {trl.__version__}")

# Use DPOConfig (latest TRL) or fall back to TrainingArguments
try:
    from trl import DPOConfig
    print("Using DPOConfig (latest TRL)")
    
    training_args = DPOConfig(
        output_dir="/kaggle/working/dpo_generator",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=True,  # Use bf16 instead of fp16 for latest
        gradient_checkpointing=True,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        beta=0.1,  # DPO parameter in config
    )
except ImportError:
    from transformers import TrainingArguments
    print("Using TrainingArguments (older TRL)")
    
    training_args = TrainingArguments(
        output_dir="/kaggle/working/dpo_generator",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
    )

print("Training configuration ready")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
```

---

## CELL 5: Train with Auto-Detected API (4-6 hours)

```python
from trl import DPOTrainer
from transformers import TrainerCallback
import time
import inspect

# Custom logging callback
class DetailedLoggingCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            elapsed = time.time() - self.start_time
            hours, mins = int(elapsed // 3600), int((elapsed % 3600) // 60)
            
            print(f"\n{'='*70}")
            print(f"Step {state.global_step} | {hours}h {mins}m elapsed")
            print(f"{'='*70}")
            if 'loss' in logs: print(f"  Loss: {logs['loss']:.4f}")
            if 'eval_loss' in logs: print(f"  Eval Loss: {logs['eval_loss']:.4f}")
            if 'learning_rate' in logs: print(f"  LR: {logs['learning_rate']:.2e}")
            
            if state.global_step > 0:
                remaining = (elapsed / state.global_step) * (state.max_steps - state.global_step)
                r_hours, r_mins = int(remaining // 3600), int((remaining % 3600) // 60)
                print(f"  Est. remaining: {r_hours}h {r_mins}m")
            print(f"{'='*70}\n")

print("="*70)
print("INITIALIZING DPO TRAINER")
print("="*70)

# Ensure model is on GPU
print("\nMoving model to GPU...")
model = model.to('cuda')
model.train()
print(f"Model device: {next(model.parameters()).device}")

# Detect DPOTrainer signature
sig = inspect.signature(DPOTrainer.__init__)
params = sig.parameters

print(f"\nDetected parameters: {list(params.keys())[:10]}...")

# Build kwargs based on what's actually accepted
trainer_kwargs = {
    'model': model,
    'ref_model': None,  # Let DPO create reference model
    'args': training_args,
    'train_dataset': train_dataset,
    'eval_dataset': val_dataset,
    'callbacks': [DetailedLoggingCallback()],
}

# Add tokenizer/processing_class based on what API accepts
if 'processing_class' in params:
    trainer_kwargs['processing_class'] = tokenizer
    print("✓ Using processing_class (new API)")
elif 'tokenizer' in params:
    trainer_kwargs['tokenizer'] = tokenizer
    print("✓ Using tokenizer (old API)")

# Beta should already be in DPOConfig, but add if needed
if 'beta' in params and not hasattr(training_args, 'beta'):
    trainer_kwargs['beta'] = 0.1
    print("✓ Adding beta=0.1 as parameter")

# Don't add max_length - it's deprecated and causes warnings
print(f"\nInitializing with: {list(trainer_kwargs.keys())}")

# Create trainer
print("\nCreating DPOTrainer...")
trainer = DPOTrainer(**trainer_kwargs)
print("✓ Trainer created")

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print(f"Logging every {training_args.logging_steps} steps")
print(f"Estimated time: 4-6 hours")
print("\nYou should see progress output within 2-3 minutes...")
print("First log at step 50 (~10-15 minutes)\n")

# Train
print("Calling trainer.train()...")
trainer.train()

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

# Save
print("\nSaving model...")
trainer.save_model("/kaggle/working/dpo_generator_final")
tokenizer.save_pretrained("/kaggle/working/dpo_generator_final")
print("Model saved to: /kaggle/working/dpo_generator_final")
```
```

---

## CELL 6: Test & Download

```python
# Test generation
print("="*70)
print("TESTING GENERATOR")
print("="*70)

# Load trained model
test_model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/working/dpo_generator_final",
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = test_model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
        do_sample=True,
        temperature=0.7,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# Test on 3 examples
for i in range(3):
    ex = val_data[i]
    print(f"\nExample {i+1}:")
    print(f"Prompt: {ex['prompt'][:100]}...")
    
    generated = generate(ex['prompt'])
    print(f"\nGenerated: {generated}")
    print(f"Reference: {ex['chosen']}")
    print("-"*70)

# Zip for download
import shutil
shutil.make_archive('/kaggle/working/dpo_final', 'zip', '/kaggle/working/dpo_generator_final')

print("\n" + "="*70)
print("DOWNLOAD: dpo_final.zip")
print("="*70)
print("Right sidebar → Output → dpo_final.zip → Download")

```

---

## SUCCESS CRITERIA

Your DPO generator should:
- ✅ Training loss decreases
- ✅ Eval loss < 2.0
- ✅ Generates fluent, cooperative responses
- ✅ Better than baseline GPT-2

---

## AFTER DOWNLOAD

1. Extract `dpo_final.zip`
2. Move to `GriceBench/models/dpo_generator/`
3. Run evaluation (see `scripts/evaluate_generator.py`)

---

## TROUBLESHOOTING

### "CUDA out of memory"
- Reduce batch_size to 1
- Increase gradient_accumulation_steps to 16

### "Loss is NaN"
- Lower learning rate to 1e-7
- Check data has no empty responses

### "Model not improving"
- Try higher beta (0.2 or 0.3)
- Verify preference pairs formatted correctly

---

**Estimated GPU time: 5-6 hours** (well under 30-hour Kaggle limit!)
