# Chapter 13: Generator Training with Maxim Feedback - KAGGLE IMPLEMENTATION GUIDE

**Component C: Training a Dialogue Generator to avoid violations**

---

## Overview

This chapter implements **Direct Preference Optimization (DPO)** to train a dialogue generator that produces cooperative responses by learning from your detector's feedback.

### What You're Building:

- **Input:** Conversation context + Evidence
- **Output:** Cooperative response (no maxim violations)
- **Method:** Train with preference pairs (repaired > violated responses)
- **Platform:** Kaggle GPU (30 hours/week available)

### Timeline Estimate:
- Data preparation: 1-2 days (local)
- Model training: 6-8 hours (Kaggle GPU)
- Evaluation: 2-3 days
- **Total:** ~1 week

---

## Prerequisites

### What You Already Have: ✅
1. **Repair pairs:** 50K examples of violated → repaired responses
2. **Detector model:** 96% F1 for violation detection  
3. **Test data:** 652 examples for evaluation
4. **Kaggle account:** With 30 GPU hours/week

### What You Need to Create:
1. **DPO preference dataset:** Format repair pairs for DPO
2. **Base generator model:** Fine-tune GPT-2 or T5 on dialogue first
3. **DPO training setup:** Kaggle notebook with TRL library
4. **Evaluation pipeline:** Measure violation rates

---

## PART 1: Data Preparation (Local - 1 Day)

### Step 1.1: Create Preference Pairs

Your repair data already contains natural preference pairs:
- **Preferred (y_w):** Repaired response (cooperative)
- **Dispreferred (y_l):** Original violated response

Create a script to convert your repair data to DPO format.

**File:** `scripts/prepare_dpo_data.py`

```python
"""
Convert repair data to DPO preference format
"""

import json
from pathlib import Path

def create_dpo_dataset():
    """Convert repair pairs to DPO format."""
    
    # Load repair data
    with open('data_processed/repair_data/repair_train.json', 'r') as f:
        repair_data = json.load(f)
    
    dpo_data = []
    
    for example in repair_data:
        # Extract components
        context = extract_context(example['input_text'])
        evidence = extract_evidence(example['input_text'])
        violated_response = extract_response(example['input_text'])
        repaired_response = example['target_text']
        
        # Create DPO format
        dpo_example = {
            'prompt': f"Context: {context}\nEvidence: {evidence}\nGenerate cooperative response:",
            'chosen': repaired_response,  # Preferred (cooperative)
            'rejected': violated_response  # Dispreferred (violated)
        }
        
        dpo_data.append(dpo_example)
    
    # Save
    output_dir = Path('data_processed/dpo_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split: 90% train, 10% validation
    split_point = int(len(dpo_data) * 0.9)
    train_data = dpo_data[:split_point]
    val_data = dpo_data[split_point:]
    
    with open(output_dir / 'dpo_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_dir / 'dpo_val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Created {len(train_data)} training examples")
    print(f"Created {len(val_data)} validation examples")

def extract_context(input_text):
    """Extract context from repair input."""
    # Parse the structured input
    # Format: [REPAIR] [VIOLATION=X] [CONTEXT] ... [EVIDENCE] ... [RESPONSE] ...
    parts = input_text.split('[CONTEXT]')
    if len(parts) > 1:
        context_part = parts[1].split('[EVIDENCE]')[0].strip()
        return context_part
    return ""

def extract_evidence(input_text):
    """Extract evidence from repair input."""
    parts = input_text.split('[EVIDENCE]')
    if len(parts) > 1:
        evidence_part = parts[1].split('[RESPONSE]')[0].strip()
        return evidence_part
    return ""

def extract_response(input_text):
    """Extract violated response from repair input."""
    parts = input_text.split('[RESPONSE]')
    if len(parts) > 1:
        return parts[1].strip()
    return ""

if __name__ == "__main__":
    create_dpo_dataset()
```

**Run this locally:**
```bash
python scripts/prepare_dpo_data.py
```

**Expected output:**
- `data_processed/dpo_data/dpo_train.json` (~45K examples)
- `data_processed/dpo_data/dpo_val.json` (~5K examples)

---

## PART 2: Base Model Fine-Tuning (Optional but Recommended)

### Why Fine-Tune First?

DPO works best when starting from a model already trained on your domain. You have two options:

**Option A:** Use pre-trained GPT-2 or Flan-T5 directly
- Faster (skip this step)
- Less domain-adapted

**Option B:** Fine-tune on Topical-Chat first
- Better domain fit
- Adds 2-4 hours GPU time

**Recommendation:** Start with Option A (faster), try Option B if results are weak.

---

## PART 3: DPO Training on Kaggle (6-8 Hours GPU)

### Step 3.1: Upload Data to Kaggle

1. Zip your DPO data:
   ```bash
   # On local machine
   cd data_processed
   zip -r dpo_data.zip dpo_data/
   ```

2. Upload to Kaggle Datasets:
   - Go to kaggle.com/datasets
   - Create new dataset "gricebench-dpo-data"
   - Upload `dpo_data.zip`

### Step 3.2: Create Kaggle Notebook

Create `KAGGLE_DPO_TRAINING.md` locally, then transfer to Kaggle notebook.

**6-Cell Kaggle Notebook Structure:**

---

### CELL 1: Setup & Check

```python
print("="*70)
print("Chapter 13: DPO Generator Training")
print("="*70)

# Check GPU
import torch
if not torch.cuda.is_available():
    raise RuntimeError("NO GPU! Enable GPU T4 x2 in Settings")

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Install TRL (DPO library)
!pip install -q trl==0.7.4
!pip install -q peft==0.7.1

print("\nLibraries installed!")
```

---

### CELL 2: Load Data

```python
import json
from pathlib import Path
from datasets import Dataset

# Data paths
DATA_DIR = Path("/kaggle/input/gricebench-dpo-data/dpo_data")

# Load DPO data
with open(DATA_DIR / "dpo_train.json") as f:
    train_data = json.load(f)

with open(DATA_DIR / "dpo_val.json") as f:
    val_data = json.load(f)

print(f"Train examples: {len(train_data):,}")
print(f"Val examples: {len(val_data):,}")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Show example
print("\nExample:")
print(f"Prompt: {train_data[0]['prompt'][:100]}...")
print(f"Chosen: {train_data[0]['chosen'][:80]}...")
print(f"Rejected: {train_data[0]['rejected'][:80]}...")
```

---

### CELL 3: Load Base Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model choice: GPT-2 medium (good balance of size/performance)
MODEL_NAME = "gpt2-medium"  # 355M params

print(f"\nLoading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,  # Use FP16 for speed
    device_map="auto"
)

print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
```

---

### CELL 4: DPO Training Configuration

```python
from trl import DPOTrainer, DPOConfig

# DPO Configuration
training_args = DPOConfig(
    output_dir="/kaggle/working/dpo_generator",
    
    # Training
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Small batch for GPU memory
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Effective batch = 2*8 = 16
    
    # Learning
    learning_rate=5e-7,  # Very low for DPO
    lr_scheduler_type="cosine",
    warmup_steps=100,
    
    # DPO-specific
    beta=0.1,  # DPO temperature parameter
    
    # Optimization
    fp16=True,
    gradient_checkpointing=True,
    
    # Logging
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    
    # Save best
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,
)

print("DPO Configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Beta (DPO temperature): {training_args.beta}")
```

---

### CELL 5: Run DPO Training (4-6 hours)

```python
from trl import DPOTrainer

print("="*70)
print("TRAINING DPO GENERATOR")
print("="*70)

# Initialize DPO Trainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    max_length=512,
    max_prompt_length=256,
)

# Train!
print("\nStarting DPO training...")
print("This will take 4-6 hours. Estimated completion: [TIME + 6hrs]")

trainer.train()

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

# Save final model
trainer.save_model("/kaggle/working/dpo_generator_final")
tokenizer.save_pretrained("/kaggle/working/dpo_generator_final")

print("\nModel saved to /kaggle/working/dpo_generator_final")
```

---

### CELL 6: Test Generation

```python
# Load trained model for inference
trained_model = AutoModelForCausalLM.from_pretrained(
    "/kaggle/working/dpo_generator_final",
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_response(context, evidence):
    """Generate cooperative response."""
    prompt =f"""Context: {context}
Evidence: {evidence}
Generate cooperative response:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = trained_model.generate(
        **inputs,
        max_new_tokens=128,
        num_beams=4,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the generated part
    response = response.replace(prompt, "").strip()
    
    return response

# Test on validation examples
print("="*70)
print("TESTING GENERATOR")
print("="*70)

for i in range(3):
    ex = val_data[i]
    context = ex['prompt'].split('Evidence:')[0].replace('Context:', '').strip()
    evidence = ex['prompt'].split('Evidence:')[1].replace('Generate cooperative response:', '').strip()
    
    print(f"\nExample {i+1}:")
    print(f"Context: {context[:100]}...")
    
    generated = generate_response(context, evidence)
    print(f"\nGenerated: {generated}")
    print(f"Reference (cooperative): {ex['chosen']}")
    print("-"*70)

# Download button instructions
import shutil
shutil.make_archive('/kaggle/working/dpo_final', 'zip', '/kaggle/working/dpo_generator_final')

print("\n" + "="*70)
print("DOWNLOAD: dpo_final.zip from Output tab")
print("="*70)
```

---

## PART 4: Evaluation (Local - 2-3 Days)

After downloading the trained generator, evaluate it comprehensively.

### Step 4.1: Automatic Evaluation with Detector

**File:** `scripts/evaluate_generator.py`

```python
"""
Evaluate generator by measuring violation rates
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Assuming you have detector loading code
from evaluate_detector import load_detector

def evaluate_generator():
    """
    Generate responses and measure violation rates.
    """
    
    # Load generator
    print("Loading DPO generator...")
    tokenizer = AutoTokenizer.from_pretrained('models/dpo_generator')
    generator = AutoModelForCausalLM.from_pretrained('models/dpo_generator')
    generator.eval()
    
    # Load detector
    print("Loading detector...")
    detector = load_detector()
    
    # Load test data
    with open('data_processed/dpo_data/dpo_val.json') as f:
        test_data = json.load(f)[:100]  # Test on 100 examples
    
    violations_count = {
        'quantity': 0,
        'quality': 0,
        'relation': 0,
        'manner': 0,
        'none': 0
    }
    
    print(f"\nGenerating {len(test_data)} responses...")
    
    for ex in test_data:
        # Generate response
        prompt = ex['prompt']
        inputs = tokenizer(prompt, return_tensors='pt')
        
        with torch.no_grad():
            outputs = generator.generate(**inputs, max_new_tokens=128, num_beams=4)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, '').strip()
        
        # Run detector
        # [Create proper detector input format]
        detector_input = format_for_detector(prompt, response)
        predictions = run_detector(detector, detector_input)
        
        # Count violations
        if not any(predictions.values()):
            violations_count['none'] += 1
        else:
            for maxim, is_violated in predictions.items():
                if is_violated:
                    violations_count[maxim] += 1
    
    # Report
    print("\n" + "="*70)
    print("GENERATOR EVALUATION RESULTS")
    print("="*70)
    
    total = len(test_data)
    print(f"\nViolation Rates:")
    for maxim, count in violations_count.items():
        rate = count / total
        print(f"  {maxim.capitalize()}: {rate:.1%} ({count}/{total})")
    
    cooperative_rate = violations_count['none'] / total
    print(f"\n  Cooperative (no violations): {cooperative_rate:.1%}")
    
    # Save results
    results = {
        'violation_counts': violations_count,
        'total_examples': total,
        'cooperative_rate': cooperative_rate
    }
    
    with open('results/generator_evaluation/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
```

**Expected Results:**
- Cooperative rate: 60-80% (no violations)
- Lower violation rates than baseline
- Quality violations should be lowest (detector helps most here)

---

## Timeline & Resource Usage

### Kaggle GPU Hours:

| Task | GPU Hours | When |
|------|-----------|------|
| DPO Training (3 epochs) | 4-6 hours | Week 1 |
| Evaluation generation | 1 hour | Week 1 |
| Ablation experiments (optional) | 3-4 hours | Week 2 |
| **Total** | **8-11 hours** | **Under 30hr limit!** |

### Week-by-Week Plan:

**Week 1:**
- Day 1-2: Prepare DPO data locally
- Day 3: Upload to Kaggle
- Day 4-5: Run DPO training (6 hours GPU)
- Day 6-7: Download and quick test

**Week 2:**
- Day 1-3: Full evaluation (detector-based + examples)
- Day 4-5: Error analysis
- Day 6-7: Documentation

---

## Success Criteria

✅ **Training succeeds if:**
- Loss decreases over epochs
- Model generates fluent responses
- No NaN losses or crashes

✅ **Generator succeeds if:**
- Cooperative rate > 65%
- Better than baseline (pre-trained model without DPO)
- Quality violations < 10%

---

## Next Steps After Chapter 13

Once generator training completes:

1. **Update task.md:** Mark Chapter 13 complete
2. **Document results:** Add to walkthrough
3. **Move to Chapter 15:** Paper writing with full system

You'll have:
- ✅ Complete 3-component system (detect → repair → generate)
- ✅ Strong publication story
- ✅ Ready for top-tier conference submission

---

## Troubleshooting

### If DPO training fails:
- Reduce batch size to 1
- Increase gradient accumulation to 16
- Lower learning rate to 1e-7

### If generation is incoherent:
- Check prompt format
- Try greedy decoding (num_beams=1, do_sample=False)
- Verify tokenizer pad token set correctly

### If violation rates don't improve:
- Check detector is loading correctly
- Verify preference pairs are formatted right
- Try higher beta value (0.2 or 0.3)

---

**Ready to start? Let's begin with data preparation!**
