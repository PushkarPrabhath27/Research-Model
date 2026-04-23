# Chapter 14: DPO Generator Evaluation - Kaggle Notebook
# Complete evaluation of DPO model vs baseline GPT-2

# ============================================================================
# CELL 1: Install Required Packages (if needed)
# ============================================================================
# Most packages are pre-installed in Kaggle, but let's verify versions

import sys
print("Python version:", sys.version)

import transformers
import torch
import peft

print(f"✓ transformers: {transformers.__version__}")
print(f"✓ torch: {torch.__version__}")
print(f"✓ peft: {peft.__version__}")
print("\n✅ All packages ready!")

# ============================================================================
# CELL 2: Setup and Configuration
# ============================================================================

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from peft import PeftModel

# Configuration
CONFIG = {
    'dpo_model_path': '/kaggle/input/dpo-generator-model',  # Files are in dataset root
    'detector_model_path': '/kaggle/input/gricean-maxim-detector-model',
    'test_data_path': '/kaggle/input/dpo-test-data/dpo_val.json',
    'num_examples': 100,  # Number of examples to evaluate
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_new_tokens': 100,
    'temperature': 0.7,
    'top_p': 0.9,
}

print("="*70)
print("CONFIGURATION")
print("="*70)
for key, value in CONFIG.items():
    print(f"{key}: {value}")
print("="*70)

# ============================================================================
# CELL 3: Load Models
# ============================================================================

print("\n" + "="*70)
print("LOADING MODELS")
print("="*70)

# 1. Load Baseline GPT-2 Medium
print("\n[1/4] Loading baseline GPT-2 Medium...")
baseline_model = AutoModelForCausalLM.from_pretrained(
    "gpt2-medium",
    torch_dtype=torch.float16 if CONFIG['device'] == 'cuda' else torch.float32
).to(CONFIG['device'])
baseline_tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
baseline_tokenizer.padding_side = "left"
print(f"✓ Baseline loaded: {baseline_model.num_parameters():,} parameters")

# 2. Load DPO Model with LoRA Adapters
print("\n[2/4] Loading DPO model with LoRA adapters...")
base_model = AutoModelForCausalLM.from_pretrained(
    "gpt2-medium",
    torch_dtype=torch.float16 if CONFIG['device'] == 'cuda' else torch.float32
)
dpo_model = PeftModel.from_pretrained(
    base_model,
    CONFIG['dpo_model_path']
).to(CONFIG['device'])
dpo_tokenizer = AutoTokenizer.from_pretrained(CONFIG['dpo_model_path'])
dpo_tokenizer.pad_token = dpo_tokenizer.eos_token
dpo_tokenizer.padding_side = "left"
print(f"✓ DPO model loaded with LoRA adapters")

# 3. Load Detector Model (PyTorch checkpoint format)
print("\n[3/4] Loading detector model...")
# The detector is saved as a PyTorch checkpoint, not HuggingFace format
# We need to load the architecture and then load the weights

# Load the base model architecture (DeBERTa-v3-base with 4 labels for 4 maxims)
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer

detector_model = DebertaV2ForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=4  # 4 maxims: quantity, quality, relation, manner
)

# Load the trained weights from PyTorch checkpoint
checkpoint_path = f"{CONFIG['detector_model_path']}/best_model.pt"
checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'], weights_only=False)

# Extract state dict (handle different checkpoint formats)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Fix key mismatch: The checkpoint has 'encoder.' prefix that needs to be removed
# Checkpoint keys: encoder.encoder.layer.0... -> Model expects: deberta.encoder.layer.0...
new_state_dict = {}
for key, value in state_dict.items():
    # Remove 'encoder.' prefix and replace with 'deberta.'
    if key.startswith('encoder.'):
        new_key = 'deberta.' + key[len('encoder.'):]  # Strip 'encoder.' and add 'deberta.'
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# Load the corrected state dict
detector_model.load_state_dict(new_state_dict, strict=False)  # strict=False to ignore missing pooler weights

detector_model = detector_model.to(CONFIG['device'])

# Load tokenizer (use the base model's tokenizer)
detector_tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
print(f"✓ Detector loaded from PyTorch checkpoint")

# 4. Set to Eval Mode
print("\n[4/4] Setting models to evaluation mode...")
baseline_model.eval()
dpo_model.eval()
detector_model.eval()
print("✓ All models ready!")

print("\n" + "="*70)
print("MODEL LOADING COMPLETE")
print("="*70)

# ============================================================================
# CELL 4: Define Helper Functions
# ============================================================================

maxims = ['quantity', 'quality', 'relation', 'manner']

def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """Generate response from a model"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(CONFIG['device'])
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text[len(prompt):].strip()
    return response

def detect_violations(context, response, threshold=0.5):
    """Run detector on a response"""
    detector_input = f"[CONTEXT] {context} [RESPONSE] {response}"
    
    inputs = detector_tokenizer(
        detector_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(CONFIG['device'])
    
    with torch.no_grad():
        outputs = detector_model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    result = {}
    for i, maxim in enumerate(maxims):
        result[f"{maxim}_prob"] = float(probs[i])
        result[f"{maxim}_violated"] = bool(probs[i] > threshold)
    
    result['any_violation'] = any(result[f"{m}_violated"] for m in maxims)
    result['cooperative'] = not result['any_violation']
    
    return result

print("✅ Helper functions defined!")

# ============================================================================
# CELL 5: Load Test Data
# ============================================================================

print("\n" + "="*70)
print("LOADING TEST DATA")
print("="*70)

with open(CONFIG['test_data_path'], 'r') as f:
    test_data = json.load(f)

# Handle different data formats
if isinstance(test_data, dict) and 'examples' in test_data:
    test_data = test_data['examples']

# Limit to configured number of examples
test_data = test_data[:CONFIG['num_examples']]

print(f"✓ Loaded {len(test_data)} test examples")
print(f"Sample prompt: {test_data[0].get('prompt', '')[:100]}...")
print("="*70)

# ============================================================================
# CELL 6: Run Evaluation
# ============================================================================

print("\n" + "="*70)
print(f"EVALUATING ON {len(test_data)} EXAMPLES")
print("="*70)

results = {
    'baseline': [],
    'dpo': [],
    'examples': []
}

for i, example in enumerate(tqdm(test_data, desc="Generating & Detecting")):
    prompt = example.get('prompt', '')
    context = example.get('context', prompt)
    
    # Generate from both models
    baseline_response = generate_response(
        baseline_model,
        baseline_tokenizer,
        prompt,
        max_new_tokens=CONFIG['max_new_tokens'],
        temperature=CONFIG['temperature'],
        top_p=CONFIG['top_p']
    )
    
    dpo_response = generate_response(
        dpo_model,
        dpo_tokenizer,
        prompt,
        max_new_tokens=CONFIG['max_new_tokens'],
        temperature=CONFIG['temperature'],
        top_p=CONFIG['top_p']
    )
    
    # Detect violations
    baseline_violations = detect_violations(context, baseline_response)
    dpo_violations = detect_violations(context, dpo_response)
    
    results['baseline'].append(baseline_violations)
    results['dpo'].append(dpo_violations)
    
    # Store first 20 examples for inspection
    if i < 20:
        results['examples'].append({
            'prompt': prompt[:100] + "..." if len(prompt) > 100 else prompt,
            'baseline_response': baseline_response,
            'dpo_response': dpo_response,
            'baseline_violations': baseline_violations,
            'dpo_violations': dpo_violations
        })

print("\n✅ Evaluation complete!")

# ============================================================================
# CELL 7: Calculate Metrics
# ============================================================================

print("\n" + "="*70)
print("CALCULATING METRICS")
print("="*70)

baseline_data = results['baseline']
dpo_data = results['dpo']
n = len(baseline_data)

metrics = {
    'baseline': {},
    'dpo': {},
    'improvements': {}
}

# Per-maxim violation rates
for maxim in maxims:
    baseline_violations = sum(r[f"{maxim}_violated"] for r in baseline_data)
    dpo_violations = sum(r[f"{maxim}_violated"] for r in dpo_data)
    
    baseline_rate = baseline_violations / n
    dpo_rate = dpo_violations / n
    
    metrics['baseline'][f"{maxim}_violation_rate"] = baseline_rate
    metrics['dpo'][f"{maxim}_violation_rate"] = dpo_rate
    metrics['improvements'][f"{maxim}_improvement"] = baseline_rate - dpo_rate
    metrics['improvements'][f"{maxim}_improvement_pct"] = (
        (baseline_rate - dpo_rate) / baseline_rate * 100 if baseline_rate > 0 else 0
    )

# Overall cooperative rate
baseline_cooperative = sum(r['cooperative'] for r in baseline_data) / n
dpo_cooperative = sum(r['cooperative'] for r in dpo_data) / n

metrics['baseline']['overall_cooperative_rate'] = baseline_cooperative
metrics['dpo']['overall_cooperative_rate'] = dpo_cooperative
metrics['improvements']['cooperative_improvement'] = dpo_cooperative - baseline_cooperative
metrics['improvements']['cooperative_improvement_pct'] = (
    (dpo_cooperative - baseline_cooperative) / (1 - baseline_cooperative) * 100
    if baseline_cooperative < 1 else 0
)

print("✅ Metrics calculated!")

# ============================================================================
# CELL 8: Display Results
# ============================================================================

print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)

print("\n📊 VIOLATION RATES:")
print("-"*70)
print(f"{'Maxim':<15} {'Baseline':<12} {'DPO':<12} {'Improvement':<15}")
print("-"*70)

for maxim in maxims:
    baseline_rate = metrics['baseline'][f"{maxim}_violation_rate"]
    dpo_rate = metrics['dpo'][f"{maxim}_violation_rate"]
    improvement = metrics['improvements'][f"{maxim}_improvement"]
    improvement_pct = metrics['improvements'][f"{maxim}_improvement_pct"]
    
    print(f"{maxim.capitalize():<15} {baseline_rate:>6.1%}      {dpo_rate:>6.1%}      "
          f"{improvement:>+6.1%} ({improvement_pct:>+5.1f}%)")

print("-"*70)
baseline_coop = metrics['baseline']['overall_cooperative_rate']
dpo_coop = metrics['dpo']['overall_cooperative_rate']
coop_improvement = metrics['improvements']['cooperative_improvement']
coop_improvement_pct = metrics['improvements']['cooperative_improvement_pct']

print(f"{'Cooperative':<15} {baseline_coop:>6.1%}      {dpo_coop:>6.1%}      "
      f"{coop_improvement:>+6.1%} ({coop_improvement_pct:>+5.1f}%)")
print("="*70)

print("\n✅ SUMMARY:")
improvements = [metrics['improvements'][f"{m}_improvement"] > 0 for m in maxims]
print(f"  • Maxims improved: {sum(improvements)}/4")
print(f"  • Overall cooperative rate improved: {coop_improvement > 0}")

if sum(improvements) >= 3:
    print("\n🎉 EXCELLENT! DPO training significantly improved generation quality!")
elif sum(improvements) >= 2:
    print("\n✓ GOOD! DPO training improved generation on multiple maxims.")
else:
    print("\n⚠️  Mixed results. DPO training showed limited improvement.")

# ============================================================================
# CELL 9: Show Example Outputs
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE GENERATED RESPONSES (First 5)")
print("="*70)

for i, ex in enumerate(results['examples'][:5], 1):
    print(f"\n{'='*70}")
    print(f"Example {i}:")
    print(f"{'='*70}")
    print(f"\nPrompt: {ex['prompt']}\n")
    
    print("Baseline Response:")
    print(f"  {ex['baseline_response']}")
    baseline_viols = [m for m in maxims if ex['baseline_violations'][f'{m}_violated']]
    print(f"  Violations: {', '.join(baseline_viols) if baseline_viols else 'None'}\n")
    
    print("DPO Response:")
    print(f"  {ex['dpo_response']}")
    dpo_viols = [m for m in maxims if ex['dpo_violations'][f'{m}_violated']]
    print(f"  Violations: {', '.join(dpo_viols) if dpo_viols else 'None'}")

# ============================================================================
# CELL 10: Save Results
# ============================================================================

# Save full results
output_data = {
    'config': CONFIG,
    'metrics': metrics,
    'examples': results['examples'],
    'num_evaluated': len(test_data)
}

with open('dpo_evaluation_results.json', 'w') as f:
    json.dump(output_data, f, indent=2)

# Save metrics as CSV
with open('violation_rates.csv', 'w') as f:
    f.write("Maxim,Baseline_Rate,DPO_Rate,Improvement,Improvement_Pct\n")
    for maxim in maxims:
        baseline = metrics['baseline'][f"{maxim}_violation_rate"]
        dpo = metrics['dpo'][f"{maxim}_violation_rate"]
        improvement = metrics['improvements'][f"{maxim}_improvement"]
        improvement_pct = metrics['improvements'][f"{maxim}_improvement_pct"]
        f.write(f"{maxim},{baseline:.4f},{dpo:.4f},{improvement:.4f},{improvement_pct:.2f}\n")

print("\n" + "="*70)
print("RESULTS SAVED!")
print("="*70)
print("✓ dpo_evaluation_results.json")
print("✓ violation_rates.csv")
print("\nDownload these files from the Output tab (right sidebar)")
print("="*70)
