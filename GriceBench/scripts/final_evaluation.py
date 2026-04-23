"""
Final Evaluation: Detector V2 + DPO Optimized Model

This script loads both trained models and evaluates the complete system
on the test set to calculate final cooperative rates and maxim improvements.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# ============================================
# Detector V2 Model (Same as training)
# ============================================

class MaximDetectorV2(nn.Module):
    """Improved detector with deeper classification heads"""
    
    def __init__(self, model_name, num_maxims=4, dropout=0.15):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, 1)
            )
            for _ in range(num_maxims)
        ])
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = torch.cat([
            classifier(pooled)
            for classifier in self.classifiers
        ], dim=1)
        return logits

# ============================================
# Load Models
# ============================================

print("="*60)
print("LOADING MODELS")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load Detector V2
print("\n1. Loading Detector V2...")
detector_model_name = 'microsoft/deberta-v3-base'
detector_tokenizer = AutoTokenizer.from_pretrained(detector_model_name)

detector = MaximDetectorV2(detector_model_name).to(device)
checkpoint = torch.load('best_model_v2.pt', map_location=device, weights_only=False)
detector.load_state_dict(checkpoint['model_state_dict'])
detector.eval()

# Load temperatures
with open('temperatures.json') as f:
    temperatures = json.load(f)

print("✓ Detector V2 loaded")

# Load DPO Model
print("\n2. Loading DPO Optimized Model...")
base_model_name = 'gpt2-medium'
dpo_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
dpo_tokenizer.pad_token = dpo_tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32
).to(device)

# Load LoRA adapters
dpo_model = PeftModel.from_pretrained(
    base_model,
    'dpo_training_final_outcome'
).to(device)
dpo_model.eval()

print("✓ DPO Model loaded")

# Load baseline model for comparison
print("\n3. Loading Baseline Model...")
baseline_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32
).to(device)
baseline_model.eval()

print("✓ Baseline Model loaded")

# ============================================
# Load Test Data
# ============================================

print("\n" + "="*60)
print("LOADING TEST DATA")
print("="*60)

# Load test set
test_path = Path('data_processed/test_data.json')
if not test_path.exists():
    # Use validation data as test
    test_path = Path('data_processed/detector_data/detector_val.json')

with open(test_path) as f:
    test_data = json.load(f)

print(f"✓ Loaded {len(test_data)} test examples")

# ============================================
# Evaluation Functions
# ============================================

def detect_violations(detector, tokenizer, context, response, temperatures, device):
    """Detect maxim violations in a response"""
    
    text = f"Context: {context} Response: {response}"
    
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = detector(input_ids, attention_mask)
    
    maxims = ['quantity', 'quality', 'relation', 'manner']
    scores = {}
    
    for i, maxim in enumerate(maxims):
        temp = temperatures[maxim]
        scaled_logit = logits[0, i] / temp
        prob = torch.sigmoid(scaled_logit).item()
        scores[maxim] = prob
    
    return scores

def generate_response(model, tokenizer, prompt, max_length=100, device='cpu'):
    """Generate a response from the model"""
    
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=inputs['input_ids'].shape[1] + max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

# ============================================
# Run Evaluation
# ============================================

print("\n" + "="*60)
print("RUNNING EVALUATION")
print("="*60)

results = {
    'baseline': defaultdict(list),
    'dpo': defaultdict(list)
}

threshold = 0.5  # Standard threshold

print("\nGenerating and evaluating responses...")

for item in tqdm(test_data[:100], desc="Evaluating"):  # Evaluate on first 100 examples
    
    context = item.get('context', item.get('prompt', ''))
    
    # Generate baseline response
    baseline_response = generate_response(baseline_model, dpo_tokenizer, context, device=device)
    
    # Generate DPO response
    dpo_response = generate_response(dpo_model, dpo_tokenizer, context, device=device)
    
    # Detect violations in baseline
    baseline_scores = detect_violations(detector, detector_tokenizer, context, baseline_response, temperatures, device)
    
    # Detect violations in DPO
    dpo_scores = detect_violations(detector, detector_tokenizer, context, dpo_response, temperatures, device)
    
    # Record violations (score > threshold = violation)
    for maxim in ['quantity', 'quality', 'relation', 'manner']:
        results['baseline'][maxim].append(1 if baseline_scores[maxim] > threshold else 0)
        results['dpo'][maxim].append(1 if dpo_scores[maxim] > threshold else 0)

# ============================================
# Calculate Metrics
# ============================================

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

maxims = ['quantity', 'quality', 'relation', 'manner']

print("\nViolation Rates:\n")
print(f"{'Maxim':<12} {'Baseline':>10} {'DPO':>10} {'Improvement':>15}")
print("-" * 60)

improvements = {}

for maxim in maxims:
    baseline_rate = np.mean(results['baseline'][maxim]) * 100
    dpo_rate = np.mean(results['dpo'][maxim]) * 100
    improvement = ((baseline_rate - dpo_rate) / baseline_rate * 100) if baseline_rate > 0 else 0
    
    improvements[maxim] = improvement
    
    print(f"{maxim.capitalize():<12} {baseline_rate:>9.1f}% {dpo_rate:>9.1f}% {improvement:>+14.1f}%")

# Calculate cooperative rate
baseline_cooperative = []
dpo_cooperative = []

for i in range(len(results['baseline']['quantity'])):
    baseline_violations = sum(results['baseline'][m][i] for m in maxims)
    dpo_violations = sum(results['dpo'][m][i] for m in maxims)
    
    baseline_cooperative.append(1 if baseline_violations == 0 else 0)
    dpo_cooperative.append(1 if dpo_violations == 0 else 0)

baseline_coop_rate = np.mean(baseline_cooperative) * 100
dpo_coop_rate = np.mean(dpo_cooperative) * 100
coop_improvement = dpo_coop_rate - baseline_coop_rate

print("-" * 60)
print(f"{'Cooperative':<12} {baseline_coop_rate:>9.1f}% {dpo_coop_rate:>9.1f}% {coop_improvement:>+14.1f} pp")

# ============================================
# Save Results
# ============================================

final_results = {
    'violation_rates': {
        'baseline': {m: float(np.mean(results['baseline'][m]) * 100) for m in maxims},
        'dpo': {m: float(np.mean(results['dpo'][m]) * 100) for m in maxims}
    },
    'improvements': {m: float(improvements[m]) for m in maxims},
    'cooperative_rate': {
        'baseline': float(baseline_coop_rate),
        'dpo': float(dpo_coop_rate),
        'improvement': float(coop_improvement)
    }
}

with open('final_evaluation_results.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n✓ Saved results to final_evaluation_results.json")

print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
