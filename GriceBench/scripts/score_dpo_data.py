"""
Score DPO Training Data with Detector V2

This script loads the trained Detector V2 model and scores all DPO
training and validation pairs (both chosen and rejected responses).
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
from tqdm import tqdm

# ============================================
# Model Architecture (same as training)
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
# Load Model and Tokenizer
# ============================================

print("Loading Detector V2...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_name = 'microsoft/deberta-v3-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = MaximDetectorV2(model_name).to(device)

# Load trained weights
checkpoint = torch.load('best_model_v2.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded")

# Load temperature scaling
with open('temperatures.json') as f:
    temperatures = json.load(f)

print(f"✓ Temperatures loaded: {temperatures}")

# ============================================
# Scoring Function
# ============================================

def score_response(context, response, evidence=None):
    """Score a response for maxim violations"""
    
    # Construct input text
    if evidence:
        text = f"Context: {context} Evidence: {evidence} Response: {response}"
    else:
        text = f"Context: {context} Response: {response}"
    
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get logits
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    # Apply temperature scaling and sigmoid
    maxims = ['quantity', 'quality', 'relation', 'manner']
    scores = {}
    
    for i, maxim in enumerate(maxims):
        temp = temperatures[maxim]
        scaled_logit = logits[0, i] / temp
        prob = torch.sigmoid(scaled_logit).item()
        scores[maxim] = prob
    
    return scores

# ============================================
# Score DPO Training Data
# ============================================

print("\nScoring DPO training data...")

# Load DPO training data
dpo_train_path = Path('data_processed/dpo_data/dpo_train.json')
with open(dpo_train_path) as f:
    dpo_train = json.load(f)

print(f"Loaded {len(dpo_train)} training pairs")

# Score each pair
scored_data = []

for item in tqdm(dpo_train, desc="Scoring training pairs"):
    # Extract fields
    prompt = item.get('prompt', item.get('context', ''))
    chosen = item.get('chosen', item.get('chosen_response', ''))
    rejected = item.get('rejected', item.get('rejected_response', ''))
    
    # Score chosen response
    chosen_scores = score_response(prompt, chosen)
    
    # Score rejected response
    rejected_scores = score_response(prompt, rejected)
    
    # Add scores to item
    scored_item = item.copy()
    scored_item['chosen_scores'] = chosen_scores
    scored_item['rejected_scores'] = rejected_scores
    
    # Calculate margins
    margins = {
        maxim: rejected_scores[maxim] - chosen_scores[maxim]
        for maxim in ['quantity', 'quality', 'relation', 'manner']
    }
    scored_item['margins'] = margins
    scored_item['avg_margin'] = sum(margins.values()) / len(margins)
    
    scored_data.append(scored_item)

# Save scored data
output_path = Path('data_processed/dpo_data/dpo_train_scored.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(scored_data, f, indent=2)

print(f"\n✓ Saved scored data to {output_path}")

# ============================================
# Score DPO Validation Data
# ============================================

print("\nScoring DPO validation data...")

dpo_val_path = Path('data_processed/dpo_data/dpo_val.json')
if dpo_val_path.exists():
    with open(dpo_val_path) as f:
        dpo_val = json.load(f)
    
    print(f"Loaded {len(dpo_val)} validation pairs")
    
    scored_val = []
    for item in tqdm(dpo_val, desc="Scoring validation pairs"):
        prompt = item.get('prompt', item.get('context', ''))
        chosen = item.get('chosen', item.get('chosen_response', ''))
        rejected = item.get('rejected', item.get('rejected_response', ''))
        
        chosen_scores = score_response(prompt, chosen)
        rejected_scores = score_response(prompt, rejected)
        
        scored_item = item.copy()
        scored_item['chosen_scores'] = chosen_scores
        scored_item['rejected_scores'] = rejected_scores
        
        margins = {
            maxim: rejected_scores[maxim] - chosen_scores[maxim]
            for maxim in ['quantity', 'quality', 'relation', 'manner']
        }
        scored_item['margins'] = margins
        scored_item['avg_margin'] = sum(margins.values()) / len(margins)
        
        scored_val.append(scored_item)
    
    val_output_path = Path('data_processed/dpo_data/dpo_val_scored.json')
    with open(val_output_path, 'w') as f:
        json.dump(scored_val, f, indent=2)
    
    print(f"✓ Saved scored validation data to {val_output_path}")

# ============================================
# Statistics
# ============================================

print("\n" + "="*60)
print("SCORING STATISTICS")
print("="*60)

import numpy as np

margins_by_maxim = {m: [] for m in ['quantity', 'quality', 'relation', 'manner']}
avg_margins = []

for item in scored_data:
    for maxim, margin in item['margins'].items():
        margins_by_maxim[maxim].append(margin)
    avg_margins.append(item['avg_margin'])

print("\nMargin Statistics (rejected - chosen):")
print("Positive margin = chosen is better\n")

for maxim in ['quantity', 'quality', 'relation', 'manner']:
    margins = np.array(margins_by_maxim[maxim])
    print(f"{maxim.upper()}:")
    print(f"  Mean:   {margins.mean():.3f}")
    print(f"  Std:    {margins.std():.3f}")
    print(f"  Min:    {margins.min():.3f}")
    print(f"  Max:    {margins.max():.3f}")
    print(f"  >0.15:  {(margins > 0.15).mean()*100:.1f}%")
    print(f"  >0.20:  {(margins > 0.20).mean()*100:.1f}%")
    print()

avg_margins = np.array(avg_margins)
print("AVERAGE MARGIN:")
print(f"  Mean:   {avg_margins.mean():.3f}")
print(f"  Std:    {avg_margins.std():.3f}")
print(f"  >0.15:  {(avg_margins > 0.15).mean()*100:.1f}%")
print(f"  >0.20:  {(avg_margins > 0.20).mean()*100:.1f}%")

print("\n" + "="*60)
print("SCORING COMPLETE!")
print("="*60)
print("\nNext step: Run filter_dpo_pairs.py to filter by margin quality")
