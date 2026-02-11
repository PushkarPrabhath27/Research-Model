"""
Script to generate Part 5: Error Analysis Jupyter Notebook
Avoids PowerShell escaping issues by using a standalone Python script
"""

import json
from pathlib import Path

# Define notebook structure
notebook = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

# Cell 1: Title
notebook['cells'].append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '# GriceBench Part 5: Error Analysis\n',
        '\n',
        '**Objective:** Comprehensive error analysis of the Gricean Maxim Detector\n',
        '\n',
        '## Analysis Components:\n',
        '1. Confusion matrices per maxim\n',
        '2. Hardest examples (confident errors)\n',
        '3. Failure mode categorization\n',
        '4. Qualitative examples for paper\n',
        '\n',
        '**Runtime:** ~10-15 minutes on Kaggle GPU'
    ]
})

# Cell 2: Imports
notebook['cells'].append({
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# Cell 1: Imports and Setup\n',
        'import os\n',
        'import json\n',
        'import torch\n',
        'import torch.nn as nn\n',
        'import numpy as np\n',
        'import pandas as pd\n',
        'import matplotlib.pyplot as plt\n',
        'import seaborn as sns\n',
        'from pathlib import Path\n',
        'from datetime import datetime\n',
        'from sklearn.metrics import confusion_matrix, classification_report, f1_score\n',
        'from transformers import AutoTokenizer, AutoModel\n',
        'import gc\n',
        'import sys\n',
        '\n',
        '# Seaborn style\n',
        'sns.set_style("whitegrid")\n',
        'sns.set_palette("husl")\n',
        'plt.rcParams["figure.figsize"] = (12, 8)\n',
        'plt.rcParams["font.size"] = 11\n',
        '\n',
        '# Device\n',
        'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n',
        'print(f"Device: {device}")\n',
        'if torch.cuda.is_available():\n',
        '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n',
        '    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")\n',
        '\n',
        '# Logging\n',
        'def log(msg):\n',
        '    timestamp = datetime.now().strftime("%H:%M:%S")\n',
        '    print(f"[{timestamp}] {msg}")\n',
        '    sys.stdout.flush()\n',
        '\n',
        'log("✅ Setup complete!")'
    ]
})

# Add remaining cells with proper string formatting
cells_code = [
    # Cell 3: Config
    '''# Cell 2: Configuration
CONFIG = {
    "detector_path": "/kaggle/input/gricean-maxim-detector-model",
    "val_data_path": "/kaggle/input/gricebench-test-data/val_examples.json",
    "output_dir": "/kaggle/working/error_analysis",
    "threshold": 0.5,
    "top_k_errors": 10,
    "batch_size": 16,
    "max_length": 512,
}

MAXIMS = ["quantity", "quality", "relation", "manner"]

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(f"{CONFIG['output_dir']}/confusion_matrices", exist_ok=True)

log(f"✅ Config loaded")''',
    
    # Cell 4: Model architecture
    '''# Cell 3: Define Detector Architecture
class ViolationDetector(nn.Module):
    """Gricean Maxim Violation Detector"""
    
    def __init__(self, model_name="microsoft/deberta-v3-base", num_labels=4, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        probs = self.sigmoid(logits)
        return {'logits': logits, 'probs': probs}

log("✅ Detector architecture defined")''',
    
    # Cell 5: Load model
    '''# Cell 4: Load Detector Model
log("="*60)
log("LOADING DETECTOR MODEL")
log("="*60)

# Tokenizer
log("\\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
log(f"  ✅ Tokenizer loaded (vocab: {len(tokenizer)})")

# Model
log("\\nCreating model...")
detector = ViolationDetector("microsoft/deberta-v3-base")
total_params = sum(p.numel() for p in detector.parameters())
log(f"  ✅ Model created ({total_params:,} params)")

# Load weights
log("\\nLoading weights...")
checkpoint_path = f"{CONFIG['detector_path']}/best_model.pt"

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    if 'metrics' in checkpoint:
        macro_f1 = checkpoint['metrics'].get('macro_f1', 'N/A')
        log(f"  Training F1: {macro_f1}")
else:
    state_dict = checkpoint

detector.load_state_dict(state_dict, strict=True)
log("  ✅ Weights loaded")

detector = detector.to(device)
detector.eval()
log(f"  ✅ Model on {device}")

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()''',
    
    # Cell 6: Load data
    '''# Cell 5: Load Validation Data
log("\\n" + "="*60)
log("LOADING VALIDATION DATA")
log("="*60)

val_path = CONFIG["val_data_path"]

# Try alternative paths if needed
if not os.path.exists(val_path):
    alternatives = [
        "/kaggle/input/gricebench-test-data/test_examples.json",
    ]
    for alt in alternatives:
        if os.path.exists(alt):
            val_path = alt
            break

log(f"Loading: {val_path}")

with open(val_path, 'r', encoding='utf-8') as f:
    val_examples = json.load(f)

log(f"  ✅ Loaded {len(val_examples)} examples")

# Distribution
log(f"\\nDistribution:")
for maxim in MAXIMS:
    count = sum(1 for ex in val_examples if ex.get('labels', {}).get(maxim, 0) == 1)
    pct = count / len(val_examples) * 100
    log(f"  {maxim}: {count} ({pct:.1f}%)")''',
    
    # Cell 7: Inference
    '''# Cell 6: Run Inference
log("\\n" + "="*60)
log("RUNNING INFERENCE")
log("="*60)

@torch.no_grad()
def batch_predict(examples, batch_size=16):
    all_probs = []
    all_preds = []
    
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i+batch_size]
        texts = [ex['input_text'] for ex in batch]
        
        inputs = tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=CONFIG['max_length'],
            padding=True
        ).to(device)
        
        outputs = detector(inputs['input_ids'], inputs['attention_mask'])
        probs = outputs['probs'].cpu().numpy()
        preds = (probs > CONFIG['threshold']).astype(int)
        
        all_probs.append(probs)
        all_preds.append(preds)
        
        if (i // batch_size + 1) % 10 == 0:
            log(f"  {i + len(batch)}/{len(examples)}...")
    
    return np.vstack(all_probs), np.vstack(all_preds)

all_probs, all_preds = batch_predict(val_examples, CONFIG['batch_size'])
all_labels = np.array([[ex['labels'].get(m, 0) for m in MAXIMS] for ex in val_examples])

exact_match = (all_preds == all_labels).all(axis=1).mean()
log(f"\\n  ✅ Done! Exact match: {exact_match:.3f}")''',
    
    # Cell 8: Confusion matrices
    '''# Cell 7: Confusion Matrices
log("\\n" + "="*60)
log("GENERATING CONFUSION MATRICES")
log("="*60)

confusion_matrices = {}
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, maxim in enumerate(MAXIMS):
    y_true = all_labels[:, i]
    y_pred = all_preds[:, i]
    cm = confusion_matrix(y_true, y_pred)
    confusion_matrices[maxim] = cm
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Violation', 'Violation'],
        yticklabels=['No Violation', 'Violation'],
        ax=axes[i],
        cbar_kws={'label': 'Count'}
    )
    axes[i].set_title(f'{maxim.capitalize()} (F1={f1:.3f})', fontweight='bold')
    axes[i].set_ylabel('True')
    axes[i].set_xlabel('Predicted')
    
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    log(f"\\n{maxim}: TN={tn}, FP={fp}, FN={fn}, TP={tp}, F1={f1:.3f}")

plt.suptitle('GriceBench Detector - Confusion Matrices', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/confusion_matrices/all_confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.show()

log(f"\\n  ✅ Saved confusion matrices")''',
    
    # Cell 9: Hardest examples
    '''# Cell 8: Hardest Examples
log("\\n" + "="*60)
log("IDENTIFYING HARDEST EXAMPLES")
log("="*60)

hardest = {}

for i, maxim in enumerate(MAXIMS):
    y_true = all_labels[:, i]
    y_pred = all_preds[:, i]
    probs = all_probs[:, i]
    
    # False positives
    fp_mask = (y_pred == 1) & (y_true == 0)
    fp_idx = np.where(fp_mask)[0]
    top_fp = fp_idx[np.argsort(-probs[fp_idx])[:CONFIG['top_k_errors']]]
    
    # False negatives
    fn_mask = (y_pred == 0) & (y_true == 1)
    fn_idx = np.where(fn_mask)[0]
    top_fn = fn_idx[np.argsort(probs[fn_idx])[:CONFIG['top_k_errors']]]
    
    hardest[maxim] = {
        'false_positives': [
            {
                'index': int(idx),
                'confidence': float(probs[idx]),
                'text': val_examples[idx]['input_text'][:200],
                'type': val_examples[idx].get('violation_type', 'unknown')
            }
            for idx in top_fp
        ],
        'false_negatives': [
            {
                'index': int(idx),
                'confidence': float(probs[idx]),
                'text': val_examples[idx]['input_text'][:200],
                'type': val_examples[idx].get('violation_type', 'unknown')
            }
            for idx in top_fn
        ]
    }
    
    log(f"\\n{maxim}: FP={len(top_fp)}, FN={len(top_fn)}")

with open(f"{CONFIG['output_dir']}/hardest_examples.json", 'w') as f:
    json.dump(hardest, f, indent=2)

log(f"\\n  ✅ Saved hardest examples")''',
    
    # Cell 10: Report
    '''# Cell 9: Generate Report
log("\\n" + "="*60)
log("GENERATING REPORT")
log("="*60)

report = []
report.append(f"# Error Analysis Report\\n\\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\\n\\n")
report.append(f"**Exact Match Accuracy:** {exact_match:.3f}\\n\\n")

report.append("## Per-Maxim Performance\\n\\n")
report.append("| Maxim | F1 | Errors | TN | FP | FN | TP |\\n")
report.append("|-------|----|----|----|----|----|----|\\n")

for i, maxim in enumerate(MAXIMS):
    y_true = all_labels[:, i]
    y_pred = all_preds[:, i]
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrices[maxim]
    errors = (y_true != y_pred).sum()
    
    report.append(f"| {maxim.capitalize()} | {f1:.3f} | {errors} | ")
    report.append(f"{cm[0,0]} | {cm[0,1]} | {cm[1,0]} | {cm[1,1]} |\\n")

report.append("\\n![Confusion Matrices](confusion_matrices/all_confusion_matrices.png)\\n")

with open(f"{CONFIG['output_dir']}/error_report.md", 'w') as f:
    f.writelines(report)

log("\\n  ✅ Report saved")
log("\\n" + "="*60)
log("ERROR ANALYSIS COMPLETE")
log("="*60)'''
]

# Add code cells
for code in cells_code:
    notebook['cells'].append({
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': code.split('\n')
    })

# Write notebook
output_path = Path('kaggle_notebooks/GRICEBENCH_PART_5_ERROR_ANALYSIS.ipynb')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✅ Notebook created: {output_path}")
print(f"   Total cells: {len(notebook['cells'])}")
