---
language:
  - en
license: apache-2.0
library_name: transformers
tags:
  - text-classification
  - multi-label-classification
  - dialogue
  - conversational-ai
  - gricean-maxims
  - cooperative-communication
  - deberta
  - nlp
  - pragmatics
datasets:
  - topical_chat
metrics:
  - f1
  - precision
  - recall
  - roc_auc
pipeline_tag: text-classification
base_model: microsoft/deberta-v3-base
model-index:
  - name: GriceBench-Detector
    results:
      - task:
          type: text-classification
          name: Multi-Label Gricean Maxim Violation Detection
        dataset:
          name: Topical-Chat (GriceBench held-out split)
          type: custom
          split: test
        metrics:
          - type: f1
            value: 0.955
            name: Macro F1
          - type: f1
            value: 1.000
            name: Quantity F1
          - type: f1
            value: 0.928
            name: Quality F1
          - type: f1
            value: 1.000
            name: Relation F1
          - type: f1
            value: 0.891
            name: Manner F1
---

<div align="center">

# 🗣️ GriceBench-Detector

**Detects cooperative communication failures in AI dialogue — one maxim at a time.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/docs/transformers)

Part of the **GriceBench** system — [GitHub](https://github.com/PushkarPrabhath27/Research-Model) | 
[🔧 Repair Model](https://huggingface.co/Pushkar27/GriceBench-Repair) | 
[⚡ DPO Generator](https://huggingface.co/Pushkar27/GriceBench-DPO)

</div>

---

## What This Model Does

GriceBench-Detector identifies which of Paul Grice's four conversational maxims
a dialogue response violates. It returns four independent violation probabilities —
one per maxim — enabling targeted, explainable repair.

| Maxim | What It Measures | Example Violation |
|-------|-----------------|-------------------|
| **Quantity** | Response informativeness | "Yes." in response to a detailed question |
| **Quality** | Factual consistency with evidence | Stating an incorrect fact contradicted by the knowledge source |
| **Relation** | Topical relevance | Responding to "Tell me about jazz" with information about classical music |
| **Manner** | Clarity and organization | Pronoun ambiguity, jargon, disorganized sentences |

---

## Quick Start

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import json

# ── Load calibration temperatures ──────────────────────────────────────────
# Download temperatures.json from the model repo
with open("temperatures.json") as f:
    temperatures = json.load(f)  # {"quantity": 0.9, "quality": 0.55, ...}

# ── Define model architecture (must match training) ────────────────────────
class MaximDetector(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base", num_maxims=4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(0.15),
                nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden // 2, hidden // 4), nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden // 4, 1)
            ) for _ in range(num_maxims)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return torch.cat([head(cls) for head in self.classifiers], dim=1)

# ── Load model ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = MaximDetector()

# Load weights (download pytorch_model.pt from this repo)
state_dict = torch.load("pytorch_model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ── Run detection ──────────────────────────────────────────────────────────
def detect_violations(context: str, response: str, evidence: str = "") -> dict:
    input_text = f"Context: {context}\nEvidence: {evidence}\nResponse: {response}"
    inputs = tokenizer(
        input_text, return_tensors="pt", max_length=512,
        truncation=True, padding=True
    )
    
    maxim_names = ["quantity", "quality", "relation", "manner"]
    temp_values = [
        temperatures.get("quantity", 0.9),
        temperatures.get("quality", 0.55),
        temperatures.get("relation", 0.75),
        temperatures.get("manner", 0.45),
    ]
    
    with torch.no_grad():
        logits = model(**inputs)  # Shape: [1, 4]
    
    # Apply temperature scaling and sigmoid
    probs = {}
    violations = {}
    for i, (maxim, temp) in enumerate(zip(maxim_names, temp_values)):
        prob = torch.sigmoid(logits[0, i] / temp).item()
        probs[maxim] = round(prob, 4)
        violations[maxim] = prob > 0.5
    
    return {
        "violations": violations,
        "probabilities": probs,
        "is_cooperative": not any(violations.values())
    }

# ── Example ────────────────────────────────────────────────────────────────
result = detect_violations(
    context="What do you think about the latest developments in AI?",
    response="Yes.",  # Too short — Quantity violation
    evidence="AI has seen rapid advancement in large language models during 2024-2025."
)
print(result)
# {'violations': {'quantity': True, 'quality': False, 'relation': False, 'manner': False},
#  'probabilities': {'quantity': 0.97, 'quality': 0.02, 'relation': 0.03, 'manner': 0.11},
#  'is_cooperative': False}
```

---

## Model Performance

Evaluated on **1,000 held-out Topical-Chat dialogue turns** (500 violation-injected, 500 clean).

| Maxim | F1 | Precision | Recall | AUC-ROC |
|-------|-----|-----------|--------|---------|
+| Quantity | **1.000** | 1.000 | 1.000 | 1.000 |
+| Quality | 0.928 | 0.866 | 1.000 | 0.999 |
+| Relation | **1.000** | 1.000 | 1.000 | 1.000 |
+| Manner | 0.891 | 0.864 | 0.919 | 0.979 |
+| **Macro Avg** | **0.955** | — | — | — |

**System-level result:** When used in the full GriceBench pipeline (Detect → Repair → Generate),
the system achieves a **95.0% cooperative rate** — outperforming Mistral-7B (89.1%) and
Qwen2.5-7B (84.2%) despite using a far smaller generator.

---

## Architecture

**Base model:** `microsoft/deberta-v3-base` (184M parameters)

**Key design choices:**
- **Four independent binary heads** (not a shared linear layer): each maxim head specializes
  independently, since Quantity violations (length) and Relation violations (semantic relevance)
  are completely different feature distributions.
- **Focal Loss** (α=0.25, γ=2.0): down-weights easy negatives to focus training on hard,
  ambiguous boundary cases — critical for minority-class violation detection.
- **Temperature scaling**: post-hoc calibration (one scalar per maxim) ensures output
  probabilities match true violation frequencies on the validation set.

**Calibrated temperatures:**

| Maxim | Temperature | Effect |
|-------|-------------|--------|
| Quantity | 0.90 | Slightly sharper predictions |
| Quality | 0.55 | More conservative (fewer false positives) |
| Relation | 0.75 | Balanced |
| Manner | 0.45 | Most conservative (Manner is inherently ambiguous) |

---

## Training Details

| Hyperparameter | Value |
+|----------------|-------|
+| Base model | microsoft/deberta-v3-base |
+| Learning rate | 2e-5 |
+| Batch size | 16 (effective, with grad accumulation ×2) |
+| Epochs | 5 |
+| Loss | Focal Loss (α=0.25, γ=2.0) |
+| Optimizer | AdamW + weight decay 0.01 |
+| Scheduler | OneCycleLR |
+| Hardware | Kaggle T4 ×2 |
+| Training time | ~2-3 hours |
+| Training examples | 4,012 (weak supervision + ~1,000 gold labels) |

**Two-stage labeling:** Weak supervision (50,000+ heuristic-labeled examples) for pre-training,
followed by gold fine-tuning on ~1,000 human-annotated examples (inter-annotator agreement
measured via Krippendorff's α).

---

## Input Format

```
Context: [multi-turn conversation history]
Evidence: [knowledge snippet from reading set — required for Quality detection]
Response: [the response being evaluated]
```

Maximum token length: 512 (response is never truncated — context is truncated if needed).

---

## Files in This Repository

| File | Description |
|------|-------------|
| `pytorch_model.pt` | Trained model weights (2.22 GB) |
| `temperatures.json` | Per-maxim calibration temperatures |

---

## Citation

If you use this model, please cite:

```bibtex
@article{prabhath2026gricebench,
  title={GriceBench: Operationalizing Gricean Maxims for Cooperative Dialogue Evaluation and Generation},
  author={Prabhath, Pushkar},
  year={2026}
}
```

---

## Related Models

| Model | Role | Link |
|-------|------|------|
| GriceBench-Detector | Detects violations (this model) | You are here |
| GriceBench-Repair | Repairs detected violations | [🔧 Repair Model](https://huggingface.co/Pushkar27/GriceBench-Repair) |
| GriceBench-DPO | Generates cooperative responses | [⚡ DPO Generator](https://huggingface.co/Pushkar27/GriceBench-DPO) |

**GitHub:** https://github.com/PushkarPrabhath27/Research-Model
