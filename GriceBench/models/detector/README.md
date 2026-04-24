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
          name: Topical-Chat (GriceBench held-out split, N=1000)
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

# 🔍 GriceBench-Detector

**Detects cooperative communication failures in AI dialogue — one Gricean maxim at a time.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace](https://img.shields.io/badge/🤗-GriceBench-yellow)](https://huggingface.co/Pushkar27)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Part of the GriceBench system** — 
[GitHub](https://github.com/PushkarPrabhath27/Research-Model) | 
[🛠 Repair Model](https://huggingface.co/Pushkar27/GriceBench-Repair) | 
[⚡ DPO Generator](https://huggingface.co/Pushkar27/GriceBench-DPO)

</div>

---

## What This Model Does

GriceBench-Detector identifies which of Paul Grice's four conversational maxims
a dialogue response violates. It returns four independent calibrated violation
probabilities — one per maxim — enabling targeted, explainable repair downstream.

| Output | Maxim | Violation Detected | Example |
|--------|-------|-------------------|---------|
| `quantity_prob` | Quantity | Response too short (<8 words) or too long (>38 words) | "Yes." to a detailed question |
| `quality_prob` | Quality | Factually inconsistent with knowledge evidence | Wrong date, incorrect name |
| `relation_prob` | Relation | Off-topic response | Jazz question answered with classical music facts |
| `manner_prob` | Manner | Ambiguous, jargon-heavy, or disorganized | Unclear pronoun references |

Used in the full GriceBench pipeline, this detector helps achieve a **95.0% cooperative rate**
— outperforming Mistral-7B-Instruct (89.1%) and Qwen2.5-7B-Instruct (84.2%).

---

## Quick Start

```python
import torch
import torch.nn as nn
import json
from transformers import AutoTokenizer, AutoModel

# ─── Define model architecture (must match training) ─────────────────────────
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

# ─── Load model and calibration ──────────────────────────────────────────────
# Download pytorch_model.pt and temperatures.json from this repo first
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = MaximDetector()
state_dict = torch.load("pytorch_model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

with open("temperatures.json") as f:
    temperatures = json.load(f)

# ─── Detect violations ───────────────────────────────────────────────────────
def detect_violations(context: str, response: str, evidence: str = "") -> dict:
    input_text = f"Context: {context}\nEvidence: {evidence}\nResponse: {response}"
    inputs = tokenizer(
        input_text, return_tensors="pt",
        max_length=512, truncation=True, padding=True
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

    probs, violations = {}, {}
    for i, (maxim, temp) in enumerate(zip(maxim_names, temp_values)):
        prob = torch.sigmoid(logits[0, i] / temp).item()
        probs[maxim] = round(prob, 4)
        violations[maxim] = prob > 0.5

    return {
        "violations": violations,
        "probabilities": probs,
        "is_cooperative": not any(violations.values())
    }

# ─── Example ─────────────────────────────────────────────────────────────────
result = detect_violations(
    context="What do you think about the latest developments in AI?",
    response="Yes.",   # Too short — Quantity violation
    evidence="AI has seen rapid advancement in large language models during 2024-2025."
)
print(result)
# {'violations': {'quantity': True, 'quality': False, 'relation': False, 'manner': False},
#  'probabilities': {'quantity': 0.97, 'quality': 0.02, 'relation': 0.03, 'manner': 0.11},
#  'is_cooperative': False}
```

---

## Performance

Evaluated on **1,000 held-out Topical-Chat dialogue turns** (500 violation-injected, 500 clean).

| Maxim | F1 | Precision | Recall | AUC-ROC |
|-------|-----|-----------|--------|---------|
| Quantity | **1.000** | 1.000 | 1.000 | 1.000 |
| Quality | 0.928 | 0.866 | 1.000 | 0.999 |
| Relation | **1.000** | 1.000 | 1.000 | 1.000 |
| Manner | 0.891 | 0.864 | 0.919 | 0.979 |
| **Macro Avg** | **0.955** | — | — | — |

---

## Architecture & Training

- **Base model:** `microsoft/deberta-v3-base` (184M parameters)
- **Heads:** 4 independent binary classification heads (one per maxim)
- **Loss:** Focal Loss (α=0.25, γ=2.0) for class imbalance
- **Calibration:** Per-head temperature scaling (see `temperatures.json`)
- **Training data:** 4,012 examples (weak supervision + ~1,000 gold labels)
- **Epochs:** 5 | **LR:** 2e-5 | **Hardware:** Kaggle T4 ×2, ~2–3 hours

**Calibrated temperatures:**

| Maxim | Temperature | Effect |
|-------|-------------|--------|
| Quantity | 0.90 | Slightly sharper |
| Quality | 0.55 | Conservative (fewer false positives) |
| Relation | 0.75 | Balanced |
| Manner | 0.45 | Most conservative (subjective maxim) |

---

## Files

| File | Description |
|------|-------------|
| `pytorch_model.pt` | Trained model weights (2.22 GB) |
| `temperatures.json` | Per-maxim calibration temperatures |

---

## Citation

```bibtex
 @article{prabhath2026gricebench,
  title={GriceBench: Operationalizing Gricean Maxims for Cooperative Dialogue Evaluation and Generation},
  author={Prabhath, Pushkar},
  year={2026},
  note={Under review, EMNLP 2026}
}
```

---

## Related Models

| Model | Role | Link |
|-------|------|------|
| GriceBench-Detector | Detects violations (this model) | You are here |
| GriceBench-Repair | Repairs detected violations | [🛠 Repair](https://huggingface.co/Pushkar27/GriceBench-Repair) |
| GriceBench-DPO | Generates cooperative responses | [⚡ DPO](https://huggingface.co/Pushkar27/GriceBench-DPO) |

**GitHub:** https://github.com/PushkarPrabhath27/Research-Model
