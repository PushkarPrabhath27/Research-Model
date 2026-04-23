<div align="center">

# 🗣️ GriceBench

### Operationalizing Gricean Maxims for Cooperative AI Dialogue

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/Pushkar27)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-blue?logo=kaggle)](https://www.kaggle.com/)

**95.0% cooperative rate · Beats Mistral-7B and Qwen2.5-7B · 360M parameters**

[📄 Paper (coming soon)](#citation) · [🤗 Models](https://huggingface.co/Pushkar27) · [🚀 Quick Start](#quick-start)

</div>

---

## What Is GriceBench?

Modern AI dialogue systems produce fluent responses that routinely fail at
**cooperative communication** — the foundational requirement of effective dialogue.
GriceBench is a framework that automatically detects and repairs these failures
using Paul Grice's four conversational maxims as a principled decomposition.

**The problem:** A response can be grammatically perfect and factually accurate,
yet still be deeply uncooperative.

| Failure Type | Example | Gricean Maxim Violated |
|-------------|---------|----------------------|
| Over-informative | "What time is it?" → 500-word history of timekeeping | Quantity |
| Factually wrong | Contradicts available knowledge evidence | Quality |
| Off-topic | Responds about classical music when asked about jazz | Relation |
| Unclear | "She said she would do it before she left" (who is she?) | Manner |

**The solution:** A three-component pipeline that enforces cooperative communication.

---

## Results

| System | Cooperative Rate | Notes |
|--------|-----------------|-------|
| GPT-2-medium (baseline) | 83.8% | No tuning |
| Qwen2.5-7B-Instruct | 84.2% | 7B parameter model |
| Mistral-7B-Instruct | 89.1% | 7B parameter model |
| GriceBench (DPO only) | 83.2% | 360M params, preference-tuned |
| GriceBench (Detect+Repair) | 93.0% | No DPO |
| **GriceBench (Full System)** | **95.0%** | **+11.2pp over baseline** |

Bootstrap 95% CI for Full System vs. Baseline: statistically significant (McNemar's test, p < 0.01).

**Detector performance (macro F1: 0.955):**

| Maxim | F1 | AUC |
|-------|-----|-----|
+| Quantity | 1.000 | 1.000 |
+| Quality | 0.928 | 0.999 |
+| Relation | 1.000 | 1.000 |
+| Manner | 0.891 | 0.979 |

---

## Architecture

```
Conversation Context
       │
       ▼
┌─────────────────────────┐
│     DPO GENERATOR       │  GPT-2-medium + LoRA (r=128)
│  Preference-aligned     │  Generates cooperative responses from the start
│  generation             │  Trained on 1,970 DPO preference pairs
└──────────┬──────────────┘
           │ generated response
           ▼
┌─────────────────────────┐
│   VIOLATION DETECTOR    │  DeBERTa-v3-base · Macro F1: 0.955
│   4 independent heads   │  One binary head per Gricean maxim
│   Temperature-scaled    │  Focal Loss + temperature calibration
└──────────┬──────────────┘
           │ per-maxim violation flags
     ┌─────▼──────┐
     │ Violations?│
     └──┬──────┬──┘
       No     Yes
        │      ├─ Relation  → FAISS retrieval (50K corpus, MRR > 0.70)
        │      ├─ Quantity  → T5-base repair (beam search, BLEU: 61.8%)
        │      ├─ Quality   → T5-base repair (beam search, BLEU: 97.8%)
        │      └─ Manner    → T5-base repair (nucleus sampling, BLEU: 92.5%)
        │               │
        └───────────────┘
               │
               ▼
    Final Cooperative Response
         (95.0% cooperative rate)
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/PushkarPrabhath27/Research-Model.git
cd Research-Model/GriceBench
pip install -r requirements.txt
```

### Run the Full Pipeline

```python
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel, PeftConfig
import torch, json

# ── Load all three models ──────────────────────────────────────────────────
# (see individual model cards for detailed loading instructions)

# Detector
from scripts.detector_inference import GriceBenchDetector
detector = GriceBenchDetector("best_model_v2.pt", "temperatures.json")

# Repair
from scripts.repair_inference_fixed import FixedRepairModel
repair = FixedRepairModel("models/repair/repair_model/")

# DPO Generator
config = PeftConfig.from_pretrained("Pushkar27/GriceBench-DPO")
base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
generator = PeftModel.from_pretrained(base, "Pushkar27/GriceBench-DPO")

# ── Run pipeline ───────────────────────────────────────────────────────────
context = "What do you think about renewable energy?"
evidence = "Solar panel costs have dropped 90% since 2010."

# Step 1: Generate
response = generator.generate(context)

# Step 2: Detect violations
result = detector.detect(context, response, evidence)

# Step 3: Repair each violation
for maxim, violated in result["violations"].items():
    if violated and maxim != "relation":
        response = repair.repair(
            f"fix {maxim}: [CONTEXT] {context} [RESPONSE] {response}",
            violation_type=maxim
        )["repaired_text"]

print(f"Final cooperative response: {response}")
print(f"Is cooperative: {result['is_cooperative']}")
```

---

## Models

| Model | HuggingFace | Size | Key Metric |
|-------|-------------|------|-----------|
| 🔍 GriceBench-Detector | [Pushkar27/GriceBench-Detector](https://huggingface.co/Pushkar27/GriceBench-Detector) | 2.22 GB | Macro F1: 0.955 |
| 🔧 GriceBench-Repair | [Pushkar27/GriceBench-Repair](https://huggingface.co/Pushkar27/GriceBench-Repair) | 891 MB | Removal Rate: 93.0% |
| ⚡ GriceBench-DPO | [Pushkar27/GriceBench-DPO](https://huggingface.co/Pushkar27/GriceBench-DPO) | 25 MB adapter | Pref. Accuracy: 75.0% |

---

## Repository Structure

```
GriceBench/
├── 📊 data_raw/                    # Raw Topical-Chat data
├── 📊 data_processed/
│   ├── detector_data/              # detector_train.json, detector_val.json
│   ├── repair_data/                # repair_train.json, repair_val.json
│   ├── dpo_data/                   # dpo_train_filtered.json (1,970 pairs)
│   └── relation_repair/            # FAISS index + response corpus
├── 🤖 best_model_v2.pt             # Trained Detector weights (2.22 GB)
├── 🤖 temperatures.json            # Detector calibration temperatures
├── 🤖 models/repair/               # T5 repair model (891 MB)
├── 🤖 dpo_training_final_outcome/  # DPO LoRA adapter (25 MB)
├── 📈 results/
│   ├── phase7output/               # Final end-to-end evaluation
│   ├── part4output/                # Ablation study results
│   └── statistical_significance.json
├── 🔧 scripts/
│   ├── repair_inference_fixed.py   # Fixed repair inference (use this)
│   ├── statistical_significance.py
│   └── consistency_audit_final.py
├── 📓 KAGGLE_PHASE*.ipynb          # Kaggle training notebooks
├── 🐳 Dockerfile
└── 🐳 docker-compose.yml
```

---

## Reproduce Results

All experiments were run on Kaggle free-tier (T4/P100 GPUs).

**Phase 7 end-to-end evaluation (1,000 examples, ~1 hour on P100):**
```bash
# Upload models to Kaggle datasets first, then run:
python KAGGLE_CHAPTER14_EVALUATION.py
```

**Ablation study:**
```bash
python -c "
import json
with open('results/part4output/ablation_results.json') as f:
    data = json.load(f)
for sys, metrics in data['component_ablation'].items():
    print(f'{sys}: {metrics[\"cooperative_rate\"]*100:.1f}%')
"
```

**Statistical significance:**
```bash
python scripts/statistical_significance.py
```

---

## Theoretical Background

GriceBench operationalizes Grice's (1975) four conversational maxims:

- **Quantity:** Be as informative as required — no more, no less.
  → Operationalized: Response length between 8 and 38 words (10th–95th percentile of Topical-Chat)

- **Quality:** Be truthful — do not say what you believe to be false.
  → Operationalized: NLI-verified consistency with the knowledge evidence snippet

- **Relation:** Be relevant — make your contribution relevant to the conversation.
  → Operationalized: Cosine similarity between response and context embeddings

- **Manner:** Be clear — avoid obscurity, ambiguity, and disorder.
  → Operationalized: Readability scores, pronoun ambiguity heuristics, structural features

---

## Citation

```bibtex
@article{prabhath2026gricebench,
  title={GriceBench: Operationalizing Gricean Maxims for Cooperative Dialogue Evaluation and Generation},
  author={Prabhath, Pushkar},
  year={2026},
  note={Under review}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
