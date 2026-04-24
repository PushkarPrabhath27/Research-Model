---
language:
  - en
license: apache-2.0
library_name: peft
tags:
  - text-generation
  - dialogue
  - gricean-maxims
  - cooperative-communication
  - lora
  - dpo
  - direct-preference-optimization
  - peft
  - gpt2
  - nlp
datasets:
  - topical_chat
metrics:
  - cooperative_rate
pipeline_tag: text-generation
base_model: openai-community/gpt2-medium
---

<div align="center">

# ⚡ GriceBench-DPO

**GPT-2-medium fine-tuned with Direct Preference Optimization to generate cooperative dialogue.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEFT LoRA](https://img.shields.io/badge/🤗-PEFT%20LoRA-yellow)](https://huggingface.co/docs/peft)
[![HuggingFace](https://img.shields.io/badge/🤗-GriceBench-yellow)](https://huggingface.co/Pushkar27)

**Part of the GriceBench system** —
[GitHub](https://github.com/PushkarPrabhath27/Research-Model) |
[🔍 Detector](https://huggingface.co/Pushkar27/GriceBench-Detector) |
[🛠 Repair Model](https://huggingface.co/Pushkar27/GriceBench-Repair)

</div>

---

## What This Model Does

GriceBench-DPO is a LoRA-adapted GPT-2-medium model trained with Direct Preference
Optimization (DPO) to generate dialogue responses that comply with Gricean
conversational maxims. It is the **generation stage** of the GriceBench pipeline,
producing responses that are more likely to be cooperative *before* any
post-generation detection and repair is applied.

| Metric | Score | Context |
|--------|-------|---------|
| Standalone cooperative rate | 83.2% | Using this model alone |
| Full pipeline cooperative rate | **95.0%** | DPO + Detector + Repair |
| DPO preference accuracy | 75.0% | Held-out preference pairs (Phase 7) |
| DPO eval loss | 0.5595 | End of training |

> **Important:** The 95.0% figure requires the full pipeline. This model alone
> achieves 83.2% — still competitive with the un-tuned baseline (83.8%), with
> Relation violations dramatically reduced (~62% → ~10%).

---

## Quick Start

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load LoRA adapter on GPT-2-medium base
adapter_path = "Pushkar27/GriceBench-DPO"
config = PeftConfig.from_pretrained(adapter_path)
print(f"Base model: {config.base_model_name_or_path}")
# → openai-community/gpt2-medium

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float32,
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def generate_cooperative_response(context: str, max_new_tokens: int = 80) -> str:
    prompt = f"Context: {context}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# Example
context = "What do you think about the history of jazz music in New Orleans?"
print(generate_cooperative_response(context))
```

---

## Full Pipeline Usage (Recommended for Best Results)

```python
# For 95.0% cooperative rate, use all three GriceBench models together:
# Step 1: Generate with this DPO model
response = generate_cooperative_response(context)

# Step 2: Detect any remaining violations
# (see GriceBench-Detector model card for detection code)
result = detect_violations(context, response, evidence)

# Step 3: Repair each flagged violation
for maxim, violated in result["violations"].items():
    if violated and maxim != "relation":
        response = repair_violation(context, response, maxim)

# Final response achieves 95.0% cooperative rate across the test set
print(response)
```

Full pipeline implementation: [GitHub repository](https://github.com/PushkarPrabhath27/Research-Model)

---

## Ablation Results (Why You Need the Full Pipeline)

| Configuration | Cooperative Rate | Notes |
|---------------|-----------------|-------|
| Baseline (GPT-2, no tuning) | 83.8% | Reference |
| **This model (DPO only)** | **83.2%** | Relation violations −52pp; Manner unchanged |
| Detect + Repair (no DPO) | 93.0% | Repair handles Manner |
| **Full System** | **95.0%** | DPO + Detect + Repair combined |

**Why DPO alone barely moves the overall number:** DPO dramatically reduces
Relation violations (62% → ~10%) but cannot address Manner violations (still
~64%), which are the dominant failure mode. The repair model handles Manner.
Together: 95.0%.

---

## Training Details

### Model Architecture

| Parameter | Value |
|-----------|-------|
| Base model | `openai-community/gpt2-medium` (355M) |
| Method | LoRA (Low-Rank Adaptation) |
| LoRA rank (r) | 128 |
| LoRA alpha (α) | 256 |
| Target modules | q, k, v, o attention projections |
| Adapter size | ~25 MB |

### DPO Training

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | Direct Preference Optimization (DPO) |
| DPO β | 0.1 |
| Learning rate | 5e-7 |
| Batch size | 16 (grad accum ×8) |
| Epochs | 3 |
| Training pairs | 1,970 filtered preference pairs |
| Hardware | Kaggle P100-16GB, ~24 minutes |

**DPO loss formulation:**

The model is trained to maximize the margin between preferred (cooperative) and
rejected (violating) responses relative to the reference model:

The DPO loss maximizes the margin between chosen (y_w) and rejected (y_l) 
responses relative to a reference model:

  L_DPO = -log sigmoid( beta * [ log(pi(y_w|x)/pi_ref(y_w|x)) 
                                  - log(pi(y_l|x)/pi_ref(y_l|x)) ] )

where beta = 0.1 controls preference strength.

### Training Data

| Source | Pairs | Description |
|--------|-------|-------------|
| Human-labeled | 411 | Expert-verified cooperative/violating pairs |
| Repair-derived | ~1,200 | (original violation, T5-repaired output) |
| Synthetic (LLM) | ~1,200 | Generated via Groq API (llama-3.3-70b) |
| **Total (filtered)** | **1,970** | After conflict-detection filtering |

---

## Files

| File | Description |
|------|-------------|
| `adapter_config.json` | LoRA configuration (base model, rank, alpha) |
| `adapter_model.safetensors` | LoRA weights (~25 MB) |
| `tokenizer.json` | GPT-2 tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |
| `special_tokens_map.json` | Special token mappings |

---

## Limitations

- **Manner violations persist standalone:** DPO reduces Relation violations
  but not Manner. The full pipeline is required for the headline 95.0% result.
- **Single domain:** Trained and evaluated on Topical-Chat only.
- **English only:** No multilingual support.
- **Preference accuracy (75.0%) vs. Phase 5 training accuracy (98.7%):**
  The 75.0% figure is from held-out Phase 7 evaluation (canonical).
  The 98.7% was from in-distribution Phase 5 evaluation and is not the
  representative number.

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
| GriceBench-Detector | Detects violations | [🔍 Detector](https://huggingface.co/Pushkar27/GriceBench-Detector) |
| GriceBench-Repair | Repairs violations | [🛠 Repair](https://huggingface.co/Pushkar27/GriceBench-Repair) |
| GriceBench-DPO | Generates cooperative responses (this model) | You are here |

**GitHub:** https://github.com/PushkarPrabhath27/Research-Model
