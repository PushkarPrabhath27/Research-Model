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

**A GPT-2-medium model trained with Direct Preference Optimization to generate cooperative dialogue responses.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEFT](https://img.shields.io/badge/🤗-PEFT%20LoRA-yellow)](https://huggingface.co/docs/peft)

Part of the **GriceBench** system — [GitHub](https://github.com/PushkarPrabhath27/Research-Model) |
[🔍 Detector](https://huggingface.co/Pushkar27/GriceBench-Detector) |
[🔧 Repair Model](https://huggingface.co/Pushkar27/GriceBench-Repair)

</div>

---

## What This Model Does

GriceBench-DPO is a LoRA-adapted GPT-2-medium model fine-tuned with Direct Preference
Optimization (DPO) to generate dialogue responses that comply with Gricean conversational
maxims. It is the **first stage** of the GriceBench pipeline, producing responses that
are more likely to be cooperative before any post-generation repair is applied.

**Standalone cooperative rate: 83.2%** (vs. 83.8% un-tuned GPT-2 baseline)

When used as part of the full GriceBench pipeline (this model → Detector → Repair):
**Full system cooperative rate: 95.0%** — outperforming Mistral-7B (89.1%) and
Qwen2.5-7B (84.2%).

> **Why is standalone DPO only 83.2%?** DPO improves Relation violations dramatically
> (61% → 10%) but cannot address Manner violations, which require targeted repair.
> The 95% figure requires the full pipeline. See the Analysis section for details.

---

## Quick Start

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load LoRA adapter on top of GPT-2-medium
adapter_path = "Pushkar27/GriceBench-DPO"
config = PeftConfig.from_pretrained(adapter_path)

print(f"Base model: {config.base_model_name_or_path}")
# Base model: openai-community/gpt2-medium

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float32,
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def generate_cooperative_response(context: str, max_new_tokens: int = 80) -> str:
    """
    Generate a cooperative dialogue response.
    
    For best results, pass the output through the GriceBench-Detector
    and GriceBench-Repair models to catch any remaining violations.
    """
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
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Example ────────────────────────────────────────────────────────────────
context = "What do you think about the history of jazz music in New Orleans?"
response = generate_cooperative_response(context)
print(f"Generated: {response}")
```

---

## Full Pipeline Usage (Recommended)

For the best results (95.0% cooperative rate), use the full pipeline:

```python
# Full GriceBench pipeline: Generate → Detect → Repair
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch

# Step 1: Generate with DPO model
response = generate_cooperative_response(context)

# Step 2: Detect violations
# (see GriceBench-Detector model card for detection code)
violations = detect_violations(context, response, evidence)

# Step 3: Repair any violations found
for maxim, violated in violations["violations"].items():
    if violated and maxim != "relation":
        response = repair_violation(context, response, maxim)

# Result: cooperative response with 95.0% success rate
print(f"Final cooperative response: {response}")
```

See the [GitHub repository](https://github.com/PushkarPrabhath27/Research-Model) for the
complete pipeline implementation.

---

## Performance

### System-Level Results (Full Ablation Study, N=100 examples each)

| Configuration | Cooperative Rate | vs. Baseline |
|---------------|-----------------|--------------|
| Baseline (GPT-2-medium, no tuning) | 83.8% | — |
| **DPO Only** (this model, no repair) | **83.2%** | −0.6pp |
| Detect + Repair (no DPO) | 93.0% | +9.2pp |
| **Full System** (DPO + Detect + Repair) | **95.0%** | **+11.2pp** |

### Per-Maxim Violation Rates (DPO Only vs. Baseline)

| Maxim | Baseline Rate | DPO Rate | Change |
|-------|--------------|----------|--------|
+| Quantity | 3.0% | 3.0% | 0pp |
+| Quality | 0.0% | 0.0% | 0pp |
+| Relation | 62.0% | ~10.0% | **−52pp** ✅ |
+| Manner | 62.0% | 64.0% | +2pp ⚠️ |

DPO dramatically improves Relation violations but cannot address Manner violations.
This is why the full pipeline (adding Repair) is essential.

### DPO Training Metrics

| Metric | Value |
+|--------|-------|
+| Eval loss | 0.5595 |
+| Preference accuracy | 75.0% |
+| Reward margin | 2.69 |
+| Training time | ~24 minutes (Kaggle P100) |

---

## Model Architecture & Training

### LoRA Configuration

| Parameter | Value |
+|-----------|-------|
+| Base model | openai-community/gpt2-medium (355M params) |
+| LoRA rank (r) | 128 |
+| LoRA alpha (α) | 256 |
+| Trainable params | ~12MB adapter |
+| Target modules | q, k, v, o attention projections |

### DPO Training

**Method:** Direct Preference Optimization (DPO) — trains from preference pairs
without a separate reward model. The loss function is:

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)$$

Where $y_w$ is the cooperative ("won") response and $y_l$ is the violating ("lost") response.

| Hyperparameter | Value |
+|----------------|-------|
+| DPO β | 0.1 |
+| Learning rate | 5e-7 |
+| Batch size | 16 (effective, grad accum ×8) |
+| Epochs | 3 |
+| Training pairs | 1,970 filtered preference pairs |
+| Hardware | Kaggle P100-16GB |

### Training Data

Preference pairs come from three sources:

| Source | Pairs | Description |
|--------|-------|-------------|
+| Human-labeled | 411 | Expert-verified cooperative/violating pairs |
+| Repair-derived | ~1,200 | (original_violation, T5-repaired) pairs |
+| Synthetic (LLM) | ~1,200 | Generated via Groq API (llama-3.3-70b-versatile) |

A conflict-detection filter removed pairs where the "chosen" response scored
as more violating than the "rejected." Final: **1,970 clean pairs**.

---

## Files in This Repository

| File | Description |
|------|-------------|
| `adapter_config.json` | LoRA configuration (base model, rank, alpha) |
| `adapter_model.safetensors` | LoRA weights (25 MB) |
| `tokenizer.json` | GPT-2 tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |
| `special_tokens_map.json` | Special token mappings |

---

## Limitations

- **Manner violations persist:** DPO alone does not reduce Manner violation rate.
  The full pipeline (with GriceBench-Repair) is required to address Manner.
- **Single domain:** Trained and evaluated on Topical-Chat. Performance on other
  dialogue domains (task-oriented, medical, legal) is not characterized.
- **English only:** The system is trained exclusively on English dialogue.
- **Standalone cooperative rate (83.2%) is not the headline number:**
  The 95.0% cooperative rate requires the full pipeline. Using this model
  alone will not reproduce the system-level result.

---

## Citation

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
| GriceBench-Detector | Detects which maxim is violated | [🔍 Detector](https://huggingface.co/Pushkar27/GriceBench-Detector) |
| GriceBench-Repair | Repairs violations | [🔧 Repair Model](https://huggingface.co/Pushkar27/GriceBench-Repair) |
| GriceBench-DPO | Generates cooperative responses (this model) | You are here |

**GitHub:** https://github.com/PushkarPrabhath27/Research-Model
