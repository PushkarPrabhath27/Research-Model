---
language:
  - en
license: apache-2.0
library_name: transformers
tags:
  - text2text-generation
  - dialogue
  - gricean-maxims
  - cooperative-communication
  - t5
  - text-repair
  - seq2seq
  - nlp
datasets:
  - topical_chat
metrics:
  - bleu
pipeline_tag: text2text-generation
base_model: google-t5/t5-base
---

<div align="center">

# 🛠 GriceBench-Repair

**Rewrites Gricean maxim violations into cooperative dialogue — surgically, not generally.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace](https://img.shields.io/badge/🤗-GriceBench-yellow)](https://huggingface.co/Pushkar27)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Part of the GriceBench system** —
[GitHub](https://github.com/PushkarPrabhath27/Research-Model) |
[🔍 Detector](https://huggingface.co/Pushkar27/GriceBench-Detector) |
[⚡ DPO Generator](https://huggingface.co/Pushkar27/GriceBench-DPO)

</div>

---

## What This Model Does

GriceBench-Repair is a T5-base seq2seq model that rewrites Gricean maxim violations
into cooperative responses. It is **violation-type-aware**: different maxims use
different generation strategies because the nature of the repair task differs.

| Violation | Decoding Strategy | Why |
|-----------|------------------|-----|
| **Quantity** | Beam search (n=4) + length constraints | Needs precise length control |
| **Quality** | Beam search (n=4) + repetition penalty | Needs factual precision |
| **Manner** | Nucleus sampling (T=0.85, top-p=0.92) | Needs creative diverse rewrites |
| **Relation** | ❌ Not this model — use FAISS retrieval | Entire response is off-topic; editing can't fix it |

**Violation removal rate: 93.0%** (post-fix evaluation, N=200)

---

## Quick Start

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model_name = "Pushkar27/GriceBench-Repair"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()

def repair_violation(context: str, response: str, violation_type: str) -> str:
    """
    Repair a Gricean maxim violation.

    Args:
        context:        Conversation history
        response:       The violating response to fix
        violation_type: One of "quantity", "quality", "manner"
                        (Relation → use FAISS retrieval instead)
    Returns:
        Rewritten cooperative response string
    """
    assert violation_type in ["quantity", "quality", "manner"], \
        "Relation violations must use the FAISS retrieval system — not this model."

    input_text = f"fix {violation_type}: [CONTEXT] {context} [RESPONSE] {response}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=256, truncation=True)

    with torch.no_grad():
        if violation_type == "manner":
            # Nucleus sampling — beam search degenerates for Manner
            output_ids = model.generate(
                **inputs,
                do_sample=True, temperature=0.85, top_p=0.92,
                max_length=128, min_length=8,
                repetition_penalty=1.5, no_repeat_ngram_size=3,
            )
        else:
            # Beam search for precision
            output_ids = model.generate(
                **inputs,
                num_beams=4, max_length=128, min_length=8,
                repetition_penalty=1.5, no_repeat_ngram_size=3,
            )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ─── Examples ────────────────────────────────────────────────────────────────

# Quantity (too short)
print(repair_violation(
    context="What do you think about commercial space travel?",
    response="It's fine.",
    violation_type="quantity"
))
# → "Commercial space travel has advanced rapidly, with reusable rockets
#    making orbital access cheaper, though costs remain high for most."

# Manner (ambiguous pronouns)
print(repair_violation(
    context="Alice told Bob she would handle the project.",
    response="She said she would do it before she left.",
    violation_type="manner"
))
# → "Alice confirmed she would complete the project before leaving the office."
```

---

## Performance

**Violation removal rate: 93.0%** (corrected, post-fix evaluation)

Per-maxim BLEU scores on the repair validation set (N=401):

| Violation Type | BLEU | Notes |
|----------------|------|-------|
| Quality | **97.8%** | Near-perfect factual correction |
| Manner | **92.5%** | Strong clarity improvements |
| Quantity | 61.8% | Harder — requires insertions/deletions |
| Relation | 9.3% | ⚠️ Route to FAISS retrieval — do not use T5 for this |

**Degeneracy fix (before vs. after violation-type-aware decoding):**

| Maxim | Before Fix | After Fix | Improvement |
|-------|-----------|-----------|-------------|
| Quantity | 30.1% degenerate | 2.1% | **−28.0pp** |
| Manner | 93.3% degenerate | 4.5% | **−88.8pp** |
| Overall | 64.4% degenerate | 5.2% | **−59.2pp** |

> **Key lesson:** Beam search produces mode-collapsed outputs for Manner repairs
> (model inserts `!` as a proxy for "clarity"). Nucleus sampling eliminates this.

---

## Architecture & Training

- **Base model:** `google-t5/t5-base` (220M parameters)
- **Training pairs:** 3,210 (violation → cooperative) seq2seq pairs
- **Validation pairs:** 401
- **Epochs:** 5 | **Label smoothing:** 0.1 | **Hardware:** Kaggle T4

**Three-layer degeneracy prevention:**
1. Violation-type-aware decoding (nucleus sampling for Manner, beam for others)
2. Post-generation multi-signal filter (punctuation bursts, trigram repetition, exclamation density)
3. Graceful fallback — returns original with `is_fallback: True` flag if all attempts fail

---

## Why Relation Violations Use Retrieval

Relation violations mean the *entire response* is off-topic — there is nothing to
edit. T5 in a seq2seq framing can only edit, not generate entirely new content.
We route Relation repairs to a FAISS index over 50,000 Topical-Chat responses
(MRR > 0.70, Top-1 accuracy > 60%). See the GitHub repo for the full retrieval system.

---

## Files

| File | Description |
|------|-------------|
| `config.json` | T5-base configuration |
| `model.safetensors` | Trained model weights (891 MB) |
| `tokenizer.json` | SentencePiece tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |

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
| GriceBench-Detector | Detects which maxim was violated | [🔍 Detector](https://huggingface.co/Pushkar27/GriceBench-Detector) |
| GriceBench-Repair | Repairs violations (this model) | You are here |
| GriceBench-DPO | Generates cooperative responses | [⚡ DPO](https://huggingface.co/Pushkar27/GriceBench-DPO) |

**GitHub:** https://github.com/PushkarPrabhath27/Research-Model
