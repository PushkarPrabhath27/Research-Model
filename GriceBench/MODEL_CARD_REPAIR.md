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
  - nlp
  - seq2seq
datasets:
  - topical_chat
metrics:
  - bleu
pipeline_tag: text2text-generation
base_model: google-t5/t5-base
---

<div align="center">

# 🔧 GriceBench-Repair

**Rewrites cooperative communication failures into compliant dialogue — surgically, not generally.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Part of the **GriceBench** system — [GitHub](https://github.com/PushkarPrabhath27/Research-Model) |
[🔍 Detector](https://huggingface.co/Pushkar27/GriceBench-Detector) |
[⚡ DPO Generator](https://huggingface.co/Pushkar27/GriceBench-DPO)

</div>

---

## What This Model Does

GriceBench-Repair is a seq2seq model that takes a dialogue response flagged
for Gricean maxim violations and rewrites it to be cooperative. Unlike generic
paraphrasing or self-refinement, it is **violation-type-aware**: it uses
different generation strategies depending on which maxim was violated.

| Violation | Strategy | Why |
|-----------|----------|-----|
+| **Quantity** | Beam search (n=4) + length constraints | Needs precise length control |
+| **Quality** | Beam search (n=4) + repetition penalty | Needs factual precision |
+| **Manner** | Nucleus sampling (T=0.85, p=0.92) | Needs diverse creative rewrites |
+| **Relation** | ❌ Not handled here | Relation requires full regeneration — route to FAISS retrieval |

---

## Quick Start

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Load model
model_name = "Pushkar27/GriceBench-Repair"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()

def repair_violation(
    context: str,
    response: str,
    violation_type: str,  # "quantity", "quality", or "manner"
) -> str:
    """
    Repair a Gricean maxim violation.
    
    Args:
        context: Conversation history
        response: The violating response to fix
        violation_type: Which maxim was violated
    
    Returns:
        Rewritten cooperative response
    
    Note: Relation violations should use FAISS retrieval, not this model.
    """
    assert violation_type in ["quantity", "quality", "manner"], \
        "Relation violations must use the FAISS retrieval system."
    
    input_text = f"fix {violation_type}: [CONTEXT] {context} [RESPONSE] {response}"
    inputs = tokenizer(
        input_text, return_tensors="pt",
        max_length=256, truncation=True
    )
    
    with torch.no_grad():
        if violation_type == "manner":
            # Nucleus sampling for diverse rewrites
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                max_length=128,
                min_length=8,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
            )
        else:
            # Beam search for precision
            output_ids = model.generate(
                **inputs,
                num_beams=4,
                max_length=128,
                min_length=8,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
            )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ── Examples ────────────────────────────────────────────────────────────────

# Example 1: Quantity violation (too short)
repaired = repair_violation(
    context="What do you think about the development of commercial space travel?",
    response="It's fine.",  # Under-informative
    violation_type="quantity"
)
print(f"Repaired: {repaired}")
# Repaired: "Commercial space travel has advanced remarkably, with companies like SpaceX
#            making orbital flight more accessible, though high costs remain a barrier."

# Example 2: Manner violation (ambiguous pronouns)
repaired = repair_violation(
    context="Alice told Bob that she would handle the project.",
    response="She said she would do it before she left.",  # Ambiguous pronouns
    violation_type="manner"
)
print(f"Repaired: {repaired}")
# Repaired: "Alice confirmed she would complete the project before leaving the office."
```

---

## Performance

**Violation removal rate: 93.0%** (corrected, post-fix evaluation on 200 samples)

Per-maxim BLEU scores on the repair validation set:

| Violation Type | BLEU Score | Notes |
|----------------|-----------|-------|
+| Quality | **97.8%** | Near-perfect factual correction |
+| Manner | **92.5%** | Excellent clarity improvements |
+| Quantity | **61.8%** | Requires insertion/deletion — harder task |
+| Relation | 9.3% | ⚠️ Intentionally routed to FAISS retrieval instead |

**Degeneracy fix results** (before/after applying violation-type-aware decoding):

| Maxim | Before Fix | After Fix | Improvement |
|-------|-----------|-----------|-------------|
+| Quantity | 30.1% degenerate | 2.1% degenerate | **+28.0pp** |
+| Manner | 93.3% degenerate | 4.5% degenerate | **+88.8pp** |
+| Overall | 64.4% degenerate | 5.2% degenerate | **+59.2pp** |

**Key lesson:** Using beam search for Manner repairs causes mode collapse (the model
inserts `!` punctuation as a proxy for "clarity"). Nucleus sampling eliminates this.

---

## Architecture

**Base model:** `google-t5/t5-base` (220M parameters)

**Input format:**
```
fix {violation_type}: [CONTEXT] {conversation_context} [RESPONSE] {response_to_fix}
```

Where `{violation_type}` ∈ `{quantity, quality, manner}`.

**Three-layer degeneracy prevention:**
1. **Generation routing** — violation-type-aware decoding strategy (see above)
2. **Post-generation validation** — multi-signal degeneracy filter (punctuation bursts,
   trigram repetition, exclamation density, character-level repetition)
3. **Graceful fallback** — if all repair attempts produce degenerate output, returns
   the original response with a `is_fallback: True` flag

---

## Training Details

| Hyperparameter | Value |
+|----------------|-------|
+| Base model | google-t5/t5-base |
+| Training pairs | 3,210 seq2seq (violation → cooperative) pairs |
+| Validation pairs | 401 pairs |
+| Epochs | 5 |
+| Decoding (Qty/Ql) | Beam search, beam=4 |
+| Decoding (Manner) | Nucleus sampling, T=0.85, top-p=0.92 |
+| Label smoothing | 0.1 |
+| Hardware | Kaggle T4 |

---

## Important: Relation Violations

Relation violations (off-topic responses) **cannot be addressed by editing** — they
require generating entirely new, topically relevant content. This model's seq2seq
framing asks it to "fix" the existing response by editing, but there is nothing
to fix by editing when the entire response is off-topic.

For Relation violations, use the **FAISS retrieval system** included in the
GriceBench repository:
- 50,000 Topical-Chat responses indexed with FAISS
- MRR > 0.70, Top-1 accuracy > 60%
- See `data_processed/relation_repair/` in the GitHub repo

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
| GriceBench-Repair | Repairs violations (this model) | You are here |
| GriceBench-DPO | Generates cooperative responses | [⚡ DPO Generator](https://huggingface.co/Pushkar27/GriceBench-DPO) |

**GitHub:** https://github.com/PushkarPrabhath27/Research-Model
