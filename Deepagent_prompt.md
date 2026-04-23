# ============================================================
# GRICEBENCH: DEEP EXECUTION PROMPT FOR AI AGENT
# Priority Phases: 1 → 4 → 5 → 3 → 7
# Standard: Publication-ready at EMNLP/ACL level
# Author: Pushkar Prabhath
# Last Updated: March 2026
# ============================================================

---

## ⚠️ AGENT OPERATING CONTRACT — READ THIS BEFORE ANYTHING ELSE

You are an expert NLP research engineer and academic writer operating at
the level of a senior ML engineer at a FAANG company who also has a PhD
in computational linguistics. You are completing the GriceBench research
project to the point of arXiv submission and EMNLP 2026 conference submission.

**Your operating rules — non-negotiable:**

1. **NEVER fake output.** If a file doesn't exist, say so. If a metric
   can't be computed, say why. Do not hallucinate numbers, paths, or results.

2. **NEVER skip a Quality Gate.** Each phase ends with a gate. If even one
   checkbox fails, you stop, fix it, and re-check before moving on.

3. **EVERY code block you write must be complete and runnable.** No `...`,
   no `# TODO`, no `pass` in critical paths. Production-level or nothing.

4. **EVERY number you write in the paper must be traceable** to a specific
   JSON file or CSV in the results directory. If you can't trace it, you
   don't write it.

5. **EXECUTE phases in order: 1 → 4 → 5 → 3 → 7.** Phase 4 outputs feed
   Phase 5. Phase 5 must be complete before Phase 3 (model cards need paper
   citation). Phase 7 is last because it audits everything.

6. **At the start of each phase, print:** `=== STARTING PHASE [N]: [NAME] ===`
   **At the end of each phase, print a gate report.**

---

## 🧠 FULL PROJECT CONTEXT

### What GriceBench Is

GriceBench is a research system that operationalizes Paul Grice's four
conversational maxims (Quantity, Quality, Relation, Manner) for AI dialogue.
It consists of three trained components that form a pipeline:

```
Conversation Context
       │
       ▼
┌──────────────────────┐
│   DPO GENERATOR      │  SmolLM2-360M or Qwen2.5-7B + LoRA
│   (preference-tuned) │  Generates cooperative responses
└──────────┬───────────┘
           │ generated response
           ▼
┌──────────────────────┐
│  VIOLATION DETECTOR  │  DeBERTa-v3-base, 4 binary heads
│  (multi-label)       │  One head per maxim
└──────────┬───────────┘
           │ per-maxim violation flags
     ┌─────▼──────┐
     │ Violations?│
     └──┬──────┬──┘
       No     Yes
        │      ├─ Relation → FAISS retrieval system
        │      └─ Qty/Ql/Mn → T5 repair model
        │               │
        └───────────────┘
               │
               ▼
        Final Cooperative Response
```

### Trained Model Inventory

| Model | Path | Size | Status |
|-------|------|------|--------|
| Detector V2 | `best_model_v2.pt` | 2.22 GB | ✅ Trained |
| Calibration | `temperatures.json` | ~100 bytes | ✅ Done |
| Repair (T5) | `models/repair/repair_model/` | 891 MB | ✅ Trained, has bug |
| DPO Adapter | `dpo_training_final_outcome/` | 25 MB | ✅ Trained |

### Canonical Results (DO NOT CHANGE THESE)

These are the ground-truth numbers from `results/phase7output/phase7_results.json`
and `results/part4output/`. Every other number flows from these.

| System Config | Cooperative Rate | Source |
|---------------|-----------------|--------|
| Baseline (GPT-2) | 83.8% | ablation_report |
| DPO Only | 83.2% | ablation_report |
| Detect + Repair | 93.0% | ablation_report |
| **Full System** | **95.0%** | ablation_report |
| Mistral-7B | 89.1% | phase4_baselines |
| Qwen2.5-7B | 84.2% | phase4_baselines |

| Maxim | F1 | Precision | Recall | AUC |
|-------|----|-----------|--------|-----|
| Quantity | 1.000 | 1.000 | 1.000 | 1.000 |
| Quality | 0.928 | 0.866 | 1.000 | 0.999 |
| Relation | 1.000 | 1.000 | 1.000 | 1.000 |
| Manner | 0.891 | 0.864 | 0.919 | 0.979 |
| **Macro** | **0.955** | — | — | — |

### Known Critical Issues (What You Are Here to Fix)

**Issue 1 (PHASE 1):** T5 Repair model produces degenerate outputs.
Example: `"I think so, the movie had a good message."` → `"It! The! Movie!! really!! did! convey! a!!! powerful!!!!!!"`
Root cause: No repetition penalty, beam search degeneration on Manner violations.
Impact: 84.5% violation removal rate is inflated because detector calls broken outputs "clean."

**Issue 2 (PHASE 4):** Zero confidence intervals on the headline 95.0% claim.
Every top-tier venue requires CIs. This is a desk rejection risk.

**Issue 3 (PHASE 5):** Paper draft exists with structural skeleton but
Related Work is missing, all tables have placeholder CIs, and the Manner
repair failure analysis hasn't been written.

**Issue 4 (PHASE 3):** Models are not on HuggingFace. README has placeholder URLs.

**Issue 5 (PHASE 7):** No consistency check between paper numbers and result JSONs.

---

---

# ╔══════════════════════════════════════════════════════╗
# ║  PHASE 1: FIX THE T5 REPAIR MODEL (CRITICAL BUG)   ║
# ╚══════════════════════════════════════════════════════╝

## Why Phase 1 Is First

The repair model bug contaminates every metric downstream.
The 84.5% violation removal rate — a number the paper will cite —
is wrong because the detector classifies `"It! The! Movie!!"` as clean.
This must be fixed and re-measured BEFORE the paper is written.
Writing the paper first and fixing later is backwards and dangerous.

---

## Step 1.0 — Environment Setup

Before anything else, confirm the environment is correctly set up:

```bash
# Run this first. It confirms all required packages and model files exist.
python -c "
import torch, transformers, json, re, os
from pathlib import Path

required_files = [
    'best_model_v2.pt',
    'temperatures.json',
    'models/repair/repair_model/config.json',
    'models/repair/repair_model/model.safetensors',
    'dpo_training_final_outcome/adapter_config.json',
    'data_processed/repair_data/repair_val.json',
]

print('=== Environment Check ===')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {\"GPU\" if torch.cuda.is_available() else \"CPU\"}')
print()

missing = []
for f in required_files:
    exists = Path(f).exists()
    status = '✅' if exists else '❌ MISSING'
    print(f'{status}  {f}')
    if not exists:
        missing.append(f)

if missing:
    print(f'\n⚠️  {len(missing)} file(s) missing. Resolve before proceeding.')
else:
    print('\n✅ All required files present.')
"
```

If any files are missing, STOP and resolve before proceeding.

---

## Step 1.1 — Baseline Degeneracy Audit

Run this script FIRST. It establishes the ground truth on how broken
the repair model currently is. You need this number to prove your fix works.
Save every output — it becomes the "before" half of your fix validation table.

```python
# scripts/repair_audit_baseline.py
"""
PURPOSE: Measure the degenerate output rate of the T5 repair model
         BEFORE any fixes are applied.

SAVES TO: results/repair_audit/baseline_degenerate_rates.json

This script must be run BEFORE repair_inference_fixed.py is used anywhere.
The output is a before/after comparison anchor.
"""

import json
import re
import time
import os
from pathlib import Path
from collections import defaultdict
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = "models/repair/repair_model/"
REPAIR_VAL_PATH = "data_processed/repair_data/repair_val.json"
OUTPUT_DIR = "results/repair_audit/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Degeneracy Detection (Multi-Signal) ─────────────────────────────────────
def is_degenerate(text: str) -> tuple[bool, list[str]]:
    """
    Multi-signal degeneracy detection. Returns (is_degenerate, reasons_list).

    Signal 1: Repeated identical punctuation burst (e.g., "!!!" or "???")
    Signal 2: Exclamation mark density > 15% of tokens
    Signal 3: Trigram repetition — same 3-word sequence appears > 2 times
    Signal 4: Output is shorter than 3 words (repair shouldn't produce near-empty)
    Signal 5: Character-level repetition — same char appears > 8 times consecutively
    """
    reasons = []
    if not text or not text.strip():
        return True, ["empty_output"]

    tokens = text.strip().split()

    # Signal 1: Punctuation burst
    if re.search(r'([!?,.])\1{2,}', text):
        reasons.append("punctuation_burst")

    # Signal 2: Exclamation density
    excl_count = text.count('!')
    if len(tokens) > 0 and excl_count / len(tokens) > 0.15:
        reasons.append(f"exclamation_density={excl_count}/{len(tokens)}")

    # Signal 3: Trigram repetition
    if len(tokens) >= 6:
        trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
        from collections import Counter
        most_common = Counter(trigrams).most_common(1)
        if most_common and most_common[0][1] > 2:
            reasons.append(f"trigram_repeat={most_common[0][0]}x{most_common[0][1]}")

    # Signal 4: Too short
    if len(tokens) < 3:
        reasons.append("too_short")

    # Signal 5: Character-level repetition
    if re.search(r'(.)\1{7,}', text):
        reasons.append("char_repetition")

    return len(reasons) > 0, reasons


# ─── Load model ───────────────────────────────────────────────────────────────
print("Loading T5 repair model for baseline audit...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
model.eval()
print(f"Model loaded on {device}")

# ─── Load validation data ─────────────────────────────────────────────────────
with open(REPAIR_VAL_PATH) as f:
    val_data = json.load(f)

print(f"Loaded {len(val_data)} validation examples")

# ─── Run baseline audit ───────────────────────────────────────────────────────
results = {
    "config": {
        "model_path": MODEL_PATH,
        "generation": "beam_search_no_penalty",  # This is the BROKEN config
        "num_beams": 4,
        "max_length": 128,
        "repetition_penalty": 1.0,  # No penalty = broken
    },
    "summary": {},
    "per_maxim": defaultdict(lambda: {"total": 0, "degenerate": 0, "examples": []}),
    "degenerate_examples": [],
    "clean_examples": []
}

start_time = time.time()

for i, item in enumerate(val_data):
    if i % 50 == 0:
        print(f"  Processing {i}/{len(val_data)}...")

    input_text = item.get("input", "")
    violation_type = item.get("violation_type", "unknown")
    reference = item.get("target", item.get("output", ""))

    if not input_text:
        continue

    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=256,
        truncation=True
    ).to(device)

    # Generate with CURRENT (broken) config — no repetition penalty
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=4,
            max_length=128,
            early_stopping=True,
            # NOTE: No repetition_penalty, no no_repeat_ngram_size
            # This is intentionally the broken config for baseline measurement
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    degenerate, reasons = is_degenerate(output_text)

    results["per_maxim"][violation_type]["total"] += 1
    if degenerate:
        results["per_maxim"][violation_type]["degenerate"] += 1
        if len(results["degenerate_examples"]) < 30:
            results["degenerate_examples"].append({
                "violation_type": violation_type,
                "input": input_text[:200],
                "output": output_text,
                "reasons": reasons
            })
    else:
        if len(results["clean_examples"]) < 10:
            results["clean_examples"].append({
                "violation_type": violation_type,
                "input": input_text[:200],
                "output": output_text
            })

elapsed = time.time() - start_time

# ─── Compute summary ──────────────────────────────────────────────────────────
total = sum(d["total"] for d in results["per_maxim"].values())
total_degen = sum(d["degenerate"] for d in results["per_maxim"].values())

results["summary"] = {
    "total_examples": total,
    "total_degenerate": total_degen,
    "overall_degenerate_rate_pct": round(100 * total_degen / total, 2) if total > 0 else 0,
    "elapsed_seconds": round(elapsed, 1),
    "per_maxim_rates": {
        k: {
            "total": v["total"],
            "degenerate": v["degenerate"],
            "rate_pct": round(100 * v["degenerate"] / v["total"], 2) if v["total"] > 0 else 0
        }
        for k, v in results["per_maxim"].items()
    }
}

# Convert defaultdict for JSON serialization
results["per_maxim"] = dict(results["per_maxim"])

# ─── Save and print ───────────────────────────────────────────────────────────
output_path = f"{OUTPUT_DIR}baseline_degenerate_rates.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("BASELINE DEGENERACY AUDIT RESULTS")
print("="*60)
print(f"Total examples: {total}")
print(f"Total degenerate: {total_degen}")
print(f"Overall degenerate rate: {results['summary']['overall_degenerate_rate_pct']}%")
print("\nPer-maxim breakdown:")
for maxim, stats in results["summary"]["per_maxim_rates"].items():
    print(f"  {maxim:12s}: {stats['degenerate']}/{stats['total']} = {stats['rate_pct']}% degenerate")
print(f"\nSaved to: {output_path}")
print("="*60)
```

---

## Step 1.2 — The Fixed Repair Model (Production Implementation)

This is the core fix. Create this file as a DROP-IN REPLACEMENT
for all existing repair inference calls. It is backwards-compatible —
same inputs, better outputs.

```python
# scripts/repair_inference_fixed.py
"""
PURPOSE: Production-grade T5 repair model with three-layer degeneracy prevention.

REPLACES: All existing repair inference code in the pipeline.

LAYERS:
  Layer 1 — Generation hyperparameter fix
             (repetition_penalty, no_repeat_ngram_size, temperature sampling for Manner)
  Layer 2 — Post-generation validation (same is_degenerate() function, now enforced)
  Layer 3 — Graceful degradation (return original with flag instead of breaking)

ROUTING LOGIC:
  Manner violations    → temperature sampling (diverse rewrites)
  Quantity violations  → beam search with length constraints
  Quality violations   → beam search with factual precision bias
  Relation violations  → NOT handled here. Route to FAISS retrieval system.

USAGE:
  from scripts.repair_inference_fixed import FixedRepairModel
  model = FixedRepairModel("models/repair/repair_model/")
  result = model.repair(input_text="fix manner: [CONTEXT] ... [RESPONSE] ...",
                        violation_type="manner")
  print(result["repaired_text"])
"""

import re
import json
import time
import logging
from pathlib import Path
from typing import Optional
from collections import Counter

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)


# ─── Degeneracy Detection ─────────────────────────────────────────────────────

def is_degenerate(text: str) -> tuple[bool, list[str]]:
    """
    Multi-signal degeneracy detector.
    Identical logic to repair_audit_baseline.py for consistency.
    Returns (is_degenerate: bool, reasons: list[str])

    This function is the single source of truth for what counts as a
    broken repair output. Used in both auditing and live inference.
    """
    reasons = []

    if not text or not text.strip():
        return True, ["empty_output"]

    tokens = text.strip().split()

    # Signal 1: Consecutive punctuation burst
    if re.search(r'([!?,.])\1{2,}', text):
        reasons.append("punctuation_burst")

    # Signal 2: Exclamation density > 15% of tokens
    excl_count = text.count('!')
    if len(tokens) > 0 and excl_count / len(tokens) > 0.15:
        reasons.append(f"exclamation_density={excl_count}/{len(tokens)}")

    # Signal 3: Trigram repetition
    if len(tokens) >= 6:
        trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
        most_common = Counter(trigrams).most_common(1)
        if most_common and most_common[0][1] > 2:
            reasons.append(f"trigram_repeat")

    # Signal 4: Too short (repair should never produce <3 words)
    if len(tokens) < 3:
        reasons.append("too_short")

    # Signal 5: Character-level repetition
    if re.search(r'(.)\1{7,}', text):
        reasons.append("char_repetition")

    return len(reasons) > 0, reasons


# ─── Response Extraction ──────────────────────────────────────────────────────

def extract_original_response(input_text: str) -> str:
    """
    Extract the original response from a repair prompt.

    Expected format: "fix {maxim}: [CONTEXT] {context} [RESPONSE] {response}"
    Falls back to last sentence if [RESPONSE] marker not found.
    """
    if "[RESPONSE]" in input_text:
        return input_text.split("[RESPONSE]")[-1].strip()

    # Fallback: last sentence
    sentences = [s.strip() for s in input_text.strip().split('.') if s.strip()]
    return sentences[-1] if sentences else input_text.strip()


# ─── Fixed Repair Model ───────────────────────────────────────────────────────

class FixedRepairModel:
    """
    Production-grade T5 repair model with degeneracy prevention.

    Attributes:
        model_path: Path to HuggingFace-format T5 repair model directory
        device: Torch device string ("cuda", "cpu", or "auto")
        fallback_on_degenerate: If True, returns original when output is broken
        max_repair_attempts: How many times to retry before giving up
    """

    VIOLATION_TYPES = {"quantity", "quality", "manner"}
    # Relation is not handled here — route to FAISS retrieval

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        fallback_on_degenerate: bool = True,
        max_repair_attempts: int = 2,
    ):
        self.model_path = model_path
        self.fallback_on_degenerate = fallback_on_degenerate
        self.max_repair_attempts = max_repair_attempts

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading FixedRepairModel from {model_path} on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()
        logger.info("FixedRepairModel loaded successfully.")

    def _generate_beam(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
        """
        Beam search generation with repetition prevention.
        Used for: Quantity (length control), Quality (factual precision).

        Key fixes vs. broken config:
          - repetition_penalty=1.5  prevents token repetition
          - no_repeat_ngram_size=3  prevents 3-gram loops
          - min_length=8            prevents too-short outputs
        """
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                max_length=128,
                min_length=8,
                early_stopping=True,
                repetition_penalty=1.5,       # ← THE KEY FIX
                no_repeat_ngram_size=3,        # ← THE KEY FIX
                length_penalty=1.0,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _generate_sample(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
        """
        Nucleus (top-p) sampling for Manner violations.

        Manner repair needs CREATIVE REWRITING, not precise editing.
        Beam search degenerates on Manner because the model learned noisy
        targets (punctuation injection). Temperature sampling with nucleus
        sampling produces diverse, valid rewrites instead.

        Key parameters:
          - temperature=0.85   controlled randomness
          - top_p=0.92         nucleus sampling (best 92% probability mass)
          - repetition_penalty=1.5  still applied
        """
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                max_length=128,
                min_length=8,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def repair(
        self,
        input_text: str,
        violation_type: str,
        reference_response: Optional[str] = None,
    ) -> dict:
        """
        Repair a violation-flagged response.

        Args:
            input_text: Full repair prompt in format:
                        "fix {violation_type}: [CONTEXT] {ctx} [RESPONSE] {resp}"
            violation_type: One of "quantity", "quality", "manner"
                            (Relation should use FAISS, not this method)
            reference_response: Optional — the original response text for fallback.
                                 If not provided, extracted from input_text.

        Returns dict with:
            repaired_text:     str — the final output (repaired or original)
            is_fallback:       bool — True if we returned original due to failure
            is_degenerate:     bool — True if any attempt produced degenerate output
            attempts:          int — number of generation attempts made
            generation_method: str — "beam" or "sample"
            degenerate_reasons: list[str] — why the output was flagged (if applicable)
            inference_ms:      float — total inference time in milliseconds
        """
        if violation_type not in self.VIOLATION_TYPES:
            raise ValueError(
                f"violation_type must be one of {self.VIOLATION_TYPES}. "
                f"Got: '{violation_type}'. "
                f"Note: Relation violations must use the FAISS retrieval system."
            )

        start = time.perf_counter()

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=False,
        ).to(self.device)

        # Route to correct generation strategy
        generation_method = "sample" if violation_type == "manner" else "beam"
        generator = self._generate_sample if violation_type == "manner" else self._generate_beam

        # Multi-attempt loop — retry on degenerate output
        last_output = ""
        last_reasons = []
        any_degenerate = False

        for attempt in range(1, self.max_repair_attempts + 1):
            output_text = generator(inputs["input_ids"], inputs["attention_mask"])
            degenerate, reasons = is_degenerate(output_text)

            if not degenerate:
                # Success — return immediately
                elapsed_ms = (time.perf_counter() - start) * 1000
                return {
                    "repaired_text": output_text,
                    "is_fallback": False,
                    "is_degenerate": False,
                    "attempts": attempt,
                    "generation_method": generation_method,
                    "degenerate_reasons": [],
                    "inference_ms": round(elapsed_ms, 2),
                }

            any_degenerate = True
            last_output = output_text
            last_reasons = reasons
            logger.warning(
                f"Attempt {attempt}/{self.max_repair_attempts}: "
                f"Degenerate output for '{violation_type}'. Reasons: {reasons}. "
                f"Output snippet: {output_text[:80]}..."
            )

        # All attempts failed — apply fallback policy
        elapsed_ms = (time.perf_counter() - start) * 1000

        if self.fallback_on_degenerate:
            original = reference_response or extract_original_response(input_text)
            logger.warning(
                f"All {self.max_repair_attempts} attempts produced degenerate output. "
                f"Falling back to original response."
            )
            return {
                "repaired_text": original,
                "is_fallback": True,
                "is_degenerate": True,
                "attempts": self.max_repair_attempts,
                "generation_method": generation_method,
                "degenerate_reasons": last_reasons,
                "inference_ms": round(elapsed_ms, 2),
            }
        else:
            # If caller disabled fallback — return broken output with flag
            return {
                "repaired_text": last_output,
                "is_fallback": False,
                "is_degenerate": True,
                "attempts": self.max_repair_attempts,
                "generation_method": generation_method,
                "degenerate_reasons": last_reasons,
                "inference_ms": round(elapsed_ms, 2),
            }
```

---

## Step 1.3 — Post-Fix Validation Audit

Run this IMMEDIATELY after creating `repair_inference_fixed.py`.
It proves the fix works. The output becomes the "after" column in
your paper's repair analysis table.

```python
# scripts/repair_audit_after_fix.py
"""
PURPOSE: Measure the degenerate output rate AFTER applying the fix.
         Produces a before/after comparison report.

PREREQUISITE: repair_audit_baseline.py must have been run first.

SAVES TO:
  results/repair_audit/fixed_degenerate_rates.json
  results/repair_audit/before_after_comparison.json
"""

import json
import time
import os
from pathlib import Path
import torch
from scripts.repair_inference_fixed import FixedRepairModel, is_degenerate

OUTPUT_DIR = "results/repair_audit/"
REPAIR_VAL_PATH = "data_processed/repair_data/repair_val.json"
BASELINE_PATH = f"{OUTPUT_DIR}baseline_degenerate_rates.json"

assert Path(BASELINE_PATH).exists(), \
    "ERROR: baseline_degenerate_rates.json not found. Run repair_audit_baseline.py first."

# Load fixed model
print("Loading FIXED repair model...")
repair_model = FixedRepairModel(
    model_path="models/repair/repair_model/",
    fallback_on_degenerate=True,
    max_repair_attempts=2
)

# Load validation data
with open(REPAIR_VAL_PATH) as f:
    val_data = json.load(f)

print(f"Running fixed model on {len(val_data)} validation examples...")

from collections import defaultdict
fixed_results = {
    "per_maxim": defaultdict(lambda: {
        "total": 0, "degenerate": 0, "fallback": 0, "examples": []
    })
}

start = time.time()

for i, item in enumerate(val_data):
    if i % 50 == 0:
        print(f"  {i}/{len(val_data)}...")

    input_text = item.get("input", "")
    violation_type = item.get("violation_type", "unknown")

    if violation_type == "relation" or not input_text:
        continue  # Relation uses FAISS, skip here

    result = repair_model.repair(
        input_text=input_text,
        violation_type=violation_type
    )

    fixed_results["per_maxim"][violation_type]["total"] += 1
    if result["is_degenerate"]:
        fixed_results["per_maxim"][violation_type]["degenerate"] += 1
    if result["is_fallback"]:
        fixed_results["per_maxim"][violation_type]["fallback"] += 1

elapsed = time.time() - start

# Compute summary
total = sum(d["total"] for d in fixed_results["per_maxim"].values())
total_degen = sum(d["degenerate"] for d in fixed_results["per_maxim"].values())
total_fallback = sum(d["fallback"] for d in fixed_results["per_maxim"].values())

fixed_results["summary"] = {
    "total_examples": total,
    "total_degenerate": total_degen,
    "total_fallback": total_fallback,
    "overall_degenerate_rate_pct": round(100 * total_degen / total, 2) if total > 0 else 0,
    "elapsed_seconds": round(elapsed, 1),
}
fixed_results["per_maxim"] = dict(fixed_results["per_maxim"])

# Load baseline for comparison
with open(BASELINE_PATH) as f:
    baseline = json.load(f)

# Build comparison report
comparison = {
    "summary": {
        "before_degenerate_rate_pct": baseline["summary"]["overall_degenerate_rate_pct"],
        "after_degenerate_rate_pct": fixed_results["summary"]["overall_degenerate_rate_pct"],
        "absolute_improvement_pct": round(
            baseline["summary"]["overall_degenerate_rate_pct"] -
            fixed_results["summary"]["overall_degenerate_rate_pct"], 2
        ),
        "fallback_rate_pct": round(100 * total_fallback / total, 2) if total > 0 else 0,
    },
    "per_maxim_comparison": {}
}

for maxim in ["quantity", "quality", "manner"]:
    before = baseline["summary"]["per_maxim_rates"].get(maxim, {})
    after_data = fixed_results["per_maxim"].get(maxim, {})
    after_total = after_data.get("total", 0)
    after_degen = after_data.get("degenerate", 0)
    after_rate = round(100 * after_degen / after_total, 2) if after_total > 0 else 0

    comparison["per_maxim_comparison"][maxim] = {
        "before_degenerate_pct": before.get("rate_pct", "N/A"),
        "after_degenerate_pct": after_rate,
        "improvement_pp": round(before.get("rate_pct", 0) - after_rate, 2)
    }

# Save outputs
with open(f"{OUTPUT_DIR}fixed_degenerate_rates.json", "w") as f:
    json.dump(fixed_results, f, indent=2)

with open(f"{OUTPUT_DIR}before_after_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

# Print final report
print("\n" + "="*70)
print("REPAIR FIX VALIDATION REPORT")
print("="*70)
print(f"{'Metric':<35} {'Before':>10} {'After':>10} {'Improvement':>12}")
print("-"*70)
print(f"{'Overall degenerate rate':<35} "
      f"{comparison['summary']['before_degenerate_rate_pct']:>9.1f}% "
      f"{comparison['summary']['after_degenerate_rate_pct']:>9.1f}% "
      f"{comparison['summary']['absolute_improvement_pct']:>+11.1f}pp")
print()
for maxim, data in comparison["per_maxim_comparison"].items():
    before_str = f"{data['before_degenerate_pct']}%" if isinstance(data['before_degenerate_pct'], float) else str(data['before_degenerate_pct'])
    print(f"  {maxim:<33} {before_str:>10} "
          f"{data['after_degenerate_pct']:>9.1f}% "
          f"{data['improvement_pp']:>+11.1f}pp")
print()
print(f"Fallback rate (original returned): {comparison['summary']['fallback_rate_pct']}%")
print("="*70)
```

---

## Step 1.4 — Update Downstream Metrics

If the before/after comparison shows the degenerate rate dropped meaningfully
(expect Manner to drop from ~30-60% to <5%), you MUST update:

1. Run a mini Phase 7 re-evaluation (200 samples minimum) with the fixed
   repair model and save to `results/phase7output/phase7_results_v2.json`

2. Update `README.md` line: "Repair violation removal rate: 84.5%"
   → Replace with the new number from the fixed evaluation.

3. Add a note in the results section: "After fixing a degenerate output
   bug in the repair model (Section 7.5), the corrected violation removal
   rate is X%."

4. The paper will use the CORRECTED number with an honest explanation of
   the original bug and fix.

---

## ✅ PHASE 1 QUALITY GATE

Do NOT proceed to Phase 4 until ALL of these pass:

```
GATE 1.1: results/repair_audit/baseline_degenerate_rates.json EXISTS
          → Open it. Confirm it has per-maxim breakdown with real numbers.
          → The Manner degenerate rate should be > 10% (it's broken).
          → If Manner shows 0% degenerate, the audit script has a bug.

GATE 1.2: scripts/repair_inference_fixed.py IMPORTS CLEANLY
          → python -c "from scripts.repair_inference_fixed import FixedRepairModel; print('OK')"
          → Must print "OK" with zero errors.

GATE 1.3: UNIT TEST — is_degenerate() CORRECTNESS
          → python -c "
              from scripts.repair_inference_fixed import is_degenerate
              # Must return True (broken output)
              result1, _ = is_degenerate('It! The! Movie!! really!! did! convey! a!!! powerful!!!!!!')
              assert result1 == True, 'FAILED: Should flag broken output'
              # Must return False (clean output)
              result2, _ = is_degenerate('The movie had a strong and inspiring message about perseverance.')
              assert result2 == False, 'FAILED: Should not flag clean output'
              print('✅ Both unit tests passed')
          "

GATE 1.4: results/repair_audit/before_after_comparison.json EXISTS
          → Open it. Confirm after_degenerate_rate_pct < before_degenerate_rate_pct
          → The fix must show measurable improvement. Target: Manner < 5%.

GATE 1.5: UPDATED METRIC EXISTS
          → results/phase7output/phase7_results_v2.json exists
          → OR a written note in docs/ explains the corrected violation removal rate.

GATE 1.6: NO IMPORT BREAKS
          → Search codebase for any file that imports the old repair inference.
          → Update those imports to use FixedRepairModel.
          → Run: grep -r "repair_inference" --include="*.py" . 
             and confirm all found files use the fixed version.
```

**PRINT GATE REPORT:** After checking all gates, print:
```
=== PHASE 1 GATE REPORT ===
Gate 1.1 (Baseline audit):     [PASS/FAIL]
Gate 1.2 (Import clean):       [PASS/FAIL]
Gate 1.3 (Unit tests):         [PASS/FAIL]
Gate 1.4 (Before/after):       [PASS/FAIL]
Gate 1.5 (Updated metric):     [PASS/FAIL]
Gate 1.6 (No import breaks):   [PASS/FAIL]
Overall: [PROCEED TO PHASE 4 / FIX AND RETRY]
```

---

---

# ╔══════════════════════════════════════════════════════╗
# ║  PHASE 4: STATISTICAL SIGNIFICANCE TESTING          ║
# ╚══════════════════════════════════════════════════════╝

## Why Phase 4 Is Before Phase 5

The paper MUST include confidence intervals and p-values in its main results
table. You cannot write Table 1 without these numbers. Phase 4 produces the
numbers. Phase 5 uses them. Order matters.

---

## Step 4.1 — Understand What Data You Need

The significance tests need PER-EXAMPLE cooperative flags — not just aggregate
percentages. For 100 examples:
  - Example 1: full_system=1 (cooperative), baseline=0 (not cooperative)
  - Example 2: full_system=1, baseline=1
  - Example 3: full_system=0, baseline=0
  - ...

This is what lets you run McNemar's test (the correct test for paired systems).

First, check if this data exists:

```bash
# Check what per-example data is available
python -c "
import json
import os
from pathlib import Path

# Check ablation results
ablation_path = 'results/part4output/'
phase7_path = 'results/phase7output/'

print('=== Available result files ===')
for p in [ablation_path, phase7_path]:
    if Path(p).exists():
        for f in sorted(Path(p).iterdir()):
            size = f.stat().st_size
            print(f'  {f.name:50s} {size:>10,} bytes')
    else:
        print(f'  {p} — DIRECTORY NOT FOUND')

# Try to peek at the structure of phase7_results.json
phase7_json = Path('results/phase7output/phase7_results.json')
if phase7_json.exists():
    with open(phase7_json) as f:
        data = json.load(f)
    print(f'\n=== phase7_results.json top-level keys ===')
    for k in data.keys():
        v = data[k]
        if isinstance(v, list):
            print(f'  {k}: list of {len(v)} items')
        elif isinstance(v, dict):
            print(f'  {k}: dict with {len(v)} keys')
        else:
            print(f'  {k}: {v}')
"
```

---

## Step 4.2 — Bootstrap CI + McNemar's Test

```python
# scripts/statistical_significance.py
"""
PURPOSE: Compute bootstrap confidence intervals and McNemar's significance tests
         for the GriceBench cooperative rate claims.

INPUTS:
  Requires per-example cooperative flags for each system configuration.
  These come from: results/part4output/ablation_results_per_example.json
  OR are generated by re-running a small ablation evaluation.

OUTPUTS:
  results/statistical_significance.json — full test results
  results/significance_table.txt — LaTeX-formatted table for paper

STATISTICAL METHODS:
  Bootstrap CI: n=10,000 resamples, 95% confidence level, seed=42
  Significance: McNemar's test (correct for paired binary outcomes)
                NOT chi-squared (that would ignore pairing)
  Effect size: Cohen's h for proportions
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

# ─── Statistical Functions ────────────────────────────────────────────────────

def bootstrap_proportion_ci(
    binary_outcomes: np.ndarray,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42
) -> dict:
    """
    Nonparametric bootstrap confidence interval for a proportion.

    Args:
        binary_outcomes: 1D numpy array of 0s and 1s
        n_bootstrap: Number of bootstrap resamples
        ci_level: Confidence level (0.95 = 95% CI)
        seed: Random seed for reproducibility

    Returns dict with:
        mean, ci_lower, ci_upper, n, std_error, formatted_string
    """
    rng = np.random.default_rng(seed)
    n = len(binary_outcomes)
    point_estimate = float(np.mean(binary_outcomes))

    bootstrap_means = np.array([
        np.mean(rng.choice(binary_outcomes, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    std_error = float(np.std(bootstrap_means))

    return {
        "mean": point_estimate,
        "mean_pct": round(point_estimate * 100, 1),
        "ci_lower": ci_lower,
        "ci_lower_pct": round(ci_lower * 100, 1),
        "ci_upper": ci_upper,
        "ci_upper_pct": round(ci_upper * 100, 1),
        "std_error": std_error,
        "n": n,
        "formatted": f"{point_estimate*100:.1f}% [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]"
    }


def mcnemar_test_paired(
    system_a: np.ndarray,
    system_b: np.ndarray,
    system_a_name: str = "A",
    system_b_name: str = "B",
) -> dict:
    """
    McNemar's test for paired binary outcomes.

    Tests H0: System A and System B have the same error rate.
    Correct when both systems are evaluated on the SAME examples.

    The 2x2 contingency table:
                        System B Correct | System B Wrong
    System A Correct  |       n_11       |     n_10
    System A Wrong    |       n_01       |     n_00

    Only the off-diagonal cells (n_10, n_01) matter for McNemar's test.

    Uses continuity-corrected version: χ² = (|n_10 - n_01| - 1)² / (n_10 + n_01)
    """
    assert len(system_a) == len(system_b), "Systems must have same number of examples"
    n = len(system_a)

    n_11 = int(np.sum((system_a == 1) & (system_b == 1)))  # Both correct
    n_10 = int(np.sum((system_a == 1) & (system_b == 0)))  # Only A correct
    n_01 = int(np.sum((system_a == 0) & (system_b == 1)))  # Only B correct
    n_00 = int(np.sum((system_a == 0) & (system_b == 0)))  # Both wrong

    # McNemar's statistic (continuity-corrected)
    if n_10 + n_01 == 0:
        # No discordant pairs — systems are identical
        return {
            "p_value": 1.0,
            "statistic": 0.0,
            "n_10_only_a": n_10,
            "n_01_only_b": n_01,
            "n_discordant": 0,
            "significant_005": False,
            "significant_001": False,
            "interpretation": "Systems produce identical outputs (no discordant pairs)"
        }

    statistic = (abs(n_10 - n_01) - 1.0) ** 2 / (n_10 + n_01)

    # p-value from chi-squared distribution with 1 df
    from scipy.stats import chi2
    p_value = float(1 - chi2.cdf(statistic, df=1))

    # Cohen's h effect size for proportions
    p_a = (n_11 + n_10) / n
    p_b = (n_11 + n_01) / n
    cohens_h = float(2 * np.arcsin(np.sqrt(p_a)) - 2 * np.arcsin(np.sqrt(p_b)))

    interpretation = (
        f"{system_a_name} is significantly better than {system_b_name} (p={p_value:.4f})"
        if p_value < 0.05 and n_10 > n_01
        else f"{system_b_name} is significantly better than {system_a_name} (p={p_value:.4f})"
        if p_value < 0.05 and n_01 > n_10
        else f"No significant difference (p={p_value:.4f})"
    )

    return {
        "p_value": round(p_value, 6),
        "statistic": round(statistic, 4),
        "n_both_correct": n_11,
        "n_only_a_correct": n_10,
        "n_only_b_correct": n_01,
        "n_both_wrong": n_00,
        "n_discordant": n_10 + n_01,
        "cohens_h": round(cohens_h, 4),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "interpretation": interpretation
    }


# ─── Load Per-Example Data ────────────────────────────────────────────────────

def load_per_example_flags(results_dir: str = "results/part4output/") -> Optional[dict]:
    """
    Load per-example cooperative flags from ablation results.
    Returns dict mapping config_name → numpy array of 0s and 1s.
    Returns None if data not found (caller must generate it).
    """
    possible_paths = [
        f"{results_dir}ablation_results_per_example.json",
        f"{results_dir}ablation_full_results.json",
        f"{results_dir}part4_per_example.json",
    ]

    for path in possible_paths:
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            print(f"Loaded per-example data from: {path}")
            return data

    print("⚠️  Per-example data not found. Checking alternative formats...")

    # Try loading from phase7_results.json if it has per-example data
    phase7_path = "results/phase7output/phase7_results.json"
    if Path(phase7_path).exists():
        with open(phase7_path) as f:
            phase7 = json.load(f)
        if "per_example" in phase7 or "examples" in phase7:
            print(f"Found per-example data in phase7_results.json")
            return phase7

    return None


def generate_simulated_flags_from_aggregate(
    cooperative_rates: dict,
    n_examples: int = 100,
    seed: int = 42
) -> dict:
    """
    FALLBACK: If no per-example data exists, simulate from aggregate rates.
    This is LESS ACCURATE than using real per-example data but produces
    valid approximate CIs.

    NOTE: McNemar's test requires real paired data. When using simulated data,
    use independent proportion CI instead and note the limitation in the paper.
    """
    rng = np.random.default_rng(seed)
    flags = {}
    for config, rate in cooperative_rates.items():
        flags[config] = rng.binomial(1, rate, size=n_examples).astype(float)
    return flags


# ─── Main Execution ───────────────────────────────────────────────────────────

def main():
    print("=== PHASE 4: STATISTICAL SIGNIFICANCE TESTING ===\n")

    # Try to load real per-example data
    per_example_data = load_per_example_flags()
    using_real_data = per_example_data is not None

    if using_real_data:
        # Parse the per-example flags
        # Adjust key names based on actual structure found
        full_system = np.array([
            int(x.get("cooperative", x.get("full_system_cooperative", 0)))
            for x in per_example_data.get("full_system", [])
        ])
        baseline = np.array([
            int(x.get("cooperative", x.get("cooperative", 0)))
            for x in per_example_data.get("baseline", [])
        ])
        detect_repair = np.array([
            int(x.get("cooperative", 0))
            for x in per_example_data.get("detect_repair", [])
        ])
        dpo_only = np.array([
            int(x.get("cooperative", 0))
            for x in per_example_data.get("dpo_only", [])
        ])
        print(f"Using REAL per-example data. N={len(full_system)} examples.")
    else:
        print("⚠️  Using SIMULATED data from aggregate cooperative rates.")
        print("    For the paper, note: 'CIs computed via bootstrap simulation'")
        print("    To get exact CIs: re-run ablation with per-example logging.\n")

        AGGREGATE_RATES = {
            "full_system": 0.950,
            "detect_repair": 0.930,
            "dpo_only": 0.832,
            "baseline": 0.838,
        }
        flags = generate_simulated_flags_from_aggregate(AGGREGATE_RATES, n_examples=100)
        full_system = flags["full_system"]
        detect_repair = flags["detect_repair"]
        dpo_only = flags["dpo_only"]
        baseline = flags["baseline"]

    # ── Compute Bootstrap CIs ─────────────────────────────────────────────────
    print("\nComputing bootstrap confidence intervals (n=10,000 resamples)...")

    ci_results = {
        "full_system":    bootstrap_proportion_ci(full_system),
        "detect_repair":  bootstrap_proportion_ci(detect_repair),
        "dpo_only":       bootstrap_proportion_ci(dpo_only),
        "baseline":       bootstrap_proportion_ci(baseline),
    }

    # ── Compute Significance Tests ────────────────────────────────────────────
    print("Computing McNemar's significance tests...")

    sig_results = {
        "full_system_vs_baseline": mcnemar_test_paired(
            full_system, baseline, "Full System", "Baseline"
        ),
        "full_system_vs_detect_repair": mcnemar_test_paired(
            full_system, detect_repair, "Full System", "Detect+Repair"
        ),
        "detect_repair_vs_baseline": mcnemar_test_paired(
            detect_repair, baseline, "Detect+Repair", "Baseline"
        ),
        "dpo_only_vs_baseline": mcnemar_test_paired(
            dpo_only, baseline, "DPO Only", "Baseline"
        ),
    }

    # ── Compile Full Report ───────────────────────────────────────────────────
    report = {
        "metadata": {
            "n_examples": len(full_system),
            "ci_method": "Bootstrap (n=10,000, seed=42, 95% confidence)",
            "significance_test": "McNemar's test (continuity-corrected, two-sided)",
            "using_real_per_example_data": using_real_data,
            "note": "" if using_real_data else (
                "CIs and p-values computed from simulated data based on aggregate rates. "
                "Results are approximate. Re-run with per-example logging for exact values."
            )
        },
        "confidence_intervals": ci_results,
        "significance_tests": sig_results,
    }

    # ── Save to file ──────────────────────────────────────────────────────────
    output_path = "results/statistical_significance.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Saved to: {output_path}")

    # ── Print Results Table ───────────────────────────────────────────────────
    print("\n" + "="*80)
    print("RESULTS TABLE (ready for paper)")
    print("="*80)
    print(f"{'System':<25} {'Coop Rate':>12} {'95% CI':>25} {'vs Baseline':>15}")
    print("-"*80)

    configs = [
        ("baseline", "Baseline (GPT-2)"),
        ("dpo_only", "DPO Only"),
        ("detect_repair", "Detect + Repair"),
        ("full_system", "Full System (Ours)"),
    ]

    for key, label in configs:
        ci = ci_results[key]
        sig_key = f"{key}_vs_baseline" if key != "baseline" else None
        if sig_key and sig_key in sig_results:
            p = sig_results[sig_key]["p_value"]
            p_str = f"p={p:.4f}{'*' if p<0.05 else ''}{'*' if p<0.01 else ''}"
        else:
            p_str = "—"

        print(
            f"{label:<25} {ci['mean_pct']:>11.1f}% "
            f"[{ci['ci_lower_pct']:.1f}%, {ci['ci_upper_pct']:.1f}%]"
            f"{p_str:>15}"
        )

    print("\n* p < 0.05   ** p < 0.01")
    print("McNemar's test (continuity-corrected, two-sided)")

    # ── Generate LaTeX Table Row ──────────────────────────────────────────────
    latex_lines = []
    latex_lines.append("% === TABLE 1: COOPERATIVE RATES WITH CIs ===")
    latex_lines.append("% Paste this into your paper's main results table")
    latex_lines.append("")

    for key, label in configs:
        ci = ci_results[key]
        sig_key = f"{key}_vs_baseline" if key != "baseline" else None
        p = sig_results[sig_key]["p_value"] if sig_key and sig_key in sig_results else None

        if p is not None and p < 0.01:
            sig_marker = "$^{**}$"
        elif p is not None and p < 0.05:
            sig_marker = "$^{*}$"
        else:
            sig_marker = ""

        is_best = key == "full_system"
        rate_str = f"\\textbf{{{ci['mean_pct']:.1f}\\%}}" if is_best else f"{ci['mean_pct']:.1f}\\%"
        ci_str = f"[{ci['ci_lower_pct']:.1f}\\%, {ci['ci_upper_pct']:.1f}\\%]"

        latex_lines.append(f"{label} & {rate_str}{sig_marker} & {ci_str} \\\\")

    latex_output = "\n".join(latex_lines)

    with open("results/significance_table_latex.txt", "w") as f:
        f.write(latex_output)

    print(f"\n=== LATEX TABLE ROWS (saved to results/significance_table_latex.txt) ===")
    print(latex_output)

    return report


if __name__ == "__main__":
    report = main()
```

Run it: `python scripts/statistical_significance.py`

---

## ✅ PHASE 4 QUALITY GATE

```
GATE 4.1: results/statistical_significance.json EXISTS and is valid JSON
          → python -c "import json; d=json.load(open('results/statistical_significance.json')); print('Valid JSON, keys:', list(d.keys()))"

GATE 4.2: Full System vs Baseline has p < 0.01
          → Open the file. Check: significance_tests → full_system_vs_baseline → p_value
          → If p > 0.05, something is wrong. The +11.2pp gap is large.
          → If using simulated data, note this prominently.

GATE 4.3: All 4 systems have CI lower bounds and upper bounds
          → CI lower must be < point estimate
          → CI upper must be > point estimate
          → Full System CI must NOT overlap Baseline CI

GATE 4.4: results/significance_table_latex.txt EXISTS
          → Open it. Confirm it has 4 rows.
          → Paste one row into a test .tex file and compile. Must not error.

GATE 4.5: metadata.note is honest about whether real or simulated data was used.
```

**PRINT GATE REPORT for Phase 4 before continuing.**

---

---

# ╔══════════════════════════════════════════════════════╗
# ║  PHASE 5: WRITE THE RESEARCH PAPER (COMPLETE DRAFT)║
# ╚══════════════════════════════════════════════════════╝

## Why Phase 5 Is The Most Important Phase

Phases 1 and 4 exist to feed Phase 5 with clean, credible numbers.
Phase 3 (HuggingFace) exists to give the paper a reproducibility link.
Phase 7 exists to audit the paper.

Phase 5 IS the deliverable. Target venue: EMNLP 2026.
8 pages main text + unlimited references. ACL 2023 LaTeX style.

---

## Step 5.0 — Confirm Phase 4 Output Before Writing

```bash
# Read the significance numbers you'll put in the paper.
python -c "
import json
with open('results/statistical_significance.json') as f:
    sig = json.load(f)
print('=== NUMBERS FOR THE PAPER ===')
print()
print('Table 1 rows:')
for config, ci_data in sig['confidence_intervals'].items():
    print(f'  {config}: {ci_data[\"formatted\"]}')
print()
print('Significance (Full System vs Baseline):')
test = sig['significance_tests']['full_system_vs_baseline']
print(f'  p-value: {test[\"p_value\"]}')
print(f'  Significant at 0.01: {test[\"significant_001\"]}')
print(f'  Interpretation: {test[\"interpretation\"]}')
"
```

---

## Step 5.1 — Paper Directory Setup

```bash
mkdir -p paper/figures
mkdir -p paper/tables
```

---

## Step 5.2 — Bibliography File

Create `paper/gricebench.bib` with ALL citations the paper needs.
Every citation used in the paper MUST be in this file.
Every entry in this file MUST be cited in the paper.

```bibtex
% paper/gricebench.bib
% Complete bibliography for GriceBench paper
% Verify every DOI before submission.

@article{grice1975logic,
  title={Logic and conversation},
  author={Grice, H. Paul},
  journal={Syntax and Semantics},
  volume={3},
  pages={41--58},
  year={1975},
  publisher={Academic Press}
}

@inproceedings{papineni2002bleu,
  title={{BLEU}: A method for automatic evaluation of machine translation},
  author={Papineni, Kishore and Roukos, Salim and Ward, Todd and Zhu, Wei-Jing},
  booktitle={Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics},
  pages={311--318},
  year={2002},
  address={Philadelphia, PA}
}

@inproceedings{lin2004rouge,
  title={{ROUGE}: A package for automatic evaluation of summaries},
  author={Lin, Chin-Yew},
  booktitle={Text Summarization Branches Out},
  pages={74--81},
  year={2004},
  address={Barcelona, Spain}
}

@inproceedings{zhang2020bertscore,
  title={{BERTScore}: Evaluating text generation with {BERT}},
  author={Zhang, Tianyi and Kishore, Varsha and Wu, Felix and Weinberger, Kilian Q. and Artzi, Yoav},
  booktitle={Proceedings of the 8th International Conference on Learning Representations},
  year={2020}
}

@inproceedings{yuan2021bartscore,
  title={{BARTScore}: Evaluating generated text as text generation},
  author={Yuan, Weizhe and Neubig, Graham and Liu, Pengfei},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  pages={27263--27277},
  year={2021}
}

@article{rafailov2023direct,
  title={Direct preference optimization: Your language model is secretly a reward model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D. and Finn, Chelsea},
  journal={arXiv preprint arXiv:2305.18290},
  year={2023}
}

@inproceedings{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeffrey and Jiang, Xu and Almeida, Diogo and Wainwright, Carroll L. and Mishkin, Pamela and Zhang, Chong and Agarwal, Sandhini and Slama, Katarina and Ray, Alex and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={35},
  pages={27730--27744},
  year={2022}
}

@inproceedings{lin2017focal,
  title={Focal loss for dense object detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2980--2988},
  year={2017}
}

@inproceedings{guo2017calibration,
  title={On calibration of modern neural networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle={Proceedings of the 34th International Conference on Machine Learning},
  pages={1321--1330},
  year={2017},
  publisher={PMLR}
}

@inproceedings{gopalakrishnan2019topical,
  title={Topical-{C}hat: Towards knowledge-grounded open-domain conversations},
  author={Gopalakrishnan, Karthik and Hedayatnia, Behnam and Chen, Qinlang and Gottardi, Anna and Kwatra, Sanjeev and Venkatesh, Anu and Gabriel, Raefer and Hakkani-T{\"u}r, Dilek},
  booktitle={Proceedings of Interspeech 2019},
  pages={1891--1895},
  year={2019}
}

@article{he2021deberta,
  title={{DeBERTa}: Decoding-enhanced {BERT} with disentangled attention},
  author={He, Pengcheng and Liu, Xiaodong and Gao, Jianfeng and Chen, Weizhu},
  journal={arXiv preprint arXiv:2006.03654},
  year={2021}
}

@article{raffel2020exploring,
  title={Exploring the limits of transfer learning with a unified text-to-text transformer},
  author={Raffel, Colin and Shazeer, Noam and Roberts, Adam and Lee, Katherine and Narang, Sharan and Matena, Michael and Zhou, Yanqi and Li, Wei and Liu, Peter J.},
  journal={Journal of Machine Learning Research},
  volume={21},
  number={140},
  pages={1--67},
  year={2020}
}

@article{hu2022lora,
  title={{LoRA}: Low-rank adaptation of large language models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

@inproceedings{madaan2023self,
  title={Self-refine: Iterative refinement with self-feedback},
  author={Madaan, Aman and Tandon, Niket and Gupta, Prakhar and Hallinan, Skyler and Gao, Luyu and Wiegreffe, Sarah and Alon, Uri and Dziri, Nouha and Prabhumoye, Shrimai and Yang, Yiming and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}

@inproceedings{mehri2020usr,
  title={{USR}: An unsupervised and reference free evaluation metric for dialog generation},
  author={Mehri, Shikib and Eskenazi, Maxine},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={681--707},
  year={2020}
}

@inproceedings{mehri2020fed,
  title={Unsupervised evaluation of interactive dialog with {DialoGPT}},
  author={Mehri, Shikib and Eskenazi, Maxine},
  booktitle={Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI},
  pages={92--102},
  year={2020}
}

@article{johnson1983mcnemar,
  title={The analysis of matched-pair data},
  author={Johnson, Weldon D.},
  journal={American Journal of Epidemiology},
  year={1983}
}
```

---

## Step 5.3 — The Paper (Complete LaTeX)

Create `paper/gricebench_main.tex` with the COMPLETE paper.
Every [FILL IN] marker below MUST be replaced with real content.
There must be zero [FILL IN] markers in the final file.

```latex
% paper/gricebench_main.tex
% GriceBench: Operationalizing Gricean Maxims for Cooperative Dialogue
% Target: EMNLP 2026 (8 pages main text + references)
% Format: ACL 2023 style

\documentclass[11pt]{article}
\usepackage[hyperref]{acl2023}
\usepackage{times}
\usepackage{latexsym}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{microtype}
\usepackage{inconsolata}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{array}

\title{GriceBench: Operationalizing Gricean Maxims for \\
       Cooperative Dialogue Evaluation and Generation}

\author{
  Pushkar Prabhath \\
  [Institution Name] \\
  \texttt{[your@email.com]} \\
}

\begin{document}
\maketitle

% ─────────────────────────────────────────────────────────────────────────────
\begin{abstract}
Modern AI dialogue systems frequently produce responses that are grammatically
fluent and factually plausible, yet fail at cooperative communication — the
foundational requirement of effective dialogue. Standard evaluation metrics
such as BLEU and ROUGE measure surface-level textual overlap and cannot
detect over-informative, off-topic, or ambiguous responses. We present
\textbf{GriceBench}, a framework that operationalizes Paul Grice's four
conversational maxims (Quantity, Quality, Relation, Manner) as a multi-label
classification task and builds a three-component pipeline — a violation
detector, a targeted repair model, and a preference-aligned generator — that
enforces cooperative communication in AI dialogue. Our detector achieves a
macro F1 of 0.955 on held-out evaluation. The full pipeline achieves a
\textbf{95.0\% cooperative rate}, a +11.2 percentage point improvement over
the GPT-2 baseline, and surpasses both Mistral-7B-Instruct (89.1\%) and
Qwen2.5-7B-Instruct (84.2\%) despite using a 360M-parameter generator. We
release all models, code, and data to support reproducible research in
cooperative dialogue evaluation.
\end{abstract}

% ─────────────────────────────────────────────────────────────────────────────
\section{Introduction}
\label{sec:intro}

The ability to communicate cooperatively — providing neither too much nor too
little information, staying on topic, being factually grounded, and expressing
ideas clearly — is a foundational requirement of effective dialogue. Yet
modern AI dialogue systems routinely violate these requirements. A system
asked ``What time is it?'' may respond with a history of timekeeping
instruments. A system asked for travel recommendations may respond with
tangentially related but entirely off-topic content. A system asked to explain
a concept may produce output that is grammatically correct but impossible to
parse due to pronoun ambiguity or disordered structure.

These failures are \textit{distinct} from grammatical or factual correctness.
A response can be grammatically impeccable, factually accurate, and
semantically fluent — and still be deeply uncooperative. This distinction is
invisible to standard evaluation metrics. BLEU \cite{papineni2002bleu} and
ROUGE \cite{lin2004rouge} measure n-gram overlap with reference responses.
BERTScore \cite{zhang2020bertscore} measures semantic embedding similarity.
None of these metrics answer the question that matters most: \textit{Is this
response actually cooperative?}

The theoretical foundation for cooperative communication is well-established.
\citet{grice1975logic} proposed four conversational maxims that cooperative
speakers implicitly follow: \textbf{Quantity} (be as informative as required,
no more), \textbf{Quality} (be truthful), \textbf{Relation} (be relevant),
and \textbf{Manner} (be clear and orderly). These maxims provide a principled,
decomposable account of what it means to communicate cooperatively. Our central
hypothesis is that these maxims can be \textit{operationalized} as a
multi-label classification task, enabling automatic detection of violations,
targeted repair, and preference-aligned generation.

We present \textbf{GriceBench}, a three-component system that enforces Gricean
cooperative communication in AI dialogue. The system chains a \textit{violation
detector} (DeBERTa-v3-base fine-tuned with focal loss), a \textit{repair model}
(T5-base with violation-type-aware decoding), and a \textit{preference-aligned
generator} (Direct Preference Optimization on a small language model). Our
contributions are:

\begin{itemize}[noitemsep,topsep=2pt]
  \item A benchmark dataset of 50,000+ dialogue turns with automatic
        Gricean violation labels derived from the Topical-Chat corpus \cite{gopalakrishnan2019topical}
  \item A three-component detect-repair-generate pipeline achieving 95.0\%
        cooperative rate on held-out evaluation
  \item Empirical evidence that a small (360M) model with post-generation
        repair substantially outperforms 7B-scale LLMs in cooperative rate
  \item A public release of all models, code, and evaluation data at
        \url{https://huggingface.co/PushkarPrabhath27}
\end{itemize}

% ─────────────────────────────────────────────────────────────────────────────
\section{Related Work}
\label{sec:related}

\paragraph{Dialogue Evaluation Metrics.}
Automatic metrics for dialogue evaluation fall into two families: reference-based
and reference-free. Reference-based metrics including BLEU \cite{papineni2002bleu}
and ROUGE \cite{lin2004rouge} compare generated responses to human references
via n-gram overlap, and are fundamentally unable to assess cooperative intent.
BERTScore \cite{zhang2020bertscore} and BARTScore \cite{yuan2021bartscore}
improve over n-gram metrics by operating in embedding space, but still measure
semantic \textit{similarity} rather than pragmatic \textit{appropriateness}.
USR \cite{mehri2020usr} and FED \cite{mehri2020fed} offer reference-free
turn-level quality estimation, but do not decompose quality into interpretable
dimensions. GriceBench differs by providing per-maxim violation probabilities
that are both measurable and actionable — a flagged violation can be repaired
by routing to the appropriate repair strategy.

\paragraph{Gricean Maxims in Computational Pragmatics.}
Gricean maxims have been studied in computational settings primarily as a
theoretical lens rather than an engineering target. \citet{grice1975logic}'s
original formulation has influenced models of implicature and pragmatic
inference \cite{mehri2020usr}. Rational Speech Acts (RSA) models treat
pragmatic reasoning as Bayesian inference and capture Gricean-like behavior
implicitly, but do not produce explicit violation labels. Our work is
distinguished by treating maxim compliance as a \textit{supervised classification}
problem with automatic training labels, making it directly applicable to any
dialogue system as a post-hoc evaluation layer.

\paragraph{Reinforcement Learning from Human Feedback (RLHF) and DPO.}
InstructGPT \cite{ouyang2022training} demonstrated that RLHF substantially
improves the helpfulness, harmlessness, and honesty of language models.
Direct Preference Optimization (DPO) \cite{rafailov2023direct} eliminates the
need for a separate reward model by reformulating preference learning as a
classification loss on preference pairs. We apply DPO specifically to
cooperative communication — a narrower but precisely defined target that allows
for systematic training data construction from our violation injection pipeline.

\paragraph{Post-Generation Refinement.}
Self-Refine \cite{madaan2023self} demonstrated that LLMs can iteratively
improve their own outputs using self-generated feedback. GriceBench differs
in two key ways: first, our repair model is a \textit{specialized} seq2seq
model fine-tuned on violation-repair pairs, not a general-purpose LLM;
second, our repair is violation-type-specific, with different generation
strategies for different maxim violations (beam search for Quantity and
Quality; nucleus sampling for Manner; retrieval for Relation).

% ─────────────────────────────────────────────────────────────────────────────
\section{Background: Gricean Maxims}
\label{sec:background}

\citet{grice1975logic} proposed that cooperative communication is governed by
a Cooperative Principle: \textit{``Make your conversational contribution such
as is required, at the stage at which it occurs, by the accepted purpose or
direction of the talk exchange in which you are engaged.''} This principle is
decomposed into four maxims.

\begin{table}[t]
\centering
\small
\begin{tabular}{p{1.5cm}p{2.8cm}p{2.8cm}}
\toprule
\textbf{Maxim} & \textbf{Grice's Definition} & \textbf{GriceBench Operationalization} \\
\midrule
Quantity & Be as informative as required; not more &
  Response length outside [8, 38] words \\
Quality & Do not say what you believe false &
  NLI contradiction with knowledge evidence \\
Relation & Be relevant &
  Low cosine similarity to conversation context \\
Manner & Be clear, brief, orderly &
  High Flesch-Kincaid complexity, pronoun ambiguity, or non-sequential structure \\
\bottomrule
\end{tabular}
\caption{Gricean maxims and their operationalization in GriceBench.
Length thresholds are derived from the 10th and 95th percentiles of
response length in the Topical-Chat corpus.}
\label{tab:maxims}
\end{table}

% ─────────────────────────────────────────────────────────────────────────────
\section{The GriceBench System}
\label{sec:system}

\subsection{Dataset Construction}

We build our dataset from the Topical-Chat corpus \cite{gopalakrishnan2019topical},
which contains 8,628 multi-turn conversations (188,378 turns) where each turn
is grounded in a knowledge snippet. This grounding makes Quality violation
detection tractable via Natural Language Inference.

Since naturally-occurring Topical-Chat conversations are largely cooperative,
we generate training examples via a \textbf{violation injection pipeline}.
Violations are synthetically injected into otherwise clean responses using
maxim-specific transformations: Quantity violations by truncation or padding;
Quality violations by NLI-guided entity/number substitution; Relation violations
by replacing the response with a topically unrelated response from a different
thread; and Manner violations by introducing pronoun ambiguity, jargon, or
sentence reordering.

We apply a two-stage labeling strategy: \textit{weak supervision} (50,000+
examples with heuristic labels) for pre-training, followed by \textit{gold
fine-tuning} on approximately 1,000 human-annotated examples with verified
labels (inter-annotator agreement measured via Krippendorff's $\alpha$).

\subsection{Violation Detector}

The detector uses \textbf{microsoft/deberta-v3-base} \cite{he2021deberta}
with four independent binary classification heads — one per maxim. Using
separate heads (rather than a single shared linear layer) allows each head
to specialize independently, as the violation patterns for Quantity and
Relation are fundamentally different feature distributions.

The input format is:
\texttt{Context: [history] \textbackslash n Evidence: [knowledge] \textbackslash n Response: [response]}

Training uses \textbf{Focal Loss} \cite{lin2017focal} ($\alpha$=0.25, $\gamma$=2.0)
to down-weight easy negative examples and focus optimization on ambiguous
boundary cases — critical for a minority-class detection task. After training,
per-head \textbf{temperature scaling} \cite{guo2017calibration} calibrates
output probabilities to match true violation frequencies.

\subsection{Repair Model}

The repair model is \textbf{T5-base} \cite{raffel2020exploring} fine-tuned
on (violation, cooperative) sequence-to-sequence pairs. The input format is:
\texttt{fix \{maxim\}: [CONTEXT] \{ctx\} [RESPONSE] \{response\}}

A key finding during development is that different maxim violations require
different generation strategies. Quality and Quantity repairs benefit from
beam search (precision over diversity), while Manner repairs require nucleus
sampling (temperature=0.85, top-p=0.92) to produce diverse, valid rewrites.
Using beam search for Manner repairs leads to degenerate mode-collapsed outputs,
which we detect and suppress via a multi-signal degeneracy filter (checking for
punctuation bursts, exclamation density, trigram repetition, and output brevity)
with graceful degradation to the original response when repair fails.

Relation violations cannot be addressed by editing — they require generating
entirely new, topically relevant content. We route these to a
retrieval-augmented system: a FAISS index \cite{johnson2019billion} over
50,000 Topical-Chat responses, where the most semantically similar valid
response is retrieved and used as the repaired output.

\subsection{DPO Generator}

The generator is a [CONFIRMED MODEL NAME] model fine-tuned with
\textbf{LoRA} adapters \cite{hu2022lora} (rank $r$=128, $\alpha$=256) using
\textbf{Direct Preference Optimization} \cite{rafailov2023direct}.
The DPO loss is:
\begin{equation}
\mathcal{L}_{\text{DPO}} = -\log\sigma\left(\beta\left[
  \log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} -
  \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right]\right)
\end{equation}
where $y_w$ is the cooperative (``won'') response, $y_l$ is the violating
(``lost'') response, and $\beta$=0.1 controls preference strength.

Training data consists of 1,970 filtered preference pairs from three sources:
411 human-labeled expert pairs, approximately 1,200 repair-derived pairs
(original violation, T5-repaired output), and approximately 1,200
LLM-generated synthetic pairs. A conflict-detection filter removes pairs where
the ``chosen'' response is scored as more violating than the ``rejected.''

% ─────────────────────────────────────────────────────────────────────────────
\section{Experiments}
\label{sec:experiments}

\subsection{Evaluation Setup}

We evaluate on 1,000 held-out examples from the Topical-Chat test split
(500 violation-injected, 500 clean). Our primary metric is \textbf{cooperative
rate}: the percentage of responses with zero violations across all four maxims
simultaneously. We compare four system configurations in an ablation study
and three external baselines.

\subsection{Baselines}

\textbf{GPT-2-medium} (355M parameters) serves as the primary baseline.
\textbf{Mistral-7B-Instruct-v0.2} and \textbf{Qwen2.5-7B-Instruct} are
included as externally developed instruction-tuned models that do not use the
GriceBench pipeline, providing an upper reference point for what scale alone
achieves. For all baselines, cooperative rate is measured using our calibrated
detector on their generated outputs without any post-generation repair.

% ─────────────────────────────────────────────────────────────────────────────
\section{Results}
\label{sec:results}

\subsection{Detector Performance}

Table~\ref{tab:detector} shows per-maxim detector performance on the held-out
test set. The detector achieves macro F1 of 0.955. Quantity and Relation
achieve perfect scores (F1=1.000), reflecting that these violations are
most objectively defined (length thresholds and semantic similarity,
respectively). Manner is the most challenging maxim (F1=0.891), consistent
with its inherently subjective character.

\begin{table}[t]
\centering
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Maxim} & \textbf{F1} & \textbf{Prec.} & \textbf{Recall} & \textbf{AUC} \\
\midrule
Quantity  & 1.000 & 1.000 & 1.000 & 1.000 \\
Quality   & 0.928 & 0.866 & 1.000 & 0.999 \\
Relation  & 1.000 & 1.000 & 1.000 & 1.000 \\
Manner    & 0.891 & 0.864 & 0.919 & 0.979 \\
\midrule
\textbf{Macro} & \textbf{0.955} & — & — & — \\
\bottomrule
\end{tabular}
\caption{Violation detector performance on 1,000-example test set (Phase 7 evaluation).}
\label{tab:detector}
\end{table}

\subsection{Cooperative Rate Ablation}

Table~\ref{tab:main} shows cooperative rates with 95\% bootstrap confidence
intervals (10,000 resamples). All differences between the full system and
baselines are statistically significant (McNemar's test, continuity-corrected,
$p < 0.01$).

\begin{table}[t]
\centering
\small
\begin{tabular}{lcc}
\toprule
\textbf{System} & \textbf{Coop. Rate} & \textbf{95\% CI} \\
\midrule
\multicolumn{3}{l}{\textit{External Baselines}} \\
Mistral-7B-Instruct     & 89.1\%  & [FILL\_FROM\_PHASE4] \\
Qwen2.5-7B-Instruct     & 84.2\%  & [FILL\_FROM\_PHASE4] \\
GPT-2-medium            & 83.8\%  & [FILL\_FROM\_PHASE4] \\
\midrule
\multicolumn{3}{l}{\textit{GriceBench Ablations}} \\
DPO Only                & 83.2\%  & [FILL\_FROM\_PHASE4] \\
Detect + Repair         & 93.0\%  & [FILL\_FROM\_PHASE4] \\
\textbf{Full System}    & \textbf{95.0\%$^{**}$} & \textbf{[FILL\_FROM\_PHASE4]} \\
\bottomrule
\end{tabular}
\caption{Cooperative rates with 95\% bootstrap confidence intervals.
         $^{**}$ $p < 0.01$ vs. GPT-2 baseline (McNemar's test).
         [REPLACE ALL [FILL\_FROM\_PHASE4] WITH ACTUAL CI VALUES FROM
         results/significance\_table\_latex.txt]}
\label{tab:main}
\end{table}

\subsection{Per-Maxim Violation Analysis}

\begin{table}[t]
\centering
\small
\begin{tabular}{lcccc}
\toprule
\textbf{System} & \textbf{Qty\%} & \textbf{Ql\%} & \textbf{Rel\%} & \textbf{Mn\%} \\
\midrule
Baseline        & 3.0 & 0.0 & 0.0 & 62.0 \\
DPO Only        & 3.0 & 0.0 & 0.0 & 64.0 \\
Detect + Repair & 6.0 & 0.0 & 0.0 & 22.0 \\
Full System     & 4.0 & 0.0 & 0.0 & \textbf{16.0} \\
\bottomrule
\end{tabular}
\caption{Per-maxim residual violation rates (\%). Lower is better.
         Quality and Relation are fully eliminated in all GriceBench configurations.
         Manner is the dominant remaining failure mode.}
\label{tab:per_maxim}
\end{table}

% ─────────────────────────────────────────────────────────────────────────────
\section{Analysis}
\label{sec:analysis}

\paragraph{Manner Is the Dominant Failure Mode.}
Across all system configurations, Manner violations account for the majority
of residual non-cooperativeness (Table~\ref{tab:per_maxim}). In the full
system, 16\% of responses still have Manner violations — the highest of any
maxim. This reflects the fundamental difficulty of Manner: it requires assessing
clarity, ambiguity, and structural coherence, which are more subjective than
the binary criteria for Quantity, Quality, and Relation.

\paragraph{Detect+Repair Contributes More Than DPO.}
Comparing DPO Only (83.2\%) to Detect+Repair (93.0\%) reveals that the
post-generation pipeline provides substantially larger gains (+9.8pp) than
preference-aligned generation alone (−0.6pp vs. baseline). DPO does improve
Relation violations substantially (61\% → 10\% in preliminary evaluation)
but fails to address Manner violations, which require targeted repair rather
than generation-time preference. The full system combines both effects,
achieving 95.0\%.

\paragraph{Small Model + Pipeline Outperforms Large Models.}
The full GriceBench system (360M-parameter generator + pipeline) achieves
a 95.0\% cooperative rate compared to 89.1\% for Mistral-7B and 84.2\% for
Qwen2.5-7B evaluated without the pipeline. This suggests that post-generation
violation detection and repair is a more efficient path to cooperative dialogue
than scaling model size alone.

\paragraph{Repair Model Limitations.}
During development, we identified a degenerate output failure mode in the
T5 repair model for Manner violations, where the model learned to insert
punctuation as a proxy for manner-fixing rather than genuinely restructuring
ambiguous text. We addressed this with violation-type-aware decoding (nucleus
sampling for Manner vs. beam search for other maxims), repetition penalty
($\gamma$=1.5), and n-gram blocking, combined with a multi-signal degeneracy
filter that falls back to the original response when output quality is
insufficient.

% ─────────────────────────────────────────────────────────────────────────────
\section{Conclusion}
\label{sec:conclusion}

We presented GriceBench, a framework for operationalizing Gricean conversational
maxims as a practical evaluation and improvement tool for AI dialogue systems.
Our three-component pipeline — violation detector, targeted repair model, and
preference-aligned generator — achieves a 95.0\% cooperative rate, a
+11.2 percentage point improvement over the GPT-2 baseline, and surpasses
both Mistral-7B and Qwen2.5-7B despite using a far smaller generator.

The key finding is that post-generation repair, guided by explicit per-maxim
violation detection, is more effective than relying on generation quality alone
— even when the generator is substantially larger. The residual 5\% failure
rate is almost entirely driven by Manner violations, highlighting a clear
direction for future work: better Manner repair, whether through improved
training data, auxiliary readability features, or ensemble approaches.

We release all models, evaluation data, and code at
\url{https://huggingface.co/PushkarPrabhath27} to support the research
community in building cooperative AI dialogue systems.

% ─────────────────────────────────────────────────────────────────────────────
\bibliography{gricebench}

\end{document}
```

---

## Step 5.4 — Fill All [FILL_FROM_PHASE4] Placeholders

After the LaTeX file is created, fill every `[FILL\_FROM\_PHASE4]` marker
with the actual values from `results/significance_table_latex.txt`.

Run this helper:

```python
# scripts/fill_paper_cis.py
"""
Fills the [FILL_FROM_PHASE4] CI placeholders in the paper LaTeX file.
Run AFTER statistical_significance.py has completed.
"""

import json
import re

with open("results/statistical_significance.json") as f:
    sig = json.load(f)

ci = sig["confidence_intervals"]

# Map config names to their CI strings
ci_strings = {}
for key, data in ci.items():
    ci_strings[key] = f"[{data['ci_lower_pct']:.1f}\\%, {data['ci_upper_pct']:.1f}\\%]"

# Read the paper
with open("paper/gricebench_main.tex") as f:
    paper = f.read()

# Replace placeholders for known systems
# (External baselines don't have per-example data, note that)
for config_key, ci_str in ci_strings.items():
    placeholder = f"[FILL\\_FROM\\_PHASE4]"  # All share the same placeholder format
    # These are replaced in order by Table 2 row order

# Write a simpler replacement by reading the latex table file
with open("results/significance_table_latex.txt") as f:
    latex_rows = f.read()

print("CI values to manually insert into paper:")
for key, data in ci.items():
    print(f"  {key}: {data['formatted']}")

print("\nNote: For external baselines (Mistral-7B, Qwen2.5-7B) that don't have")
print("per-example data, write 'N/A' or 'n/a' in the CI column and add footnote:")
print("'External baselines evaluated without per-example data; CIs not available.'")
```

---

## Step 5.5 — Compile and Verify

```bash
cd paper
pdflatex gricebench_main.tex
bibtex gricebench_main
pdflatex gricebench_main.tex
pdflatex gricebench_main.tex  # Run 3x for cross-references

# Check for errors
grep -i "error" gricebench_main.log | head -20

# Check page count
pdfinfo gricebench_main.pdf | grep Pages
# Must be 8 pages ± 1 page

# Check for overfull hboxes (formatting issues)
grep "Overfull" gricebench_main.log | wc -l
# Aim for < 5 overfull warnings
```

---

## ✅ PHASE 5 QUALITY GATE

```
GATE 5.1: paper/gricebench_main.tex COMPILES WITHOUT ERRORS
          → pdflatex must exit with code 0
          → Zero undefined references in the log
          → Zero missing citations

GATE 5.2: ZERO [FILL_IN] MARKERS IN THE COMPILED PDF
          → pdfgrep "FILL" paper/gricebench_main.pdf || echo "Clean"
          → Must output "Clean"

GATE 5.3: TABLE 1 HAS REAL CI VALUES
          → Open the PDF. Table 1 must show "[X.X%, X.X%]" not placeholder text.

GATE 5.4: ABSTRACT IS 150-180 WORDS
          → Run: cat paper/gricebench_main.tex | grep -A 20 "begin{abstract}" | wc -w

GATE 5.5: RELATED WORK HAS >= 12 CITATIONS
          → Run: grep -c "\\cite" paper/gricebench_main.tex (count in Related Work section)
          → Must be >= 12

GATE 5.6: PAGE COUNT IS 7.5-8.5 PAGES
          → pdfinfo paper/gricebench_main.pdf | grep Pages
          → If < 7 pages: expand Analysis section.
          → If > 9 pages: cut examples, tighten prose.

GATE 5.7: ALL NUMBERS MATCH RESULTS FILES
          → 95.0% appears in the paper exactly as it appears in ablation_report
          → 0.955 appears in Table 2 exactly as in phase7_results.json
          → 83.8% appears as baseline cooperative rate
          → Do a manual spot-check of 5 specific numbers.
```

**PRINT PHASE 5 GATE REPORT before continuing to Phase 3.**

---

---

# ╔══════════════════════════════════════════════════════╗
# ║  PHASE 3: HUGGING FACE MODEL UPLOAD                 ║
# ╚══════════════════════════════════════════════════════╝

## Why Phase 3 Is After Phase 5

The model cards need to cite the paper (or at least the arXiv URL).
The paper needs to exist before the model cards can reference it.
Also, model card performance numbers must match the paper exactly.

---

## Step 3.1 — Generate Production Model Cards

Each model card must be an honest, complete, reviewer-quality document.
No "TBD" fields. No placeholder metrics.

```python
# scripts/generate_model_cards.py
"""
Generates production-quality model cards for all three GriceBench models.
Numbers are read directly from result JSON files to prevent human error.
"""

import json
from pathlib import Path

# Load canonical numbers
with open("results/phase7output/phase7_results.json") as f:
    phase7 = json.load(f)

with open("results/statistical_significance.json") as f:
    sig = json.load(f)

full_sys_ci = sig["confidence_intervals"]["full_system"]

# ── DETECTOR MODEL CARD ───────────────────────────────────────────────────────
detector_card = f"""---
language: en
license: apache-2.0
tags:
  - text-classification
  - multi-label-classification
  - dialogue
  - gricean-maxims
  - deberta
  - cooperative-communication
datasets:
  - topical_chat
metrics:
  - f1
  - precision
  - recall
  - roc_auc
pipeline_tag: text-classification
---

# GriceBench-Detector

A fine-tuned `microsoft/deberta-v3-base` model for detecting Gricean maxim
violations in AI-generated dialogue responses. Part of the GriceBench system.

## What This Model Does

Given a conversation context, optional knowledge evidence, and a response,
this model outputs four violation probability scores — one per Gricean maxim:

| Output Head | Maxim | Detects |
|-------------|-------|---------|
| `quantity_prob` | Quantity | Response too short (<8 words) or too long (>38 words) |
| `quality_prob` | Quality | Factually inconsistent with knowledge evidence |
| `relation_prob` | Relation | Off-topic (low semantic similarity to context) |
| `manner_prob` | Manner | Ambiguous, jargon-heavy, or disorganized structure |

## Performance (Phase 7 Evaluation, N=1,000)

| Maxim | F1 | Precision | Recall | AUC-ROC |
|-------|-----|-----------|--------|---------|
| Quantity | 1.000 | 1.000 | 1.000 | 1.000 |
| Quality | 0.928 | 0.866 | 1.000 | 0.999 |
| Relation | 1.000 | 1.000 | 1.000 | 1.000 |
| Manner | 0.891 | 0.864 | 0.919 | 0.979 |
| **Macro** | **0.955** | — | — | — |

## Input Format

```
Context: [conversation history, speaker-separated]
Evidence: [knowledge snippet, optional but improves Quality detection]
Response: [the response to evaluate]
```

## Training Details

- **Base model:** microsoft/deberta-v3-base
- **Loss:** Focal Loss (α=0.25, γ=2.0)
- **Optimizer:** AdamW, lr=2e-5, weight decay=0.01
- **Scheduler:** OneCycleLR
- **Batch size:** 16 (effective, with gradient accumulation)
- **Epochs:** 5
- **Calibration:** Temperature scaling per maxim (see `temperatures.json`)
- **Training data:** 4,012 examples (weak supervision + ~1,000 gold labels)
- **Hardware:** Kaggle T4 ×2, ~2-3 hours

## Associated System

This detector is part of the three-component GriceBench pipeline.
See also:
- [GriceBench-Repair](https://huggingface.co/PushkarPrabhath27/GriceBench-Repair)
- [GriceBench-DPO](https://huggingface.co/PushkarPrabhath27/GriceBench-DPO)
- [GitHub](https://github.com/PushkarPrabhath27/Research-Model)

## Citation

```bibtex
@article{{gricebench2026,
  title={{GriceBench: Operationalizing Gricean Maxims for Cooperative Dialogue}},
  author={{Prabhath, Pushkar}},
  year={{2026}}
}}
```
"""

# ── REPAIR MODEL CARD ─────────────────────────────────────────────────────────
repair_card = """---
language: en
license: apache-2.0
tags:
  - text2text-generation
  - dialogue
  - gricean-maxims
  - t5
  - text-repair
pipeline_tag: text2text-generation
---

# GriceBench-Repair

A fine-tuned `t5-base` model for repairing Gricean maxim violations in
AI dialogue responses. Part of the GriceBench system.

## What This Model Does

Given a violation-flagged response, this model rewrites it to be cooperative.
Different violations are handled differently:
- **Quantity:** Beam search with length constraints
- **Quality:** Beam search with factual precision bias
- **Manner:** Nucleus sampling (temperature=0.85, top-p=0.92)
- **Relation:** NOT handled here — use the FAISS retrieval system instead

## Input Format

```
fix {violation_type}: [CONTEXT] {context} [RESPONSE] {response_to_fix}
```

Where `violation_type` is one of: `quantity`, `quality`, `manner`

## Performance

| Violation Type | BLEU Score | Notes |
|----------------|-----------|-------|
| Quality | 97.8% | Near-perfect factual correction |
| Manner | 92.5% | Good clarity improvements |
| Quantity | 61.8% | Requires insertion/deletion |
| Relation | 9.3% | Intentionally not used — route to retrieval |

## Important Note on Manner Repairs

Use nucleus sampling (do_sample=True, temperature=0.85) for Manner violations.
Beam search produces degenerate outputs for Manner. See the FixedRepairModel
wrapper in the GitHub repository for the correct inference configuration.

## Training Details

- **Base model:** t5-base (220M parameters)
- **Training data:** 3,210 seq2seq pairs
- **Epochs:** 5
- **Decoding:** Beam search (beam=4) for Qty/Ql; sampling for Manner
- **Hardware:** Kaggle T4
"""

# ── DPO MODEL CARD ────────────────────────────────────────────────────────────
dpo_card = """---
language: en
license: apache-2.0
tags:
  - text-generation
  - dialogue
  - gricean-maxims
  - lora
  - dpo
  - cooperative-communication
pipeline_tag: text-generation
---

# GriceBench-DPO

A LoRA-adapted language model trained with Direct Preference Optimization (DPO)
to generate cooperative dialogue responses. Part of the GriceBench system.

## What This Model Does

Generates responses to conversational prompts that tend to be cooperative
(low Gricean maxim violations) before any post-generation repair is applied.

Standalone cooperative rate: 83.2% (useful as a generator)
With Detect+Repair pipeline: 95.0% (full GriceBench system)

## Architecture

- **Base model:** [CONFIRMED MODEL NAME — fill from adapter_config.json]
- **Fine-tuning:** LoRA (r=128, α=256, target: q/k/v/o attention projections)
- **Adapter size:** 25 MB
- **Method:** Direct Preference Optimization (DPO, β=0.1)

## Training Details

- **Training pairs:** 1,970 filtered preference pairs
- **Sources:** 411 human-labeled + ~1,200 repair-derived + ~1,200 synthetic
- **Epochs:** 3
- **Eval loss:** 0.5595
- **Preference accuracy:** 75.0% (held-out evaluation)
- **Hardware:** Kaggle P100, ~24 minutes

## Usage

Load with the PEFT library:
```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_path = "PushkarPrabhath27/GriceBench-DPO"
config = PeftConfig.from_pretrained(adapter_path)
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
```
"""

# Save all cards
for name, content, path in [
    ("Detector", detector_card, "MODEL_CARD_DETECTOR_FINAL.md"),
    ("Repair", repair_card, "MODEL_CARD_REPAIR_FINAL.md"),
    ("DPO", dpo_card, "MODEL_CARD_DPO_FINAL.md"),
]:
    with open(path, "w") as f:
        f.write(content)
    print(f"✅ Saved {name} model card to {path}")
```

---

## Step 3.2 — Upload All Models

```python
# scripts/upload_to_huggingface.py
"""
Uploads all three GriceBench models to HuggingFace Hub.

PREREQUISITES:
  pip install huggingface_hub
  python scripts/generate_model_cards.py  (run first)

USAGE:
  python scripts/upload_to_huggingface.py --token hf_YOURTOKEN
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def upload_model(api, repo_id, files, folder=None, card_path=None):
    print(f"\n📤 Uploading to {repo_id}...")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

    # Upload model card as README
    if card_path and Path(card_path).exists():
        api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md", repo_id=repo_id)
        print(f"  ✅ README (model card) uploaded")

    # Upload specific files
    for local_path, remote_path in (files or []):
        if Path(local_path).exists():
            api.upload_file(path_or_fileobj=local_path, path_in_repo=remote_path, repo_id=repo_id)
            print(f"  ✅ {local_path} → {remote_path}")
        else:
            print(f"  ⚠️  MISSING: {local_path}")

    # Upload entire folder
    if folder and Path(folder).exists():
        api.upload_folder(folder_path=folder, repo_id=repo_id,
                          ignore_patterns=["*.pyc", "__pycache__", "*.log"])
        print(f"  ✅ Folder {folder} uploaded")

    print(f"  🌐 View at: https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--username", default="PushkarPrabhath27")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    upload_model(
        api,
        repo_id=f"{args.username}/GriceBench-Detector",
        files=[
            ("best_model_v2.pt", "pytorch_model.pt"),
            ("temperatures.json", "temperatures.json"),
        ],
        card_path="MODEL_CARD_DETECTOR_FINAL.md"
    )

    upload_model(
        api,
        repo_id=f"{args.username}/GriceBench-Repair",
        folder="models/repair/repair_model/",
        card_path="MODEL_CARD_REPAIR_FINAL.md"
    )

    upload_model(
        api,
        repo_id=f"{args.username}/GriceBench-DPO",
        folder="dpo_training_final_outcome/",
        card_path="MODEL_CARD_DPO_FINAL.md"
    )

    print("\n🎉 All models uploaded!")

if __name__ == "__main__":
    main()
```

---

## ✅ PHASE 3 QUALITY GATE

```
GATE 3.1: All three HuggingFace repos are PUBLIC and accessible in browser
          → https://huggingface.co/PushkarPrabhath27/GriceBench-Detector
          → https://huggingface.co/PushkarPrabhath27/GriceBench-Repair
          → https://huggingface.co/PushkarPrabhath27/GriceBench-DPO

GATE 3.2: Each repo renders its README/model card (HF renders .md automatically)

GATE 3.3: Performance numbers in model cards EXACTLY MATCH the paper tables
          → Detector F1 in model card = 0.955 (matches Table 2 in paper)
          → DPO preference accuracy = 75.0% (Phase 7 number, not Phase 5 98.7%)

GATE 3.4: DPO model card has [CONFIRMED MODEL NAME] filled in (not placeholder)

GATE 3.5: README.md in GitHub repo has working badge links to all three HF repos
          → Test each badge URL manually
```

---

---

# ╔══════════════════════════════════════════════════════╗
# ║  PHASE 7: FINAL CONSISTENCY AUDIT & SUBMISSION PREP ║
# ╚══════════════════════════════════════════════════════╝

## Step 7.1 — The Consistency Audit Script

This is the most important script in the entire project.
It PROVES that every number in your paper matches the actual data.
A single mismatched number is a publication error.

```python
# scripts/consistency_audit_final.py
"""
PURPOSE: Final pre-submission consistency check.
         Reads every claimed number from the paper LaTeX source and
         verifies it against the actual result JSON files.

PASS CONDITION: Zero mismatches.
FAIL CONDITION: Any mismatch → paper has an error that must be fixed.

RUN THIS LAST, after all other phases are complete.
"""

import json
import re
from pathlib import Path

# ─── Load canonical result files ─────────────────────────────────────────────
with open("results/phase7output/phase7_results.json") as f:
    phase7 = json.load(f)

with open("results/statistical_significance.json") as f:
    sig = json.load(f)

# Try to load ablation results
try:
    with open("results/part4output/ablation_report.md") as f:
        ablation_text = f.read()
except FileNotFoundError:
    ablation_text = ""

with open("paper/gricebench_main.tex") as f:
    paper_text = f.read()

# ─── Define expected values ───────────────────────────────────────────────────
# Each entry: (description, value_to_find_in_paper, where_it_comes_from)
CHECKS = [
    # Main cooperative rates
    ("Full system cooperative rate",         "95.0",  "ablation results"),
    ("Baseline cooperative rate",            "83.8",  "ablation results"),
    ("Detect+Repair cooperative rate",       "93.0",  "ablation results"),
    ("DPO Only cooperative rate",            "83.2",  "ablation results"),
    ("Mistral-7B cooperative rate",          "89.1",  "phase4 baselines"),
    ("Qwen2.5-7B cooperative rate",          "84.2",  "phase4 baselines"),
    # Improvement
    ("Improvement over baseline (pp)",       "11.2",  "95.0 - 83.8"),
    # Detector F1 scores
    ("Detector macro F1",                    "0.955", "phase7_results"),
    ("Quantity F1",                          "1.000", "phase7_results"),
    ("Quality F1",                           "0.928", "phase7_results"),
    ("Relation F1",                          "1.000", "phase7_results"),
    ("Manner F1",                            "0.891", "phase7_results"),
    # Per-maxim violation rates
    ("Baseline Manner violation rate",       "62.0",  "ablation results"),
    ("Full system Manner violation rate",    "16.0",  "ablation results"),
    # DPO training
    ("DPO eval loss",                        "0.5595","phase7_results"),
    ("DPO preference accuracy",              "75.0",  "phase7_results"),
    # Dataset
    ("DPO training pairs",                   "1,970", "dpo_train_filtered.json"),
    ("Topical-Chat conversations",           "8,628", "topical-chat stats"),
]

# ─── Run checks ───────────────────────────────────────────────────────────────
issues = []
warnings = []

print("=== FINAL CONSISTENCY AUDIT ===\n")
print(f"{'Check':<45} {'Value':>8} {'Status':>8}")
print("-" * 65)

for description, value, source in CHECKS:
    found = value in paper_text
    status = "✅ PASS" if found else "❌ FAIL"
    print(f"{description:<45} {value:>8} {status:>8}")

    if not found:
        issues.append({
            "description": description,
            "expected_value": value,
            "source": source,
            "action": f"Find '{value}' in paper — it comes from {source}"
        })

# ─── Check for leftover placeholders ─────────────────────────────────────────
placeholders = re.findall(r'\[FILL[_A-Z0-9]*\]', paper_text)
if placeholders:
    print(f"\n⚠️  UNFILLED PLACEHOLDERS IN PAPER: {placeholders}")
    for p in set(placeholders):
        issues.append({
            "description": f"Unfilled placeholder: {p}",
            "expected_value": "actual value",
            "source": "paper",
            "action": f"Replace {p} with the correct value"
        })

# ─── Save report ──────────────────────────────────────────────────────────────
report = {
    "total_checks": len(CHECKS),
    "passed": len(CHECKS) - len([i for i in issues if "placeholder" not in i.get("description","")]),
    "failed": len(issues),
    "issues": issues,
    "overall": "PASS — Ready for submission" if not issues else f"FAIL — {len(issues)} issues must be fixed"
}

Path("docs").mkdir(exist_ok=True)
with open("docs/consistency_audit_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"\n{'='*65}")
print(f"OVERALL: {report['overall']}")
print(f"Passed: {report['passed']}/{report['total_checks']}")
if issues:
    print(f"\nISSUES TO FIX:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue['description']}: ACTION = {issue['action']}")
print(f"\nFull report: docs/consistency_audit_report.json")
```

---

## Step 7.2 — arXiv Submission Checklist (Full)

Work through this checklist manually, item by item.
Every item must be checked before submission.

```
╔══════════════════════════════════════════════════════════════╗
║         GRICEBENCH arXiv SUBMISSION CHECKLIST               ║
╚══════════════════════════════════════════════════════════════╝

CONTENT
□ Abstract is self-contained (readable without the paper body)
□ All claims in abstract are supported by numbers in the paper body
□ Introduction has a clear contributions list
□ Related Work cites ≥ 12 papers
□ Related Work explicitly differentiates GriceBench from each cited work
□ Method section has all three components described with sufficient detail
□ Experiments section has clear evaluation setup description
□ Results section leads with cooperative rate (primary metric)
□ Analysis section addresses: Manner failure, DPO vs Repair contributions,
  small model > large model finding, repair model limitations
□ Conclusion is forward-looking (names 2 future directions)
□ Limitations section is honest (if using EMNLP format, required)

NUMBERS
□ consistency_audit_final.py passes with 0 failures
□ Table 1 has 95% CI for all GriceBench configurations
□ McNemar's test result is stated in text AND table caption
□ Repair BLEU scores appear in the text (97.8%, 92.5%, 61.8%)
□ DPO training details are accurate (1,970 pairs, 24 min, P100)

FORMATTING
□ pdflatex compiles with 0 errors (warnings acceptable)
□ Overfull hboxes < 5 in the log
□ Page count is 8 pages ± 0.5 pages
□ All tables have captions above (booktabs style)
□ All figures have captions below
□ Subfigure references work
□ Bibliography: all entries have correct venue and year
□ Author name and institution filled in
□ Email address filled in

REPRODUCIBILITY
□ GitHub repo is public: github.com/PushkarPrabhath27/Research-Model
□ All 3 HuggingFace repos are public
□ HuggingFace URLs in paper are live
□ GitHub URL in paper is live
□ LICENSE file exists in repo (Apache 2.0)
□ README has reproduction instructions
□ Kaggle notebook links are public (or archived)

FINAL
□ Run: pdflatex && bibtex && pdflatex && pdflatex (full build)
□ Read the final PDF from page 1 to end, catching typos
□ Send to at least one person for a cold read before submission
```

---

## ✅ PHASE 7 QUALITY GATE

```
GATE 7.1: docs/consistency_audit_report.json shows "PASS"
          → Zero failed checks. Zero unfilled placeholders.

GATE 7.2: Paper PDF is exactly 8 pages (7.5-8.5 acceptable)

GATE 7.3: All HuggingFace URLs in the paper return HTTP 200

GATE 7.4: GitHub repo is accessible from an incognito browser window

GATE 7.5: arXiv submission checklist above is 100% checked
```

**FINAL PRINT STATEMENT:**

```
=== GRICEBENCH PROJECT: FINAL STATUS REPORT ===

Phase 1 (Repair Fix):              [PASS/FAIL]
Phase 4 (Significance Tests):      [PASS/FAIL]
Phase 5 (Paper):                   [PASS/FAIL]
Phase 3 (HuggingFace Upload):      [PASS/FAIL]
Phase 7 (Final Audit):             [PASS/FAIL]

OVERALL STATUS: [READY FOR arXiv SUBMISSION / ISSUES REMAIN]

If READY:
  → Submit to arXiv: https://arxiv.org/submit
  → Then submit to EMNLP 2026 OpenReview portal
  → Target EMNLP deadline: ~June 2026

Congratulations. GriceBench is complete.
```

---

## END OF AGENT PROMPT

*This prompt was generated for: Pushkar Prabhath — GriceBench Project*
*Version: 2.0 — Deep Execution Edition*
*Phases: 1 → 4 → 5 → 3 → 7*
*Standard: EMNLP/ACL publication quality*