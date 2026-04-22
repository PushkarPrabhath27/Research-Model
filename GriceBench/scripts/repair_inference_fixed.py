"""
scripts/repair_inference_fixed.py
==================================
FIXED repair model inference with three-layer degeneracy prevention.

Replaces the old repair inference wherever it is called.
It is a DROP-IN REPLACEMENT — same high-level interface, better behavior.

Layer 1 — Generation hyperparameter fix:
    - Quantity/Quality violations: beam search with repetition_penalty=1.5
      and no_repeat_ngram_size=3
    - Manner violations: temperature sampling (temperature=0.85, top_p=0.92)
      because Manner repairs need diverse rewrites, not exact copies

Layer 2 — Post-generation output validation:
    - is_degenerate() detects punctuation bursts, exclamation density,
      trigram loops, and near-empty outputs

Layer 3 — Graceful degradation:
    - If the repaired output is degenerate, return the ORIGINAL response
      with flagged metadata (is_fallback=True) instead of publishing garbage

IMPORTANT: Relation violations are NOT handled here.
    They must be routed to the FAISS retrieval system separately.

Usage:
    from scripts.repair_inference_fixed import FixedRepairModel
    model = FixedRepairModel(model_path="models/repair/repair_model/")
    result = model.repair(input_text, violation_type="manner")
    print(result["repaired_text"])  # clean output or graceful fallback

Author: GriceBench Research
Version: 2.0 (March 2026) — Production grade
"""

import re
import logging
from collections import Counter
from typing import Optional

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)


# ─── Signal Thresholds (empirically tuned on repair_val.json) ─────────────────
_EXCL_DENSITY_THRESHOLD = 0.15     # >15% of words are "!" → degenerate
_PUNCT_DENSITY_THRESHOLD = 0.20    # >20% tokens are punct-only → degenerate
_TRIGRAM_REPEAT_THRESHOLD = 2      # same 3-gram appears >2 times → degenerate
_MIN_OUTPUT_LEN = 8                # shorter than 8 chars → degenerate
_CONSEC_PUNCT_PATTERN = re.compile(r"([!?,.;:\-])\1{2,}")    # 3+ same punct chars


def is_degenerate(text: str) -> bool:
    """
    Multi-signal degeneracy detector for T5 repair output.

    Checks four independent signals and returns True if ANY fires.
    This is intentionally conservative — a false positive (unnecessary fallback)
    is always better than publishing a degenerate output.

    Signals:
        1. Consecutive identical punctuation bursts (e.g., "!!!" or ",,,,")
        2. Exclamation mark density > 15% of all word tokens
        3. Trigram repetition — same 3-gram appears more than 2 times
        4. Output too short to be a valid repaired response (< 8 characters)

    Args:
        text: The decoded repair model output (skip_special_tokens=True)

    Returns:
        True  → output is degenerate, use fallback
        False → output is plausibly valid

    Examples:
        >>> is_degenerate("It! The! Movie!! really!! did! convey!!!")
        True
        >>> is_degenerate("The movie had a strong and inspiring message.")
        False
        >>> is_degenerate("I agree.")
        False  # short but not degenerate
        >>> is_degenerate("")
        True
    """
    # Guard: null / near-empty output
    if not text or len(text.strip()) < _MIN_OUTPUT_LEN:
        logger.debug("Degenerate: too short — '%s'", text[:40])
        return True

    # Signal 1: Consecutive identical punctuation bursts
    if _CONSEC_PUNCT_PATTERN.search(text):
        logger.debug("Degenerate: consecutive punct — '%s'", text[:80])
        return True

    tokens = text.split()
    n_tokens = len(tokens)

    if n_tokens == 0:
        return True

    # Signal 2: Exclamation mark density
    excl_density = text.count("!") / n_tokens
    if excl_density > _EXCL_DENSITY_THRESHOLD:
        logger.debug(
            "Degenerate: excl density=%.2f — '%s'", excl_density, text[:80]
        )
        return True

    # Signal 3: Punctuation-only token density
    punct_only = sum(1 for t in tokens if re.match(r"^[!?,.;:\-]+$", t))
    if punct_only / n_tokens > _PUNCT_DENSITY_THRESHOLD:
        logger.debug(
            "Degenerate: punct token density=%.2f — '%s'",
            punct_only / n_tokens,
            text[:80],
        )
        return True

    # Signal 4: Trigram repetition (handles looping / stuck generation)
    if n_tokens >= 3:
        trigrams = [tuple(tokens[i : i + 3]) for i in range(n_tokens - 2)]
        max_repeat = max(Counter(trigrams).values()) if trigrams else 0
        if max_repeat > _TRIGRAM_REPEAT_THRESHOLD:
            logger.debug(
                "Degenerate: trigram repeat=%d — '%s'", max_repeat, text[:80]
            )
            return True

    return False


class FixedRepairModel:
    """
    Production-grade T5 repair model wrapper with degeneracy prevention.

    Generation strategy:
        - Quantity violations  → beam search (need precise length control)
        - Quality violations   → beam search (need factual precision)
        - Manner violations    → temperature sampling (need diverse rewrites)
        - Relation violations  → NOT HANDLED — route to FAISS retrieval

    The class maintains a small set of Prometheus-style counters in memory
    (total_repaired, total_fallback, degenerate_by_maxim) so callers can
    instrument failure rates without adding a separate monitoring dependency.

    Args:
        model_path: Path to the T5 repair model directory (HF format)
        device: "auto" selects cuda if available, else cpu

    Example:
        model = FixedRepairModel("models/repair/repair_model/")
        result = model.repair(
            "fix manner: [CONTEXT] Hi [RESPONSE] I think so, the movie was... good?",
            violation_type="manner"
        )
        # result = {
        #   "repaired_text": "I think so, the movie conveyed a clear message.",
        #   "is_fallback": False,
        #   "is_degenerate": False,
        #   "generation_method": "sample"
        # }
    """

    def __init__(self, model_path: str, device: str = "auto"):
        logger.info("Loading FixedRepairModel from '%s'...", model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = self.model.to(self.device)
        self.model.eval()

        # Internal telemetry counters (not persisted across restarts)
        self._stats = {
            "total_repaired": 0,
            "total_degenerate": 0,
            "total_fallback": 0,
            "degenerate_by_maxim": {
                "quantity": 0, "quality": 0, "manner": 0
            },
        }

        logger.info("FixedRepairModel ready on device=%s", self.device)

    # ── Private generation helpers ──────────────────────────────────────────

    def _generate_beam(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
        """
        Beam search with repetition penalties.
        Used for: Quantity, Quality violations.

        Key fixes over the broken original:
            repetition_penalty=1.5  — discourages repeating already-generated tokens
            no_repeat_ngram_size=3  — hard prevents 3-gram loops
            min_length=8            — never returns an empty or near-empty repair
        """
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                max_length=128,
                min_length=8,
                early_stopping=True,
                repetition_penalty=1.5,        # KEY FIX
                no_repeat_ngram_size=3,         # KEY FIX
                length_penalty=1.0,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _generate_sample(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
        """
        Temperature sampling with nucleus (top-p) filtering.
        Used for: Manner violations — needs diverse rewrites.

        temperature=0.85 balances creativity vs coherence.
        top_p=0.92 cuts off low-probability tail tokens.
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
                repetition_penalty=1.5,        # Also apply to sampling
                no_repeat_ngram_size=3,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _extract_original_response(self, input_text: str) -> str:
        """
        Extract the original response from the repair prompt for graceful
        fallback. The prompt format is:

            "fix {type}: [CONTEXT] {context} [RESPONSE] {original_response}"

        If parsing fails, returns the last sentence of the input.
        """
        if "[RESPONSE]" in input_text:
            return input_text.split("[RESPONSE]")[-1].strip()
        # Fallback: try splitting on [RESP] or just return last sentence
        if "[RESP]" in input_text:
            return input_text.split("[RESP]")[-1].strip()
        sentences = input_text.strip().rstrip(".").split(".")
        return (sentences[-1].strip() + ".") if sentences else input_text

    # ── Public interface ────────────────────────────────────────────────────

    def repair(
        self,
        input_text: str,
        violation_type: str,
        fallback_to_original: bool = True,
    ) -> dict:
        """
        Main repair interface. Drop-in replacement for old repair logic.

        Args:
            input_text: The formatted repair prompt. Expected format:
                "fix {violation_type}: [CONTEXT] {ctx} [RESPONSE] {response}"
            violation_type: One of 'quantity', 'quality', 'manner'.
                            'relation' is NOT supported — use FAISS retrieval.
            fallback_to_original: If True and output is degenerate,
                return the extracted original with is_fallback=True.

        Returns:
            {
                "repaired_text":    str,   # The repaired (or fallback) text
                "is_fallback":      bool,  # True if we returned the original
                "is_degenerate":    bool,  # True if degeneracy was detected
                "generation_method": str,  # "beam" or "sample"
            }

        Raises:
            ValueError: if violation_type is 'relation' (must use FAISS)
            ValueError: if violation_type is unknown
        """
        if violation_type == "relation":
            raise ValueError(
                "Relation violations must be handled by the FAISS retrieval "
                "system, not the T5 repair model. Route them there first."
            )
        if violation_type not in {"quantity", "quality", "manner"}:
            raise ValueError(
                f"Unknown violation_type='{violation_type}'. "
                "Must be one of: quantity, quality, manner."
            )

        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Route to appropriate generation strategy
        if violation_type == "manner":
            output_text = self._generate_sample(
                inputs["input_ids"], inputs["attention_mask"]
            )
            generation_method = "sample"
        else:  # quantity, quality
            output_text = self._generate_beam(
                inputs["input_ids"], inputs["attention_mask"]
            )
            generation_method = "beam"

        # Check for degeneracy
        degenerate = is_degenerate(output_text)
        self._stats["total_repaired"] += 1

        if degenerate:
            self._stats["total_degenerate"] += 1
            self._stats["degenerate_by_maxim"][violation_type] = (
                self._stats["degenerate_by_maxim"].get(violation_type, 0) + 1
            )
            logger.warning(
                "Degenerate repair output for violation_type='%s' | "
                "input[:80]='%s' | output[:80]='%s'",
                violation_type,
                input_text[:80],
                output_text[:80],
            )

            if fallback_to_original:
                self._stats["total_fallback"] += 1
                original = self._extract_original_response(input_text)
                return {
                    "repaired_text": original,
                    "is_fallback": True,
                    "is_degenerate": True,
                    "generation_method": generation_method,
                }

        return {
            "repaired_text": output_text,
            "is_fallback": False,
            "is_degenerate": degenerate,
            "generation_method": generation_method,
        }

    def get_stats(self) -> dict:
        """
        Return current session telemetry counters.
        Useful for logging / monitoring during batch repair runs.
        """
        total = self._stats["total_repaired"]
        fallback_rate = (
            self._stats["total_fallback"] / total if total > 0 else 0.0
        )
        return {
            **self._stats,
            "fallback_rate": round(fallback_rate, 4),
            "degenerate_rate": round(
                self._stats["total_degenerate"] / total if total > 0 else 0.0, 4
            ),
        }


# ─── CLI self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick smoke test — run directly to verify the is_degenerate function
    works correctly before deploying to Kaggle.

    Usage:
        C:\\Users\\pushk\\python310\\python.exe scripts\\repair_inference_fixed.py
    """
    print("=" * 60)
    print("is_degenerate() smoke tests")
    print("=" * 60)

    CASES = [
        # (text, expected, description)
        ("It! The! Movie!! really!! did! convey!!! a!!! powerful!!!!", True,  "classic degenerate output"),
        ("The movie conveyed a powerful and inspiring message.",        False, "clean repair output"),
        ("I think the author made a strong case for this argument.",    False, "normal length clean"),
        ("yes! yes! yes! yes! yes! yes! yes! yes! yes!",               True,  "exclamation density"),
        ("A B C A B C A B C A B C.",                                   True,  "trigram repeat"),
        ("",                                                            True,  "empty string"),
        ("OK.",                                                         True,  "too short"),
        ("I agree with you on that point.",                             False, "short but ok"),
        ("Well, I think so.",                                           False, "very short but ok"),
        ("Paris Paris Paris Paris Paris Paris Paris.",                  True,  "word repeat (trigram)"),
    ]

    all_passed = True
    for text, expected, desc in CASES:
        result = is_degenerate(text)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        if result != expected:
            all_passed = False
        print(f"  {status} ({desc})")
        if result != expected:
            print(f"         Expected={expected}, Got={result}")
            print(f"         Text: '{text[:80]}'")

    print()
    if all_passed:
        print("All tests passed. repair_inference_fixed.py is ready.")
    else:
        print("FAILURES detected. Fix is_degenerate() before using in production.")
