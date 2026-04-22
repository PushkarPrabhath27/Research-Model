"""
scripts/identify_dpo_model.py
==============================
Phase 2 — Single source of truth for DPO model identity.

This script reads the actual adapter files in dpo_training_final_outcome/
and resolves the base model name once and for all. It is the ONLY place
to look up what the DPO generator's base model is — all documentation
must be updated to match what this script reports.

Run:
    C:\\Users\\pushk\\python310\\python.exe scripts\\identify_dpo_model.py

Output:
    docs/dpo_model_identity.json   ← single source of truth
    Prints resolved identity to console

Author: GriceBench Research
Version: 1.0 — March 2026
"""

import json
import os
import sys

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_DIR         = os.path.join(BASE_DIR, "dpo_training_final_outcome")
ADAPTER_CONFIG_PATH = os.path.join(ADAPTER_DIR, "adapter_config.json")
HISTORY_PATH        = os.path.join(ADAPTER_DIR, "history (1).json")
TOKENIZER_CONFIG    = os.path.join(ADAPTER_DIR, "tokenizer_config.json")
VOCAB_PATH          = os.path.join(ADAPTER_DIR, "vocab.json")
MERGES_PATH         = os.path.join(ADAPTER_DIR, "merges.txt")
OUTPUT_DIR          = os.path.join(BASE_DIR, "docs")
OUTPUT_PATH         = os.path.join(OUTPUT_DIR, "dpo_model_identity.json")


def check_file(path: str, label: str) -> bool:
    exists = os.path.exists(path)
    print(f"  {'✅' if exists else '❌'} {label}: {path}")
    return exists


def infer_base_model_from_tokenizer(tok_config: dict) -> str | None:
    """
    Try to infer the base model from tokenizer-level hints.
    Different models have distinct tokenizer signatures.
    """
    tok_class = tok_config.get("tokenizer_class", "")
    name_or_path = tok_config.get("name_or_path", "")
    model_type = tok_config.get("model_type", "")

    # SmolLM2 uses the LlamaTokenizer / PreTrainedTokenizerFast
    if "llama" in tok_class.lower() or "smollm" in name_or_path.lower():
        return "HuggingFaceTB/SmolLM2-360M-Instruct"

    # Qwen uses Qwen2Tokenizer
    if "qwen" in tok_class.lower() or "qwen" in name_or_path.lower():
        if "7b" in name_or_path.lower():
            return "Qwen/Qwen2.5-7B-Instruct"
        return name_or_path if name_or_path else "Qwen/Qwen2.5-7B-Instruct"

    # GPT-2 uses GPT2Tokenizer
    if "gpt2" in tok_class.lower() or "gpt2" in name_or_path.lower():
        if "medium" in name_or_path.lower():
            return "gpt2-medium"
        return name_or_path if name_or_path else "gpt2-medium"

    # Try vocab size as a discriminating signal
    return None


def infer_base_model_from_vocab(vocab_path: str) -> str | None:
    """
    GPT-2's vocab.json has exactly 50,257 entries.
    SmolLM2 / LLaMA has different vocab size.
    """
    if not os.path.exists(vocab_path):
        return None
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    n = len(vocab)
    print(f"  Vocab size: {n}")
    if n == 50257:
        return "gpt2-medium"  # GPT-2 family vocab
    elif n in [32000, 32001]:
        return "HuggingFaceTB/SmolLM2-360M-Instruct"  # LLaMA-family
    elif n in [151936, 152064]:
        return "Qwen/Qwen2.5-7B-Instruct"  # Qwen2.5 vocab
    return None


def main():
    print("=" * 60)
    print("DPO MODEL IDENTITY RESOLVER")
    print("=" * 60)

    # ── Check files exist ──────────────────────────────────────────────────────
    print("\n[1] Checking files:")
    has_adapter = check_file(ADAPTER_CONFIG_PATH, "adapter_config.json")
    has_history = check_file(HISTORY_PATH, "history (1).json")
    has_tok_cfg = check_file(TOKENIZER_CONFIG, "tokenizer_config.json")
    has_vocab   = check_file(VOCAB_PATH, "vocab.json")
    has_merges  = check_file(MERGES_PATH, "merges.txt (GPT-2 specific)")

    if not has_adapter:
        print("\n❌ adapter_config.json NOT FOUND.")
        print(f"   Expected at: {ADAPTER_CONFIG_PATH}")
        print("   Make sure dpo_training_final_outcome/ folder is intact.")
        sys.exit(1)

    # ── Read adapter_config.json — THE primary source ─────────────────────────
    print("\n[2] Reading adapter_config.json:")
    with open(ADAPTER_CONFIG_PATH) as f:
        adapter_config = json.load(f)

    print(json.dumps(adapter_config, indent=2))

    base_model_from_adapter = adapter_config.get("base_model_name_or_path", "")
    lora_rank   = adapter_config.get("r", adapter_config.get("lora_rank", "UNKNOWN"))
    lora_alpha  = adapter_config.get("lora_alpha", "UNKNOWN")
    target_mods = adapter_config.get("target_modules", "UNKNOWN")
    peft_type   = adapter_config.get("peft_type", "UNKNOWN")

    # ── Read tokenizer_config.json — secondary signal ─────────────────────────
    tok_config = {}
    inferred_from_tok = None
    if has_tok_cfg:
        print("\n[3] Reading tokenizer_config.json:")
        with open(TOKENIZER_CONFIG) as f:
            tok_config = json.load(f)
        print(json.dumps(tok_config, indent=2))
        inferred_from_tok = infer_base_model_from_tokenizer(tok_config)
        if inferred_from_tok:
            print(f"\n  → Inferred from tokenizer: {inferred_from_tok}")

    # ── Infer from vocab size — tertiary signal ────────────────────────────────
    print("\n[4] Checking vocab.json (vocab size signal):")
    inferred_from_vocab = infer_base_model_from_vocab(VOCAB_PATH)
    if inferred_from_vocab:
        print(f"  → Inferred from vocab size: {inferred_from_vocab}")

    # ── GPT-2 specific signal: merges.txt only exists for BPE models (GPT-2) ──
    is_bpe = has_merges
    print(f"\n[5] BPE tokenizer (GPT-2 family): {is_bpe}")

    # ── Resolve final identity ─────────────────────────────────────────────────
    print("\n[6] Resolving final base model identity:")

    if base_model_from_adapter and base_model_from_adapter not in ["", "UNKNOWN"]:
        resolved_base_model = base_model_from_adapter
        resolution_method = "adapter_config.json → base_model_name_or_path (most reliable)"
    elif inferred_from_tok:
        resolved_base_model = inferred_from_tok
        resolution_method = "tokenizer_config.json heuristic"
    elif inferred_from_vocab:
        resolved_base_model = inferred_from_vocab
        resolution_method = "vocab size heuristic"
    elif is_bpe:
        resolved_base_model = "gpt2-medium"  # GPT-2 is the only BPE model in the project
        resolution_method = "BPE tokenizer presence heuristic (merges.txt)"
    else:
        resolved_base_model = "UNKNOWN — manual inspection required"
        resolution_method = "no signal found"

    print(f"  Resolved:  {resolved_base_model}")
    print(f"  Method:    {resolution_method}")

    # ── Training history (optional info) ──────────────────────────────────────
    training_info = {}
    if has_history:
        print("\n[7] Reading training history:")
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        if isinstance(history, list) and history:
            print(json.dumps(history[0], indent=2))
            training_info = history[0] if isinstance(history[0], dict) else {}
        elif isinstance(history, dict):
            print(json.dumps(history, indent=2))
            training_info = history

    # ── Save output ────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    identity = {
        "resolved_base_model": resolved_base_model,
        "resolution_method": resolution_method,
        "adapter_config": {
            "base_model_name_or_path": base_model_from_adapter,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "target_modules": target_mods,
            "peft_type": peft_type,
        },
        "tokenizer_hints": {
            "tokenizer_class": tok_config.get("tokenizer_class", "UNKNOWN"),
            "name_or_path": tok_config.get("name_or_path", "UNKNOWN"),
            "vocab_size_signal": inferred_from_vocab,
            "has_bpe_merges_file": is_bpe,
        },
        "training_info": training_info,
        "phase5_vs_phase7_discrepancy": (
            "Phase 5 eval (98.7% preference accuracy) used a different base model "
            "(likely GPT-2-medium) evaluated on the training distribution. "
            "Phase 7 eval (75.0% preference accuracy) is the canonical number — "
            f"it used {resolved_base_model} evaluated on a held-out test set."
        ),
        "documentation_action_required": (
            f"Update README.md, MODEL_CARD_DPO.md, and CHAPTER_13_IMPLEMENTATION.md "
            f"to consistently use: {resolved_base_model}"
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(identity, f, indent=2)

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESOLVED DPO MODEL IDENTITY")
    print("=" * 60)
    print(f"  Base model:      {resolved_base_model}")
    print(f"  LoRA rank:       {lora_rank}")
    print(f"  LoRA alpha:      {lora_alpha}")
    print(f"  Target modules:  {target_mods}")
    print(f"  PEFT type:       {peft_type}")
    print(f"\n  Saved to: {OUTPUT_PATH}")

    if "UNKNOWN" in resolved_base_model:
        print("\n⚠️  WARNING: Could not auto-resolve base model.")
        print("   Manual action: open adapter_config.json and check 'base_model_name_or_path'")
        print(f"   Path: {ADAPTER_CONFIG_PATH}")
        return 1

    print("\n✅ Model identity resolved. Next steps:")
    print(f"   1. Run: C:\\Users\\pushk\\python310\\python.exe scripts\\validate_dpo_loading.py")
    print(f"   2. Update MODEL_CARD_DPO.md with: {resolved_base_model}")
    print(f"   3. Update README.md Model Zoo with: {resolved_base_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
