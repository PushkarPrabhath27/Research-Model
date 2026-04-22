"""
scripts/validate_dpo_loading.py
================================
Phase 2 — Validate that the DPO LoRA adapter loads correctly with its base model.

This script:
1. Reads docs/dpo_model_identity.json (created by identify_dpo_model.py)
2. Downloads/loads the base model + LoRA adapter
3. Runs a small generation test
4. Saves docs/dpo_validation_result.json

Run AFTER identify_dpo_model.py:
    C:\\Users\\pushk\\python310\\python.exe scripts\\validate_dpo_loading.py

If this script fails, it means the adapter and base model are incompatible.
The fix is documented in the output.

Author: GriceBench Research
Version: 1.0 — March 2026
"""

import json
import os
import sys
import time

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_PATH   = os.path.join(BASE_DIR, "dpo_training_final_outcome")
IDENTITY_PATH  = os.path.join(BASE_DIR, "docs", "dpo_model_identity.json")
OUTPUT_PATH    = os.path.join(BASE_DIR, "docs", "dpo_validation_result.json")

TEST_PROMPTS = [
    "Context: Do you enjoy watching sports?\nResponse:",
    "Context: What do you think about climate change?\nResponse:",
    "Context: Have you seen any good movies recently?\nResponse:",
]


def main():
    print("=" * 60)
    print("DPO MODEL VALIDATION")
    print("=" * 60)

    # ── Step 1: Load identity ──────────────────────────────────────────────────
    if not os.path.exists(IDENTITY_PATH):
        print(f"\n❌ Identity file not found: {IDENTITY_PATH}")
        print("   Run identify_dpo_model.py first.")
        _save_failure("identity_file_not_found", "UNKNOWN", "Run identify_dpo_model.py first.")
        return 1

    with open(IDENTITY_PATH) as f:
        identity = json.load(f)

    base_model = identity["resolved_base_model"]
    lora_rank  = identity["adapter_config"]["lora_rank"]
    lora_alpha = identity["adapter_config"]["lora_alpha"]

    print(f"\nBase model to load: {base_model}")
    print(f"Adapter path: {ADAPTER_PATH}")
    print(f"LoRA rank={lora_rank}, alpha={lora_alpha}")

    if "UNKNOWN" in base_model:
        print("\n❌ Base model is UNKNOWN — cannot validate.")
        print("   Run identify_dpo_model.py and resolve the base model first.")
        _save_failure("unknown_base_model", base_model, "Resolve base model identity first.")
        return 1

    # ── Step 2: Import (lazy — only fail if not installed) ─────────────────────
    print("\nImporting libraries...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel, PeftConfig
    except ImportError as e:
        print(f"\n❌ Missing library: {e}")
        print("   Install: pip install transformers peft torch")
        _save_failure("missing_library", base_model, str(e))
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Step 3: Verify adapter config is consistent ────────────────────────────
    print("\nReading PeftConfig from adapter...")
    try:
        peft_config = PeftConfig.from_pretrained(ADAPTER_PATH)
        adapter_base = peft_config.base_model_name_or_path
        print(f"  PeftConfig.base_model_name_or_path = '{adapter_base}'")
        if adapter_base and adapter_base != base_model:
            print(f"  ⚠️  Adapter config says '{adapter_base}' but identity resolved '{base_model}'")
            print(f"  Using adapter config value: {adapter_base}")
            base_model = adapter_base   # Trust the actual config over our inference
    except Exception as e:
        print(f"  ⚠️  Could not read PeftConfig: {e} — using identity-resolved model")

    # ── Step 4: Load base model ────────────────────────────────────────────────
    print(f"\nLoading base model '{base_model}'...")
    print("  (This may take a few minutes on first run — downloading weights)")
    start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("  Set pad_token = eos_token")

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else None,
        )
        print(f"  Base model loaded in {time.time()-start:.1f}s")
        print(f"  Parameters: {sum(p.numel() for p in base.parameters()):,}")
    except Exception as e:
        print(f"\n❌ Failed to load base model: {e}")
        _save_failure("base_model_load_failed", base_model, str(e))
        return 1

    # ── Step 5: Load LoRA adapter ─────────────────────────────────────────────
    print(f"\nLoading LoRA adapter from '{ADAPTER_PATH}'...")
    start = time.time()
    try:
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        model.eval()
        print(f"  Adapter loaded in {time.time()-start:.1f}s")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
    except Exception as e:
        print(f"\n❌ Failed to load LoRA adapter: {e}")
        print("\nPossible causes:")
        print("  1. Base model architecture doesn't match the adapter")
        print("  2. Adapter was trained with different target_modules")
        print(f"\nAdapter config target_modules: {identity['adapter_config']['target_modules']}")
        print("Recommended action: Retrain DPO on Kaggle with explicit base model name.")
        _save_failure("adapter_load_failed", base_model, str(e))
        return 1

    # ── Step 6: Generation test ────────────────────────────────────────────────
    print("\nRunning generation tests...")
    test_outputs = []
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device if hasattr(model, "device") else device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        test_outputs.append({"prompt": prompt, "generated": generated.strip()})
        print(f"\n  Prompt: {prompt}")
        print(f"  Response: {generated.strip()[:100]}")

    # ── Step 7: Save results ───────────────────────────────────────────────────
    result = {
        "status": "success",
        "base_model": base_model,
        "adapter_path": ADAPTER_PATH,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "device_used": device,
        "test_outputs": test_outputs,
        "note": (
            "All tests passed. The DPO LoRA adapter loads correctly with the base model. "
            "Use this base model name in all documentation."
        ),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ VALIDATION SUCCESSFUL")
    print(f"   Base model confirmed: {base_model}")
    print(f"   Saved to: {OUTPUT_PATH}")
    print("\nNext steps:")
    print(f"  1. Update MODEL_CARD_DPO.md: Base Model = {base_model}")
    print(f"  2. Update README.md Model Zoo: generator = {base_model}")
    print(f"  3. Update CHAPTER_13_IMPLEMENTATION.md if it references GPT-2")
    return 0


def _save_failure(reason: str, base_model: str, error_msg: str):
    """Save a failure result for debugging."""
    result = {
        "status": "failed",
        "reason": reason,
        "base_model": base_model,
        "error": error_msg,
        "recommended_action": (
            "If base model loading fails: ensure the model exists on HuggingFace "
            "and you have internet access. If adapter mismatch: retrain DPO on "
            "Kaggle with SmolLM2-360M-Instruct and save adapter_config with "
            "explicit base_model_name_or_path field."
        ),
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Failure saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
