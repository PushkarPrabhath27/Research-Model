"""
scripts/upload_to_huggingface.py
=================================
Phase 3 — Uploads all three GriceBench models to Hugging Face Hub.

Usage:
    C:\\Users\\pushk\\python310\\python.exe scripts\\upload_to_huggingface.py --token YOUR_HF_TOKEN

Prerequisites:
    pip install huggingface_hub

What this uploads:
    PushkarPrabhath27/GriceBench-Detector  ← best_model_v2.pt + temperatures.json
    PushkarPrabhath27/GriceBench-Repair    ← models/repair/repair_model/ folder
    PushkarPrabhath27/GriceBench-DPO       ← dpo_training_final_outcome/ folder

After running, verify at:
    https://huggingface.co/PushkarPrabhath27/GriceBench-Detector
    https://huggingface.co/PushkarPrabhath27/GriceBench-Repair
    https://huggingface.co/PushkarPrabhath27/GriceBench-DPO

Author: GriceBench Research
Version: 1.0 — March 2026
"""

import argparse
import json
import os
import sys
import time

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DETECTOR_FILES = [
    ("best_model_v2.pt",              "pytorch_model.pt"),
    ("temperatures.json",             "temperatures.json"),
    ("history_v2.json",               "training_history.json"),
    ("MODEL_CARD_DETECTOR_FINAL.md",  "README.md"),
]

REPAIR_FOLDER   = os.path.join(BASE_DIR, "models", "repair", "repair_model")
DPO_FOLDER      = os.path.join(BASE_DIR, "dpo_training_final_outcome")
IDENTITY_PATH   = os.path.join(BASE_DIR, "docs", "dpo_model_identity.json")


def _check_prerequisites() -> bool:
    """Verify all required files exist before attempting upload."""
    print("Checking prerequisites...")
    ok = True
    checks = [
        (os.path.join(BASE_DIR, "best_model_v2.pt"), "Detector model weights"),
        (os.path.join(BASE_DIR, "temperatures.json"), "Temperature scaling config"),
        (REPAIR_FOLDER, "Repair model folder"),
        (DPO_FOLDER, "DPO adapter folder"),
        (os.path.join(BASE_DIR, "MODEL_CARD_DETECTOR_FINAL.md"), "Detector model card"),
        (os.path.join(BASE_DIR, "MODEL_CARD_REPAIR_FINAL.md"), "Repair model card"),
        (os.path.join(BASE_DIR, "MODEL_CARD_DPO_FINAL.md"), "DPO model card"),
    ]
    for path, label in checks:
        exists = os.path.exists(path)
        print(f"  {'✅' if exists else '❌'} {label}: {path}")
        if not exists:
            ok = False
    return ok


def upload_detector(api, username: str) -> bool:
    """Upload DeBERTa detector to HuggingFace."""
    repo_id = f"{username}/GriceBench-Detector"
    print(f"\n{'='*60}")
    print(f"📤 Uploading Detector → {repo_id}")
    print(f"{'='*60}")

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"  Repo created/verified: https://huggingface.co/{repo_id}")

        for local_name, remote_name in DETECTOR_FILES:
            local_path = os.path.join(BASE_DIR, local_name)
            if os.path.exists(local_path):
                size_mb = os.path.getsize(local_path) / 1e6
                print(f"  Uploading {local_name} ({size_mb:.0f} MB)...", end=" ", flush=True)
                t0 = time.time()
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_name,
                    repo_id=repo_id,
                )
                print(f"done ({time.time()-t0:.0f}s) ✅")
            else:
                print(f"  ⚠️  MISSING (skipped): {local_name}")

        print(f"\n✅ Detector uploaded → https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"\n❌ Detector upload FAILED: {e}")
        return False


def upload_repair(api, username: str) -> bool:
    """Upload T5 repair model to HuggingFace."""
    repo_id = f"{username}/GriceBench-Repair"
    print(f"\n{'='*60}")
    print(f"📤 Uploading Repair Model → {repo_id}")
    print(f"{'='*60}")

    # Add model card from root
    model_card_src = os.path.join(BASE_DIR, "MODEL_CARD_REPAIR_FINAL.md")
    model_card_dst = os.path.join(REPAIR_FOLDER, "README.md")
    card_was_copied = False
    if os.path.exists(model_card_src) and not os.path.exists(model_card_dst):
        import shutil
        shutil.copy(model_card_src, model_card_dst)
        card_was_copied = True

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        folder_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(REPAIR_FOLDER)
            for f in files
        ) / 1e6
        print(f"  Uploading repair_model/ folder ({folder_size:.0f} MB)...")
        t0 = time.time()
        api.upload_folder(
            folder_path=REPAIR_FOLDER,
            repo_id=repo_id,
            ignore_patterns=["*.pyc", "__pycache__", "*.tmp"],
        )
        print(f"  Folder uploaded in {time.time()-t0:.0f}s ✅")
        print(f"\n✅ Repair model uploaded → https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"\n❌ Repair upload FAILED: {e}")
        return False
    finally:
        # Clean up temp README if we added one
        if card_was_copied and os.path.exists(model_card_dst):
            os.remove(model_card_dst)


def upload_dpo(api, username: str) -> bool:
    """Upload DPO LoRA adapter to HuggingFace."""
    repo_id = f"{username}/GriceBench-DPO"
    print(f"\n{'='*60}")
    print(f"📤 Uploading DPO Adapter → {repo_id}")
    print(f"{'='*60}")

    # Add model card
    model_card_src = os.path.join(BASE_DIR, "MODEL_CARD_DPO_FINAL.md")
    model_card_dst = os.path.join(DPO_FOLDER, "README.md")
    card_was_copied = False
    if os.path.exists(model_card_src) and not os.path.exists(model_card_dst):
        import shutil
        shutil.copy(model_card_src, model_card_dst)
        card_was_copied = True

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        folder_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(DPO_FOLDER)
            for f in files
        ) / 1e6
        print(f"  Uploading dpo_training_final_outcome/ folder ({folder_size:.0f} MB)...")
        t0 = time.time()
        api.upload_folder(
            folder_path=DPO_FOLDER,
            repo_id=repo_id,
            ignore_patterns=["*.pyc", "__pycache__", "history*", "*.tmp"],
        )
        print(f"  Folder uploaded in {time.time()-t0:.0f}s ✅")
        print(f"\n✅ DPO adapter uploaded → https://huggingface.co/{repo_id}")
        return True

    except Exception as e:
        print(f"\n❌ DPO upload FAILED: {e}")
        return False
    finally:
        if card_was_copied and os.path.exists(model_card_dst):
            os.remove(model_card_dst)


def main():
    parser = argparse.ArgumentParser(
        description="Upload all GriceBench models to HuggingFace Hub."
    )
    parser.add_argument("--token",    required=True, help="HuggingFace write-access token")
    parser.add_argument("--username", default="PushkarPrabhath27", help="HF username")
    parser.add_argument("--skip-detector", action="store_true")
    parser.add_argument("--skip-repair",   action="store_true")
    parser.add_argument("--skip-dpo",      action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("GriceBench HuggingFace Upload")
    print("=" * 60)
    print(f"Username: {args.username}")
    print(f"Target repos:")
    print(f"  {args.username}/GriceBench-Detector")
    print(f"  {args.username}/GriceBench-Repair")
    print(f"  {args.username}/GriceBench-DPO")

    # Check prerequisites
    if not _check_prerequisites():
        print("\n⚠️  Some files are missing. Create the missing model cards first:")
        print("   - MODEL_CARD_DETECTOR_FINAL.md")
        print("   - MODEL_CARD_REPAIR_FINAL.md")
        print("   - MODEL_CARD_DPO_FINAL.md")
        print("\nThese should have been created by the earlier phase steps.")
        print("Continuing with available files...")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("\n❌ huggingface_hub not installed.")
        print("   Install: C:\\Users\\pushk\\python310\\python.exe -m pip install huggingface_hub")
        return 1

    api = HfApi(token=args.token)

    # Verify token
    try:
        user = api.whoami()
        print(f"\nAuthenticated as: {user['name']}")
    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        print("   Check your HF token at: https://huggingface.co/settings/tokens")
        return 1

    results = {}
    if not args.skip_detector:
        results["detector"] = upload_detector(api, args.username)
    if not args.skip_repair:
        results["repair"] = upload_repair(api, args.username)
    if not args.skip_dpo:
        results["dpo"] = upload_dpo(api, args.username)

    # Summary
    print(f"\n{'='*60}")
    print("UPLOAD SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        if not success:
            all_ok = False
        print(f"  {model:10}: {status}")

    if all_ok:
        print(f"\n🎉 All models uploaded successfully!")
        print(f"\nVerify at:")
        print(f"  https://huggingface.co/{args.username}/GriceBench-Detector")
        print(f"  https://huggingface.co/{args.username}/GriceBench-Repair")
        print(f"  https://huggingface.co/{args.username}/GriceBench-DPO")
        print(f"\nNext step: Update README.md with live HF badge links")
    else:
        print("\n⚠️  Some uploads failed. Re-run with --skip-* flags for completed ones.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
