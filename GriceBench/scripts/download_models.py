"""
Download pretrained GriceBench models from Hugging Face
"""

import os
import argparse
from pathlib import Path
import requests
from tqdm import tqdm


def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_detector(models_dir):
    """Download detector model"""
    print("\\n📥 Downloading Detector Model (735MB)...")
    
    detector_dir = Path(models_dir) / "detector"
    detector_dir.mkdir(parents=True, exist_ok=True)
    
    # Download checkpoint
    url = "https://huggingface.co/yourusername/GriceBench-Detector/resolve/main/best_model.pt"
    destination = detector_dir / "best_model.pt"
    
    if destination.exists():
        print(f"✅ {destination} already exists, skipping")
    else:
        download_file(url, destination)
        print(f"✅ Detector saved to {destination}")


def download_repair(models_dir):
    """Download repair model"""
    print("\\n📥 Downloading Repair Model (900MB)...")
    
    repair_dir = Path(models_dir) / "repair"
    
    # Use git lfs to clone
    import subprocess
    
    if repair_dir.exists():
        print(f"✅ {repair_dir} already exists, skipping")
    else:
        subprocess.run([
            "git", "clone",
            "https://huggingface.co/yourusername/GriceBench-Repair",
            str(repair_dir)
        ], check=True)
        print(f"✅ Repair model saved to {repair_dir}")


def download_dpo(models_dir):
    """Download DPO generator"""
    print("\\n📥 Downloading DPO Generator (1.4GB)...")
    
    dpo_dir = Path(models_dir) / "dpo"
    
    # Use git lfs to clone
    import subprocess
    
    if dpo_dir.exists():
        print(f"✅ {dpo_dir} already exists, skipping")
    else:
        subprocess.run([
            "git", "clone",
            "https://huggingface.co/yourusername/GriceBench-DPO",
            str(dpo_dir)
        ], check=True)
        print(f"✅ DPO model saved to {dpo_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download GriceBench models")
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to save models (default: models/)"
    )
    parser.add_argument(
        "--detector",
        action="store_true",
        help="Download detector model only"
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Download repair model only"
    )
    parser.add_argument(
        "--dpo",
        action="store_true",
        help="Download DPO generator only"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models"
    )
    
    args = parser.parse_args()
    
    # Default to all if none specified
    if not (args.detector or args.repair or args.dpo or args.all):
        args.all = True
    
    print("="*60)
    print("GriceBench Model Downloader")
    print("="*60)
    
    if args.all or args.detector:
        download_detector(args.models_dir)
    
    if args.all or args.repair:
        download_repair(args.models_dir)
    
    if args.all or args.dpo:
        download_dpo(args.models_dir)
    
    print("\\n" + "="*60)
    print("✅ Download complete!")
    print("="*60)
    print(f"\\nModels saved to: {Path(args.models_dir).absolute()}")
    print("\\nUsage:")
    print("  python scripts/quick_eval_simple.py --detector models/detector/best_model.pt")


if __name__ == "__main__":
    main()
