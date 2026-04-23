"""
Download GriceBench raw datasets
Downloads Wizard of Wikipedia, TopicalChat, and LIGHT datasets
"""

import argparse
import os
import requests
from pathlib import Path
from tqdm import tqdm
import tarfile
import zipfile


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


def extract_archive(archive_path, extract_to):
    """Extract tar.gz or zip archive"""
    print(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_to)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    
    print(f"  ✅ Extracted to {extract_to}")


def download_wizard_of_wikipedia(data_dir):
    """Download Wizard of Wikipedia dataset"""
    print("\n📥 Downloading Wizard of Wikipedia...")
    
    output_dir = Path(data_dir) / "wizard_of_wikipedia"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ParlAI download URLs (example - replace with actual URLs)
    urls = {
        "train": "http://parl.ai/downloads/wizard_of_wikipedia/train.json",
        "test": "http://parl.ai/downloads/wizard_of_wikipedia/test_random_split.json",
        "valid": "http://parl.ai/downloads/wizard_of_wikipedia/valid_random_split.json"
    }
    
    for split, url in urls.items():
        dest = output_dir / f"{split}.json"
        if dest.exists():
            print(f"  ✅ {split}.json already exists, skipping")
        else:
            print(f"  Downloading {split}.json...")
            download_file(url, str(dest))
    
    print("✅ Wizard of Wikipedia download complete")


def download_topicalchat(data_dir):
    """Download TopicalChat dataset"""
    print("\n📥 Downloading TopicalChat...")
    
    output_dir = Path(data_dir) / "topicalchat"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GitHub release URL
    url = "https://github.com/alexa/Topical-Chat/archive/refs/heads/master.zip"
    archive_path = output_dir / "topicalchat.zip"
    
    if archive_path.exists():
        print("  ✅ Archive already exists, skipping download")
    else:
        download_file(url, str(archive_path))
    
    # Extract
    extract_archive(str(archive_path), str(output_dir))
    
    print("✅ TopicalChat download complete")


def download_light(data_dir):
    """Download LIGHT dataset"""
    print("\n📥 Downloading LIGHT...")
    
    output_dir = Path(data_dir) / "light"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ParlAI download (example - replace with actual URL)
    url = "http://parl.ai/downloads/light/light_data.tar.gz"
    archive_path = output_dir / "light_data.tar.gz"
    
    if archive_path.exists():
        print("  ✅ Archive already exists, skipping download")
    else:
        download_file(url, str(archive_path))
    
    # Extract
    extract_archive(str(archive_path), str(output_dir))
    
    print("✅ LIGHT download complete")


def download_sample_data(data_dir):
    """Download small sample dataset for testing"""
    print("\n📥 Downloading sample data...")
    
    output_dir = Path(data_dir) / "sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data (for testing without full download)
    sample_data = [
        {
            "context": ["Hello! How are you?"],
            "response": "I'm doing well, thank you for asking!",
            "speaker": "agent_2"
        }
    ] * 100
    
    import json
    with open(output_dir / "sample_dialogues.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Sample data created")


def main():
    parser = argparse.ArgumentParser(
        description="Download GriceBench datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python scripts/download_data.py --all
  
  # Download only sample data (for testing)
  python scripts/download_data.py --sample
  
  # Download specific dataset
  python scripts/download_data.py --wizard
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_raw",
        help="Directory to save datasets (default: data_raw/)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets"
    )
    parser.add_argument(
        "--wizard",
        action="store_true",
        help="Download Wizard of Wikipedia only"
    )
    parser.add_argument(
        "--topicalchat",
        action="store_true",
        help="Download TopicalChat only"
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Download LIGHT only"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Download sample data for testing"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Same as --all (for compatibility)"
    )
    
    args = parser.parse_args()
    
    # Default to sample if nothing specified
    if not (args.all or args.wizard or args.topicalchat or args.light or args.sample or args.raw):
        args.sample = True
    
    if args.raw:
        args.all = True
    
    print("="*60)
    print("GRICEBENCH DATA DOWNLOADER")
    print("="*60)
    print(f"Output directory: {args.data_dir}")
    
    # Create data directory
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    if args.sample:
        download_sample_data(args.data_dir)
    
    if args.all or args.wizard:
        try:
            download_wizard_of_wikipedia(args.data_dir)
        except Exception as e:
            print(f"⚠️ Wizard download failed: {e}")
            print("   You may need to manually download from: https://parl.ai/projects/wizard_of_wikipedia/")
    
    if args.all or args.topicalchat:
        try:
            download_topicalchat(args.data_dir)
        except Exception as e:
            print(f"⚠️ TopicalChat download failed: {e}")
            print("   You may need to manually download from: https://github.com/alexa/Topical-Chat")
    
    if args.all or args.light:
        try:
            download_light(args.data_dir)
        except Exception as e:
            print(f"⚠️ LIGHT download failed: {e}")
            print("   You may need to manually download from: https://parl.ai/projects/light/")
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("="*60)
    print(f"\nData saved to: {Path(args.data_dir).absolute()}")
    print("\nNext steps:")
    print("  1. Preprocess data: python scripts/prepare_detector_data.py")
    print("  2. Train detector: python scripts/train_detector.py")


if __name__ == "__main__":
    main()
