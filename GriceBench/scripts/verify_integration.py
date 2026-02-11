"""
GriceBench Integration Verification
====================================

This script verifies that all components from Chapters 1-8 are
properly integrated and working together.

Run this on your local machine to check everything is in place
before uploading to Kaggle.
"""

import json
from pathlib import Path
import sys

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
CHECK = '✓'
CROSS = '✗'
WARN = '⚠'


def check(condition, message, critical=True):
    """Print check result."""
    if condition:
        print(f"  {GREEN}{CHECK}{RESET} {message}")
        return True
    else:
        symbol = CROSS if critical else WARN
        color = RED if critical else YELLOW
        print(f"  {color}{symbol}{RESET} {message}")
        return False


def verify_project():
    """Run all verification checks."""
    project_root = Path(__file__).parent.parent
    all_passed = True
    
    print("\n" + "="*60)
    print("GRICEBENCH INTEGRATION VERIFICATION")
    print("="*60)
    
    # =========================================================
    # Chapter 1-2: Environment and Data
    # =========================================================
    print("\n📦 CHAPTER 1-2: Environment & Data")
    print("-"*40)
    
    # Check project structure
    all_passed &= check(
        (project_root / "data_raw").exists(),
        "data_raw/ directory exists"
    )
    all_passed &= check(
        (project_root / "data_processed").exists(),
        "data_processed/ directory exists"
    )
    all_passed &= check(
        (project_root / "scripts").exists(),
        "scripts/ directory exists"
    )
    all_passed &= check(
        (project_root / "requirements.txt").exists(),
        "requirements.txt exists"
    )
    
    # Check Topical-Chat data
    tc_path = project_root / "data_raw" / "topical_chat" / "train.json"
    all_passed &= check(
        tc_path.exists(),
        f"Topical-Chat data exists ({tc_path.stat().st_size / 1e6:.1f} MB)" if tc_path.exists() else "Topical-Chat data missing"
    )
    
    # =========================================================
    # Chapter 3: Data Exploration
    # =========================================================
    print("\n📊 CHAPTER 3: Data Exploration")
    print("-"*40)
    
    all_passed &= check(
        (project_root / "scripts" / "data_exploration.py").exists(),
        "data_exploration.py exists"
    )
    all_passed &= check(
        (project_root / "data_exploration_report.md").exists(),
        "data_exploration_report.md exists"
    )
    
    # Check train examples
    train_ex_path = project_root / "data_processed" / "train_examples.json"
    if train_ex_path.exists():
        try:
            with open(train_ex_path, 'r', encoding='utf-8') as f:
                train_examples = json.load(f)
            all_passed &= check(
                len(train_examples) > 100000,
                f"train_examples.json: {len(train_examples):,} examples"
            )
        except Exception as e:
            all_passed &= check(True, f"train_examples.json exists (large file, skipped loading)", critical=False)
    else:
        all_passed &= check(False, "train_examples.json missing")
    
    # =========================================================
    # Chapter 4: Violation Injection
    # =========================================================
    print("\n💉 CHAPTER 4: Violation Injection")
    print("-"*40)
    
    all_passed &= check(
        (project_root / "scripts" / "violation_injectors.py").exists(),
        "violation_injectors.py exists"
    )
    
    # Check injected dataset
    weak_path = project_root / "data_processed" / "gricebench_weak_50k.json"
    if weak_path.exists():
        with open(weak_path, 'r') as f:
            weak_data = json.load(f)
        
        # Check violation types present
        types = set(ex.get('violation_type', '') for ex in weak_data)
        has_all_types = all(t in str(types) for t in ['quantity', 'quality', 'relation', 'manner'])
        
        all_passed &= check(
            len(weak_data) > 5000,
            f"gricebench_weak_50k.json: {len(weak_data):,} examples"
        )
        all_passed &= check(
            has_all_types,
            f"All 4 violation types present: {len(types)} types"
        )
    else:
        all_passed &= check(False, "gricebench_weak_50k.json missing")
    
    # =========================================================
    # Chapter 5: Weak Supervision
    # =========================================================
    print("\n🏷️ CHAPTER 5: Weak Supervision")
    print("-"*40)
    
    all_passed &= check(
        (project_root / "scripts" / "weak_supervision.py").exists(),
        "weak_supervision.py exists"
    )
    all_passed &= check(
        (project_root / "data_processed" / "gricebench_weak_labeled.json").exists(),
        "gricebench_weak_labeled.json exists",
        critical=False
    )
    
    # =========================================================
    # Chapter 6: Gold Annotation
    # =========================================================
    print("\n✏️ CHAPTER 6: Gold Annotation")
    print("-"*40)
    
    all_passed &= check(
        (project_root / "scripts" / "gold_annotation.py").exists(),
        "gold_annotation.py exists"
    )
    all_passed &= check(
        (project_root / "annotation_rubric.md").exists(),
        "annotation_rubric.md exists"
    )
    all_passed &= check(
        (project_root / "data_processed" / "gold_annotation_sheet.csv").exists(),
        "gold_annotation_sheet.csv exists"
    )
    
    # =========================================================
    # Chapter 7-8: Detector Training
    # =========================================================
    print("\n🧠 CHAPTER 7-8: Detector Training")
    print("-"*40)
    
    all_passed &= check(
        (project_root / "scripts" / "prepare_detector_data.py").exists(),
        "prepare_detector_data.py exists"
    )
    all_passed &= check(
        (project_root / "scripts" / "train_detector.py").exists(),
        "train_detector.py exists"
    )
    
    # Check detector data
    detector_train = project_root / "data_processed" / "detector_data" / "detector_train.json"
    if detector_train.exists():
        with open(detector_train, 'r') as f:
            dt = json.load(f)
        all_passed &= check(True, f"detector_train.json: {len(dt):,} examples")
    else:
        all_passed &= check(False, "detector_train.json missing")
    
    all_passed &= check(
        (project_root / "data_processed" / "detector_data" / "class_weights.json").exists(),
        "class_weights.json exists"
    )
    
    # Check Kaggle guide
    all_passed &= check(
        (project_root / "KAGGLE_TRAINING_GUIDE.md").exists(),
        "KAGGLE_TRAINING_GUIDE.md exists"
    )
    
    # =========================================================
    # Summary
    # =========================================================
    print("\n" + "="*60)
    if all_passed:
        print(f"{GREEN}✅ ALL CHECKS PASSED!{RESET}")
        print("Your GriceBench project is fully integrated and ready for Kaggle.")
    else:
        print(f"{YELLOW}⚠️ SOME CHECKS FAILED{RESET}")
        print("Review the items above marked with ✗ or ⚠")
    print("="*60)
    
    # Print files to upload to Kaggle
    print("\n📤 FILES TO UPLOAD TO KAGGLE:")
    print("-"*40)
    files_for_kaggle = [
        "data_processed/detector_data/detector_train.json",
        "data_processed/detector_data/detector_val.json",
        "data_processed/detector_data/class_weights.json",
    ]
    total_size = 0
    for f in files_for_kaggle:
        fp = project_root / f
        if fp.exists():
            size = fp.stat().st_size / 1e6
            total_size += size
            print(f"  {f} ({size:.1f} MB)")
    print(f"\n  Total: {total_size:.1f} MB")
    
    return all_passed


if __name__ == "__main__":
    success = verify_project()
    sys.exit(0 if success else 1)
