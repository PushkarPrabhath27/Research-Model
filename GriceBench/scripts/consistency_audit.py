import os
import sys
import json
import re

PAPER_PATH = "paper/gricebench_main.tex"
EXPECTED = {
    "full_system_cooperative_rate": 0.950,
    "baseline_cooperative_rate": 0.838,
    "detect_repair_cooperative_rate": 0.930,
    "dpo_only_cooperative_rate": 0.832,
    "detector_macro_f1": 0.955,
    "quantity_f1": 1.000,
    "quality_f1": 0.928,
    "relation_f1": 1.000,
    "manner_f1": 0.891,
    "improvement_full_vs_baseline_pp": 0.112,
    "repair_violation_removal_rate": 0.996,
    "mistral_cooperative_rate": 0.891,
    "qwen_cooperative_rate": 0.842,
}

def main():
    print("============================================================")
    print("GRICEBENCH CONSISTENCY AUDIT")
    print("============================================================")
    
    if not os.path.exists(PAPER_PATH):
        print(f"ERROR: {PAPER_PATH} not found.")
        return 1
        
    with open(PAPER_PATH) as f:
        text = f.read()
        
    all_ok = True
    for key, val in EXPECTED.items():
        pct = val * 100
        pattern = f"{pct:.1f}"
        if pattern not in text:
            print(f"[FAIL] {key}: {pattern} not found in paper")
            all_ok = False
        else:
            print(f"[OK] {key}: {pattern}%")
            
    if all_ok:
        print("\n[OK] ALL CHECKS PASSED")
        return 0
    else:
        print("\n[FAIL] CONSISTENCY ERRORS FOUND")
        return 1

if __name__ == "__main__":
    sys.exit(main())
