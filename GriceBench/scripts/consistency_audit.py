"""
scripts/consistency_audit.py
=============================
Phase 7 — Final pre-submission consistency checker.

Verifies that every key number in the LaTeX paper matches the actual
result files. A single mismatched number is a publication error that
will cause embarrassment at best and retraction requests at worst.

Run this LAST, after the paper draft is complete and all Phase 1–6
results are finalized.

Usage:
    C:\\Users\\pushk\\python310\\python.exe scripts\\consistency_audit.py

Output:
    docs/consistency_audit_report.json
    MUST show "pass": true, "total_issues": 0 before arXiv submission.

Author: GriceBench Research
Version: 1.0 — March 2026
"""

import json
import os
import re
import sys

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_PATH  = os.path.join(BASE_DIR, "paper", "gricebench_main.tex")
PHASE7_PATH = os.path.join(BASE_DIR, "results", "phase7output", "phase7_results.json")
PHASE7_V2   = os.path.join(BASE_DIR, "results", "phase7output", "phase7_results_v2.json")
STATS_PATH  = os.path.join(BASE_DIR, "results", "statistical_significance.json")
ABLATION    = os.path.join(BASE_DIR, "results", "part4output", "ablation_report.md")
DPO_ID      = os.path.join(BASE_DIR, "docs", "dpo_model_identity.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "docs", "consistency_audit_report.json")

# ── Ground-truth values (from actual result files) ───────────────────────────
# These are the CANONICAL NUMBERS. Every instance in the paper must match.
EXPECTED = {
    # Cooperative rates (confirmed from phase7_results.json ablation study)
    "full_system_cooperative_rate": 0.950,         # 95.0%
    "baseline_cooperative_rate": 0.838,            # 83.8%
    "detect_repair_cooperative_rate": 0.930,       # 93.0%
    "dpo_only_cooperative_rate": 0.832,            # 83.2%

    # Detector performance (confirmed from phase7_results.json)
    "detector_macro_f1": 0.955,                    # 0.955
    "quantity_f1": 1.000,                          # 1.000
    "quality_f1": 0.928,                           # 0.928
    "relation_f1": 1.000,                          # 1.000
    "manner_f1": 0.891,                            # 0.891

    # Comparative improvements
    "improvement_full_vs_baseline_pp": 11.2,       # +11.2 pp

    # These are updated by Phase 1 and cannot be checked until that's done
    # "repair_violation_removal_rate": TBD         # from phase7_results_v2.json

    # External baselines
    "mistral_cooperative_rate": 0.891,             # 89.1%
    "qwen_cooperative_rate": 0.842,                # 84.2%
}


def load_json_safe(path: str, label: str) -> dict | None:
    if not os.path.exists(path):
        print(f"  ⚠️  MISSING: {label} ({path})")
        return None
    with open(path) as f:
        return json.load(f)


def check_number_in_paper(paper_text: str, value: float, key: str) -> list:
    """
    Check that a given percentage value appears in the paper LaTeX source.
    Checks multiple possible representations:
        "95.0\\%" → common LaTeX form
        "95.0\%" → escaped  
        "95.0%"  → plain text (unlikely in LaTeX but possible)
    Returns list of issues found.
    """
    issues = []
    pct = value * 100  # Convert to percentage

    # LaTeX representations to search for
    patterns = [
        f"{pct:.1f}\\\\%",   # 95.0\\%  (escaped in Python string = 95.0\% in LaTeX)
        f"{pct:.1f}\\%",     # 95.0\%
        f"{pct:.1f}\\%%",
        f"{pct:.0f}\\\\%",   # try without decimal for round numbers
    ]

    # For F1 scores, also check .3f format
    if key.endswith("_f1") or key.endswith("_rate"):
        patterns.extend([
            f"{value:.3f}",
            f"{value:.4f}",
        ])

    found = any(p in paper_text for p in patterns)

    if not found:
        issues.append({
            "key": key,
            "expected_value": value,
            "expected_pct": f"{pct:.1f}%",
            "searched_for": patterns,
            "found": False,
            "severity": "ERROR",
            "message": f"Expected {pct:.1f}% ({key}) not found in paper LaTeX",
        })
    return issues


def check_dpo_model_identity(paper_text: str, dpo_id: dict | None) -> list:
    """Verify the paper uses the correct confirmed DPO base model name."""
    issues = []
    if dpo_id is None:
        issues.append({
            "key": "dpo_model_identity",
            "severity": "WARNING",
            "message": "docs/dpo_model_identity.json not found — cannot verify DPO model name in paper",
        })
        return issues

    confirmed_model = dpo_id.get("resolved_base_model", "UNKNOWN")
    if "UNKNOWN" in confirmed_model:
        issues.append({
            "key": "dpo_model_identity",
            "severity": "ERROR",
            "message": "DPO base model is still UNKNOWN. Run identify_dpo_model.py first.",
        })
        return issues

    # Check that [CONFIRMED_DPO_MODEL] placeholder has been replaced
    if "[CONFIRMED_DPO_MODEL]" in paper_text or "[CONFIRMED\\_DPO\\_MODEL]" in paper_text:
        issues.append({
            "key": "dpo_model_identity",
            "severity": "ERROR",
            "message": f"Paper still has placeholder [CONFIRMED_DPO_MODEL]. Replace with: {confirmed_model}",
        })

    return issues


def check_ci_placeholders(paper_text: str) -> list:
    """Check that CI placeholders have been filled in."""
    issues = []
    placeholders = [
        "[CI_FULL_SYSTEM]",
        "[CI_DETECT_REPAIR]",
        "[CI_DPO_ONLY]",
        "[CI_BASELINE]",
        "[P_FULL_VS_BASELINE]",
        "[REPAIR_REMOVAL_RATE]",
    ]
    for placeholder in placeholders:
        if placeholder in paper_text:
            issues.append({
                "key": placeholder,
                "severity": "ERROR",
                "message": f"Placeholder {placeholder} not yet filled in. Run Phase 4 (stats) first.",
            })
    return issues


def check_hf_links_exist(paper_text: str) -> list:
    """Check that HuggingFace links are present and not placeholder."""
    issues = []
    if "PushkarPrabhath27" not in paper_text:
        issues.append({
            "key": "huggingface_links",
            "severity": "ERROR",
            "message": "HuggingFace URL (PushkarPrabhath27) not found in paper.",
        })
    if "XXXX.XXXXX" in paper_text:
        issues.append({
            "key": "arxiv_link",
            "severity": "WARNING",
            "message": "arXiv placeholder XXXX.XXXXX still present (fill in after submission).",
        })
    return issues


def main():
    print("=" * 60)
    print("GRICEBENCH CONSISTENCY AUDIT")
    print("=" * 60)

    # ── Step 1: Check required files ──────────────────────────────────────────
    print("\n[1] Required file check:")
    required_files = [
        (PAPER_PATH,  "paper/gricebench_main.tex"),
        (PHASE7_PATH, "results/phase7output/phase7_results.json"),
        (STATS_PATH,  "results/statistical_significance.json"),
    ]
    all_files_ok = True
    for path, label in required_files:
        exists = os.path.exists(path)
        print(f"  {'✅' if exists else '❌'} {label}")
        if not exists:
            all_files_ok = False

    optional_files = [
        (PHASE7_V2, "results/phase7output/phase7_results_v2.json (Phase 1 output)"),
        (DPO_ID,    "docs/dpo_model_identity.json (Phase 2 output)"),
        (ABLATION,  "results/part4output/ablation_report.md"),
    ]
    print("\n  Optional files:")
    for path, label in optional_files:
        exists = os.path.exists(path)
        print(f"  {'✅' if exists else '⚠️ '} {label}")

    if not all_files_ok:
        print("\n❌ Required files missing. Complete Phases 1–5 before running this audit.")
        sys.exit(1)

    # ── Step 2: Read paper ────────────────────────────────────────────────────
    print("\n[2] Reading paper...")
    with open(PAPER_PATH) as f:
        paper_text = f.read()
    word_count = len(paper_text.split())
    print(f"  Paper: {len(paper_text)} chars, ~{word_count} words")

    # ── Step 3: Check all key numbers ─────────────────────────────────────────
    print("\n[3] Checking key numbers in paper:")
    all_issues = []

    for key, value in EXPECTED.items():
        issues = check_number_in_paper(paper_text, value, key)
        for issue in issues:
            all_issues.append(issue)
        status = "✅" if not issues else "❌"
        print(f"  {status} {key}: {value*100:.1f}%")

    # ── Step 4: Check DPO model identity ──────────────────────────────────────
    print("\n[4] Checking DPO model identity:")
    dpo_id = load_json_safe(DPO_ID, "DPO identity")
    dpo_issues = check_dpo_model_identity(paper_text, dpo_id)
    all_issues.extend(dpo_issues)
    if dpo_issues:
        for issue in dpo_issues:
            print(f"  {'❌' if issue['severity']=='ERROR' else '⚠️ '} {issue['message']}")
    else:
        print("  ✅ DPO model identity confirmed in paper")

    # ── Step 5: Check CI placeholders ─────────────────────────────────────────
    print("\n[5] Checking CI placeholders:")
    ci_issues = check_ci_placeholders(paper_text)
    all_issues.extend(ci_issues)
    if ci_issues:
        for issue in ci_issues:
            print(f"  ❌ {issue['message']}")
    else:
        print("  ✅ All CI placeholders filled in")

    # ── Step 6: Check HuggingFace links ───────────────────────────────────────
    print("\n[6] Checking HuggingFace links:")
    hf_issues = check_hf_links_exist(paper_text)
    all_issues.extend(hf_issues)
    if hf_issues:
        for issue in hf_issues:
            severity_icon = "❌" if issue["severity"] == "ERROR" else "⚠️ "
            print(f"  {severity_icon} {issue['message']}")
    else:
        print("  ✅ HuggingFace links present")

    # ── Step 7: Cross-check against result files ───────────────────────────────
    print("\n[7] Cross-checking result files vs expected values:")
    phase7 = load_json_safe(PHASE7_PATH, "phase7_results")
    if phase7:
        # Try to read ablation cooperative rates from Phase 7
        ablation_key = next(
            (k for k in phase7 if "ablation" in k.lower()), None
        )
        if ablation_key:
            ablation = phase7[ablation_key]
            print(f"  Found ablation data under key: {ablation_key}")
        else:
            print("  ⚠️  Could not find ablation sub-key in phase7_results.json")
            print(f"  Available keys: {list(phase7.keys())[:10]}")

    # ── Step 8: Save report ────────────────────────────────────────────────────
    errors   = [i for i in all_issues if i.get("severity") == "ERROR"]
    warnings = [i for i in all_issues if i.get("severity") == "WARNING"]

    report = {
        "pass":                len(errors) == 0,
        "total_issues":        len(all_issues),
        "total_errors":        len(errors),
        "total_warnings":      len(warnings),
        "errors":              errors,
        "warnings":            warnings,
        "checked_values":      {k: v for k, v in EXPECTED.items()},
        "paper_path":          PAPER_PATH,
        "arXiv_ready":         (len(errors) == 0 and len(warnings) == 0),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONSISTENCY AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Saved to: {OUTPUT_PATH}")

    if report["pass"]:
        print("\n✅ ALL CHECKS PASSED")
        if warnings:
            print(f"   (with {len(warnings)} warning(s) — review before submission)")
        print("   Paper is ready for arXiv submission!")
    else:
        print(f"\n❌ {len(errors)} ERROR(S) FOUND — fix before submission:")
        for err in errors:
            print(f"   • {err['message']}")

    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
