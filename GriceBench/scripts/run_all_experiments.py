"""
Run complete GriceBench experiment reproduction
Executes all 5 parts of the paper experiments in sequence
"""

import subprocess
import argparse
import json
import time
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Execute command and log results"""
    print(f"\\n{'='*70}")
    print(f"📍 {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✅ SUCCESS (took {duration:.1f}s)")
        print(result.stdout)
        return True
    else:
        print(f"❌ FAILED (after {duration:.1f}s)")
        print(result.stderr)
        return False


def part1_relation_repair():
    """Part 1: Build relation repair system"""
    print("\\n" + "="*70)
    print("PART 1: RELATION REPAIR SYSTEM")
    print("="*70)
    
    # Create corpus
    success = run_command(
        "python scripts/create_response_corpus.py --data_dir data_processed --output data_processed/relation_repair/response_corpus.json",
        "Creating response corpus"
    )
    if not success:
        return False
    
    # Build FAISS index
    success = run_command(
        "python scripts/build_retrieval_system.py --corpus data_processed/relation_repair/response_corpus.json --output data_processed/relation_repair/",
        "Building FAISS retrieval index"
    )
    if not success:
        return False
    
    # Evaluate
    success = run_command(
        "python scripts/evaluate_relation_repair.py --retrieval_dir data_processed/relation_repair --output results/part1_relation_repair.json",
        "Evaluating relation repair"
    )
    
    return success


def part2_human_eval():
    """Part 2: Prepare human evaluation"""
    print("\\n" + "="*70)
    print("PART 2: HUMAN EVALUATION SETUP")
    print("="*70)
    
    # Prepare samples
    success = run_command(
        "python scripts/prepare_human_eval_samples.py --num_samples 100 --output data_processed/human_eval/samples_blinded.json",
        "Preparing human evaluation samples"
    )
    
    if success:
        print("\\n📋 Human evaluation interface ready!")
        print("To start annotation, run:")
        print("  python scripts/human_eval_gradio.py --samples data_processed/human_eval/samples_blinded.json --port 7860")
    
    return success


def part3_baselines():
    """Part 3: Baseline comparisons (Kaggle only)"""
    print("\\n" + "="*70)
    print("PART 3: BASELINE COMPARISONS (KAGGLE)")
    print("="*70)
    
    print("⚠️  This step requires Kaggle (2 hours on T4 GPU)")
    print("📋 Instructions:")
    print("  1. Upload kaggle_notebooks/GRICEBENCH_PART_3_BASELINES.ipynb to Kaggle")
    print("  2. Add dataset: gricebench-test-data")
    print("  3. Enable GPU T4")
    print("  4. Run All")
    print("  5. Download results to results/part3_baselines/\\n")
    
    return True  # Manual step


def part4_ablations():
    """Part 4: Ablation studies (Kaggle only)"""
    print("\\n" + "="*70)
    print("PART 4: ABLATION STUDIES (KAGGLE)")
    print("="*70)
    
    print("⚠️  This step requires Kaggle (3 hours on T4 GPU)")
    print("📋 Instructions:")
    print("  1. Upload kaggle_notebooks/GRICEBENCH_PART_4_ABLATIONS.ipynb to Kaggle")
    print("  2. Add datasets:")
    print("     - gricean-maxim-detector-model")
    print("     - gricebench-repair-model")
    print("     - dpo-generator-model")
    print("     - gricebench-test-data")
    print("  3. Enable GPU T4")
    print("  4. Run All")
    print("  5. Download results to results/part4output/\\n")
    
    return True  # Manual step


def part5_error_analysis():
    """Part 5: Error analysis (Kaggle only)"""
    print("\\n" + "="*70)
    print("PART 5: ERROR ANALYSIS (KAGGLE)")
    print("="*70)
    
    print("⚠️  This step requires Kaggle (30 min on T4 GPU)")
    print("📋 Instructions:")
    print("  1. Upload kaggle_notebooks/GRICEBENCH_PART_5_ERROR_ANALYSIS.ipynb to Kaggle")
    print("  2. Add datasets:")
    print("     - gricean-maxim-detector-model")
    print("     - gricebench-detector-validation")
    print("  3. Enable GPU T4")
    print("  4. Run All")
    print("  5. Download results to results/part5output/\\n")
    
    return True  # Manual step


def main():
    parser = argparse.ArgumentParser(description="Run GriceBench experiments")
    parser.add_argument(
        "--parts",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated list of parts to run (default: 1,2,3,4,5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_log.json",
        help="Path to save experiment log"
    )
    
    args = parser.parse_args()
    
    parts = [int(p) for p in args.parts.split(',')]
    
    print("="*70)
    print("GRICEBENCH EXPERIMENT RUNNER")
    print("="*70)
    print(f"Running parts: {parts}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    start_time = time.time()
    
    # Part 1
    if 1 in parts:
        results['part1'] = part1_relation_repair()
    
    # Part 2
    if 2 in parts:
        results['part2'] = part2_human_eval()
    
    # Part 3
    if 3 in parts:
        results['part3'] = part3_baselines()
    
    # Part 4
    if 4 in parts:
        results['part4'] = part4_ablations()
    
    # Part 5
    if 5 in parts:
        results['part5'] = part5_error_analysis()
    
    total_time = time.time() - start_time
    
    # Save log
    log = {
        "timestamp": datetime.now().isoformat(),
        "parts_run": parts,
        "results": results,
        "total_time_seconds": total_time,
        "success_rate": sum(results.values()) / len(results) if results else 0
    }
    
    with open(args.output, 'w') as f:
        json.dump(log, f, indent=2)
    
    # Summary
    print("\\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Parts completed: {sum(results.values())}/{len(results)}")
    print(f"\\nResults saved to: {args.output}")
    
    print("\\n📋 Next steps:")
    print("  1. Complete Kaggle experiments (Parts 3-5)")
    print("  2. Run human evaluation annotation")
    print("  3. Analyze all results with:")
    print("     python scripts/analyze_all_results.py")


if __name__ == "__main__":
    main()
