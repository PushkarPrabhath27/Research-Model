"""
Comprehensive benchmark suite for GriceBench models
Tests multiple models, batch sizes, and generates performance report
"""

import argparse
import torch
import json
import time
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import subprocess


def run_benchmark(model_name, batch_sizes, num_samples=1000):
    """Run benchmark for a model across batch sizes"""
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\\n  Testing batch size {batch_size}...")
        
        # Run latency profiling
        cmd = f"python scripts/profile_latency.py --model {model_name} --batch_size {batch_size} --num_samples {num_samples} --output profiling/temp_{model_name}_{batch_size}.json"
        subprocess.run(cmd, shell=True, capture_output=True)
        
        # Load results
        with open(f"profiling/temp_{model_name}_{batch_size}.json", 'r') as f:
            data = json.load(f)
        
        results[f"batch_{batch_size}"] = data['results']
    
    return results


def generate_markdown_report(all_results, output_path):
    """Generate markdown performance report"""
    lines = []
    
    lines.append("# GriceBench Performance Benchmark Report\\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
    
    lines.append("## Executive Summary\\n\\n")
    lines.append("This report presents comprehensive performance benchmarks for all GriceBench models.\\n\\n")
    
    for model_name, results in all_results.items():
        lines.append(f"### {model_name.upper()} Model\\n\\n")
        
        # Performance table
        lines.append("| Batch Size | P50 Latency | P95 Latency | P99 Latency | Throughput |\\n")
        lines.append("|------------|-------------|-------------|-------------|------------|\\n")
        
        for batch_key, metrics in results.items():
            batch_size = batch_key.split('_')[1]
            lines.append(f"| {batch_size} | {metrics['p50_ms']:.1f}ms | {metrics['p95_ms']:.1f}ms | {metrics['p99_ms']:.1f}ms | {metrics['throughput_samples_per_sec']:.0f} samples/s |\\n")
        
        lines.append("\\n")
        
        # Recommendations
        best_batch = max(results.items(), key=lambda x: x[1]['throughput_samples_per_sec'])
        lines.append(f"**Recommended Batch Size:** {best_batch[0].split('_')[1]} (highest throughput: {best_batch[1]['throughput_samples_per_sec']:.0f} samples/sec)\\n\\n")
    
    lines.append("---\\n\\n")
    lines.append("## Hardware Configuration\\n\\n")
    lines.append(f"- **GPU:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\\n")
    lines.append(f"- **CUDA Version:** {torch.version.cuda if torch.cuda.is_available() else 'N/A'}\\n")
    lines.append(f"- **PyTorch Version:** {torch.__version__}\\n")
    
    # Write report
    with open(output_path, 'w') as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive benchmarks")
    parser.add_argument(
        "--models",
        type=str,
        default="detector,repair,dpo",
        help="Comma-separated list of models to benchmark"
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,8,16,32,64",
        help="Comma-separated batch sizes"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples per benchmark"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/results.json",
        help="Output file"
    )
    
    args = parser.parse_args()
    
    models = args.models.split(',')
    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
    
    print("="*60)
    print("GRICEBENCH BENCHMARK SUITE")
    print("="*60)
    print(f"Models: {models}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Samples per test: {args.num_samples}")
    
    # Create output dir
    Path("profiling").mkdir(exist_ok=True)
    Path("benchmarks").mkdir(exist_ok=True)
    
    # Run benchmarks
    all_results = {}
    
    for model in models:
        print(f"\\n{'='*60}")
        print(f"Benchmarking: {model}")
        print("="*60)
        
        results = run_benchmark(model, batch_sizes, args.num_samples)
        all_results[model] = results
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\\n✅ Results saved to {output_path}")
    
    # Generate report
    report_path = output_path.parent / "report.md"
    generate_markdown_report(all_results, report_path)
    print(f"✅ Report saved to {report_path}")


if __name__ == "__main__":
    main()
