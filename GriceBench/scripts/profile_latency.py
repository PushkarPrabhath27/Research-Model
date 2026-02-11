"""
Profile inference latency of GriceBench models
Measures P50, P95, P99 latencies and throughput
"""

import argparse
import torch
import json
import time
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm


def measure_latency(model, tokenizer, num_samples, batch_size, device):
    """Measure inference latency"""
    latencies = []
    
    # Create sample data
    texts = ["Sample text for latency profiling"] * batch_size
    
    print(f"\\nWarming up...")
    # Warmup
    for _ in range(10):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            _ = model(**inputs)
    
    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    print(f"\\nMeasuring latency over {num_samples} batches...")
    # Measure
    for _ in tqdm(range(num_samples)):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    # Calculate statistics
    latencies = np.array(latencies)
    
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "throughput_samples_per_sec": float((batch_size * num_samples) / (np.sum(latencies) / 1000))
    }


def main():
    parser = argparse.ArgumentParser(description="Profile inference latency")
    parser.add_argument("--model", type=str, default="detector", choices=["detector", "repair", "dpo"])
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of batches to measure")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--output", type=str, default="profiling/latency_profile.json")
    
    args = parser.parse_args()
    
    print("="*60)
    print("LATENCY PROFILING")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num samples: {args.num_samples}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model
    print(f"\\nLoading model...")
    if args.model == "detector":
        from scripts.train_detector import ViolationDetector
        model = ViolationDetector("microsoft/deberta-v3-base")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    elif args.model == "repair":
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
    elif args.model == "dpo":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    
    model = model.to(device)
    model.eval()
    
    # Profile
    results = measure_latency(model, tokenizer, args.num_samples, args.batch_size, device)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_results = {
        "model": args.model,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "device": str(device),
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    # Print summary
    print(f"\\n{'='*60}")
    print("RESULTS")
    print("="*60)
    print(f"  Mean latency: {results['mean_ms']:.2f}ms")
    print(f"  P50 latency: {results['p50_ms']:.2f}ms")
    print(f"  P95 latency: {results['p95_ms']:.2f}ms")
    print(f"  P99 latency: {results['p99_ms']:.2f}ms")
    print(f"  Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
    print(f"\\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
