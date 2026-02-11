"""
Profile memory usage of GriceBench models
Measures GPU and CPU memory consumption across different batch sizes
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import psutil
import gc


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def get_cpu_memory():
    """Get current CPU memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024**2


def profile_model(model_name, model, tokenizer, batch_sizes, device):
    """Profile model across different batch sizes"""
    results = {}
    
    for batch_size in batch_sizes:
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create dummy batch
        texts = ["Sample text for profiling memory usage"] * batch_size
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        
        # Initial memory
        initial_gpu = get_gpu_memory()
        initial_cpu = get_cpu_memory()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Final memory
        final_gpu = get_gpu_memory()
        final_cpu = get_cpu_memory()
        peak_gpu = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        results[f"batch_{batch_size}"] = {
            "batch_size": batch_size,
            "gpu_memory_mb": round(final_gpu, 2),
            "gpu_memory_peak_mb": round(peak_gpu, 2),
            "gpu_memory_delta_mb": round(final_gpu - initial_gpu, 2),
            "cpu_memory_mb": round(final_cpu, 2),
            "cpu_memory_delta_mb": round(final_cpu - initial_cpu, 2)
        }
        
        print(f"  Batch {batch_size}: GPU={final_gpu:.0f}MB, CPU={final_cpu:.0f}MB")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Profile memory usage")
    parser.add_argument(
        "--model",
        type=str,
        default="detector",
        choices=["detector", "repair", "dpo"],
        help="Model to profile"
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,8,16,32,64",
        help="Comma-separated batch sizes to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profiling/memory_profile.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
    
    print("="*60)
    print("MEMORY PROFILING")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"\\nLoading {args.model}...")
    
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
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Profile
    print(f"\\nProfiling across batch sizes...")
    results = profile_model(args.model, model, tokenizer, batch_sizes, device)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_results = {
        "model": args.model,
        "device": str(device),
        "results": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\\n✅ Results saved to {output_path}")
    
    # Summary
    print(f"\\nSummary:")
    print(f"  Min GPU memory: {min(r['gpu_memory_mb'] for r in results.values()):.0f}MB (batch 1)")
    print(f"  Max GPU memory: {max(r['gpu_memory_mb'] for r in results.values()):.0f}MB (batch {max(batch_sizes)})")


if __name__ == "__main__":
    main()
