# GriceBench Performance Optimization Guide

## Overview

This guide covers performance optimization techniques for deploying GriceBench models in production.

## Table of Contents

1. [Profiling & Analysis](#profiling--analysis)
2. [Model Quantization](#model-quantization)
3. [ONNX Export](#onnx-export)
4. [TensorRT Optimization](#tensorrt-optimization)
5. [Batch Processing](#batch-processing)
6. [Caching Strategies](#caching-strategies)
7. [Hardware Selection](#hardware-selection)

---

## Profiling & Analysis

### Memory Profiling

```bash
# Profile GPU memory usage
python scripts/profile_memory.py \\
    --model detector \\
    --batch_sizes 1,8,16,32,64 \\
    --output profiling/memory_profile.json
```

**Expected Output:**
```json
{
  "detector": {
    "batch_1": {"gpu_memory_mb": 2048, "cpu_memory_mb": 1024},
    "batch_8": {"gpu_memory_mb": 2560, "cpu_memory_mb": 1280},
    "batch_16": {"gpu_memory_mb": 3072, "cpu_memory_mb": 1536}
  }
}
```

### Latency Profiling

```bash
# Profile inference latency
python scripts/profile_latency.py \\
    --model detector \\
    --num_samples 1000 \\
    --batch_size 16 \\
    --output profiling/latency_profile.json
```

**Key Metrics:**
- **P50 Latency**: Median inference time
- **P95 Latency**: 95th percentile
- **P99 Latency**: 99th percentile
- **Throughput**: Samples per second

### PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True
) as prof:
    # Run inference
    outputs = model(inputs)

# Print report
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export Chrome trace
prof.export_chrome_trace("trace.json")
# View at chrome://tracing
```

---

## Model Quantization

### Dynamic Quantization (Fastest)

**Detector:**
```python
import torch

# Load model
detector = ViolationDetector.from_pretrained("models/detector")

# Apply dynamic quantization
quantized_detector = torch.quantization.quantize_dynamic(
    detector.encoder,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save
torch.save(quantized_detector.state_dict(), "models/detector_int8.pt")
```

**Performance Gain:**
- **Latency**: 1.5-2x faster
- **Memory**: 2-4x smaller
- **Accuracy**: <1% degradation

### Static Quantization (Most Accurate)

```python
from torch.quantization import quantize_static

# Calibration dataset
calibration_data = load_calibration_data(num_samples=500)

# Prepare model
detector.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(detector, inplace=True)

# Calibrate
with torch.no_grad():
    for batch in calibration_data:
        detector(batch['input_ids'], batch['attention_mask'])

# Quantize
quantized_detector = torch.quantization.convert(detector, inplace=False)
```

### INT8 vs FP16 Comparison

| Precision | Latency (ms) | Memory (GB) | F1 Score |
|-----------|--------------|-------------|----------|
| FP32 | 42 | 2.9 | 0.968 |
| FP16 | 28 | 1.5 | 0.967 |
| INT8 | 18 | 0.8 | 0.962 |

---

## ONNX Export

### Export Detector

```python
import torch
import onnx

# Load PyTorch model
detector = ViolationDetector.from_pretrained("models/detector")
detector.eval()

# Dummy input
batch_size = 1
seq_length = 128
dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
dummy_attention_mask = torch.ones(batch_size, seq_length)

# Export
torch.onnx.export(
    detector,
    (dummy_input_ids, dummy_attention_mask),
    "models/detector.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits', 'probs'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size'},
        'probs': {0: 'batch_size'}
    }
)

# Verify
onnx_model = onnx.load("models/detector.onnx")
onnx.checker.check_model(onnx_model)
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np

# Create session
session = ort.InferenceSession(
    "models/detector.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Prepare inputs
inputs = {
    'input_ids': input_ids.numpy(),
    'attention_mask': attention_mask.numpy()
}

# Run inference
outputs = session.run(None, inputs)
logits, probs = outputs
```

**Performance:**
- **Latency**: 20-30% faster than PyTorch
- **Memory**: 10-15% lower
- **Compatibility**: CPU, GPU, TensorRT, OpenVINO

---

## TensorRT Optimization

### Prerequisites

```bash
# Install TensorRT
pip install nvidia-tensorrt==8.6.1

# Verify
python -c "import tensorrt; print(tensorrt.__version__)"
```

### Convert ONNX to TensorRT

```bash
# Using trtexec
trtexec \\
    --onnx=models/detector.onnx \\
    --saveEngine=models/detector.trt \\
    --fp16 \\
    --batch=16 \\
    --workspace=4096 \\
    --verbose
```

### TensorRT Python API

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Build engine
with trt.Builder(TRT_LOGGER) as builder, \\
     builder.create_network() as network, \\
     trt.OnnxParser(network, TRT_LOGGER) as parser:
    
    # Parse ONNX
    with open("models/detector.onnx", 'rb') as model:
        parser.parse(model.read())
    
    # Build config
    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30  # 4GB
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Build engine
    engine = builder.build_engine(network, config)
    
    # Serialize
    with open("models/detector.trt", "wb") as f:
        f.write(engine.serialize())
```

**Performance Gain:**
- **Latency**: 3-5x faster than PyTorch
- **Throughput**: 4-6x higher
- **GPU Utilization**: 80-95%

---

## Batch Processing

### Optimal Batch Sizes

| Model | GPU | Optimal Batch | Throughput |
|-------|-----|---------------|------------|
| Detector | T4 | 64 | 450 samples/sec |
| Detector | V100 | 128 | 920 samples/sec |
| Repair | T4 | 32 | 180 samples/sec |
| DPO | T4 | 16 | 25 samples/sec |

### Dynamic Batching

```python
class DynamicBatcher:
    def __init__(self, model, max_batch_size=64, max_wait_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = []
    
    def add(self, input_data):
        self.queue.append(input_data)
        
        # Batch when full or timeout
        if len(self.queue) >= self.max_batch_size:
            return self.process_batch()
        
        # Wait for more samples
        time.sleep(self.max_wait_ms / 1000)
        return self.process_batch()
    
    def process_batch(self):
        if not self.queue:
            return []
        
        batch = self.queue[:self.max_batch_size]
        self.queue = self.queue[self.max_batch_size:]
        
        # Run model
        outputs = self.model(batch)
        return outputs
```

---

## Caching Strategies

### Response Caching

```python
from functools import lru_cache
import hashlib

class CachedDetector:
    def __init__(self, model, cache_size=10000):
        self.model = model
        self.cache = {}
        self.max_size = cache_size
    
    def _hash_input(self, text):
        return hashlib.md5(text.encode()).hexdigest()
    
    @lru_cache(maxsize=10000)
    def detect(self, text):
        cache_key = self._hash_input(text)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Run model
        result = self.model(text)
        
        # Cache result
        if len(self.cache) < self.max_size:
            self.cache[cache_key] = result
        
        return result
```

**Cache Hit Rate:** 30-60% in production (depends on domain)

### KV Cache for Generation

```python
# Enable past_key_values caching for GPT-2
outputs = model.generate(
    **inputs,
    use_cache=True,  # Enable KV cache
    max_new_tokens=100
)
```

**Performance:**
- **Latency**: 2-3x faster for long sequences
- **Memory**: Trades memory for speed

---

## Hardware Selection

### GPU Comparison

| GPU | Memory | FP32 TFLOPS | FP16 TFLOPS | Price/Hour (AWS) |
|-----|--------|-------------|-------------|------------------|
| T4 | 16GB | 8.1 | 65 | $0.526 |
| V100 | 32GB | 14 | 112 | $3.06 |
| A10G | 24GB | 31.2 | 125 | $1.006 |
| A100 | 80GB | 19.5 | 312 | $4.10 |

### Recommended Hardware

**Development:**
- **CPU**: 8 cores, 32GB RAM
- **GPU**: T4 or RTX 3090
- **Storage**: 100GB SSD

**Production (Low Latency):**
- **GPU**: A10G or A100
- **Multi-GPU**: 2-4x T4 with load balancing

**Production (High Throughput):**
- **GPU**: 4-8x T4 with batch processing
- **Load Balancer**: NGINX or Kubernetes

---

## Performance Tuning Checklist

### Model-Level Optimizations

- [ ] FP16 mixed precision
- [ ] Dynamic/static quantization
- [ ] ONNX export
- [ ] TensorRT engine
- [ ] KV caching (for generation)

### System-Level Optimizations

- [ ] Optimal batch size
- [ ] Multi-GPU inference
- [ ] Response caching
- [ ] Async processing
- [ ] Load balancing

### Deployment Optimizations

- [ ] Docker multi-stage builds
- [ ] Model preloading
- [ ] Warm-up requests
- [ ] Monitoring & alerts
- [ ] Auto-scaling

---

## Benchmarking

### Run Benchmark Suite

```bash
# Full benchmark
python scripts/benchmark.py \\
    --models detector,repair,dpo \\
    --batch_sizes 1,8,16,32,64 \\
    --num_samples 1000 \\
    --output benchmarks/results.json

# Generate report
python scripts/generate_benchmark_report.py \\
    --input benchmarks/results.json \\
    --output benchmarks/report.md
```

### Expected Results

**Detector (T4 GPU, FP16):**
- Batch 1: 2.2ms latency, 450 samples/sec
- Batch 16: 22ms latency, 727 samples/sec
- Batch 64: 78ms latency, 820 samples/sec

**Full Pipeline (T4 GPU):**
- Latency: 83ms (p50), 120ms (p99)
- Throughput: 12 samples/sec
- GPU Utilization: 65%

---

## Troubleshooting

### High Latency

1. **Profile bottleneck**: Use PyTorch profiler
2. **Check batch size**: Increase for better GPU utilization
3. **Enable FP16**: 1.5-2x speedup
4. **Use ONNX/TensorRT**: 2-5x speedup

### Out of Memory

1. **Reduce batch size**: Start with 1 and increase
2. **Enable gradient checkpointing**: For training
3. **Quantize model**: INT8 uses 4x less memory
4. **Use model parallelism**: Split across GPUs

### Low GPU Utilization

1. **Increase batch size**: Up to GPU memory limit
2. **Reduce CPU preprocessing**: Bottleneck analysis
3. **Use DataLoader workers**: Parallel data loading
4. **Profile data pipeline**: Check I/O speed

---

**Last Updated:** 2026-01-23
