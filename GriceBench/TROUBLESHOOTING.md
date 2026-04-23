# GriceBench Troubleshooting Guide

## Common Issues & Solutions

### Installation Issues

#### Issue: `torch` installation fails

**Symptoms:**
```
ERROR: Could not find a version that satisfies the requirement torch==2.1.0
```

**Solution:**
```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only version
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

---

#### Issue: spaCy model not found

**Symptoms:**
```
OSError: [E050] Can't find model 'en_core_web_sm'
```

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

---

### Model Loading Issues

#### Issue: CUDA out of memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
1. **Reduce batch size:**
   ```python
   # Change from 64 to 16
   batch_size = 16
   ```

2. **Use FP16:**
   ```python
   model = model.half()
   ```

3. **Clear GPU cache:**
   ```python
   import gc
   gc.collect()
   torch.cuda.empty_cache()
   ```

---

#### Issue: Model checkpoint not found

**Symptoms:**
```
FileNotFoundError: models/detector/best_model.pt not found
```

**Solution:**
```bash
# Download models
python scripts/download_models.py --all

# Or manually place checkpoint in models/detector/
```

---

### Inference Issues

#### Issue: Detector returns all zeros

**Symptoms:**
```
violations: {quantity: 0, quality: 0, relation: 0, manner: 0}
probabilities: {quantity: 0.0001, quality: 0.0002, ...}
```

**Solutions:**
1. **Check model weights loaded:**
   ```python
   checkpoint = torch.load("models/detector/best_model.pt", weights_only=False)
   model.load_state_dict(checkpoint['model_state_dict'], strict=True)
   ```

2. **Verify input format:**
   ```python
   input_text = f"[CONTEXT] {context} [RESPONSE] {response}"
   ```

3. **Check threshold:**
   ```python
   # Lower threshold if probabilities are consistently low
   threshold = 0.3
   ```

---

#### Issue: Slow inference

**Symptoms:**
- Latency >500ms for detector

**Solutions:**
1. **Enable FP16:**
   ```python
   model = model.half()
   inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
   ```

2. **Increase batch size:**
   ```python
   batch_size = 64  # Process multiple samples at once
   ```

3. **Use ONNX:**
   ```bash
   python scripts/export_onnx.py --model detector --output models/detector.onnx
   ```

---

### API Issues

#### Issue: API returns 503

**Symptoms:**
```json
{"detail": "Detector model not loaded"}
```

**Solution:**
```bash
# Check logs
docker logs gricebench-api

# Verify model files exist
ls -lh models/detector/best_model.pt

# Restart API
docker-compose restart api
```

---

#### Issue: High API latency

**Symptoms:**
- P95 latency >200ms

**Solutions:**
1. **Enable batching:**
   ```python
   # In api.py, implement dynamic batching
   ```

2. **Use multiple workers:**
   ```bash
   uvicorn api:app --workers 4
   ```

3. **Add caching:**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=10000)
   def detect_cached(text):
       ...
   ```

---

### Training Issues

#### Issue: Training loss not decreasing

**Symptoms:**
- Loss stuck at high value (>0.5)

**Solutions:**
1. **Check learning rate:**
   ```python
   learning_rate = 2e-5  # Standard for DeBERTa
   ```

2. **Verify labels:**
   ```python
   # Print label distribution
   print(labels.sum(axis=0) / len(labels))
   ```

3. **Increase epochs:**
   ```python
   epochs = 10  # Instead of 3
   ```

---

#### Issue: Overfitting

**Symptoms:**
- Train loss < 0.1, val loss > 0.3

**Solutions:**
1. **Add dropout:**
   ```python
   dropout = 0.2  # Instead of 0.1
   ```

2. **Use early stopping:**
   ```python
   patience = 3  # Stop if no improvement for 3 epochs
   ```

3. **Add data augmentation:**
   ```python
   # Back-translation, paraphrasing, etc.
   ```

---

### Docker Issues

#### Issue: Docker build fails

**Symptoms:**
```
ERROR: unable to find 'requirements.txt'
```

**Solution:**
```bash
# Verify file exists
ls requirements.txt

# Build with context
docker build -t gricebench:latest .
```

---

#### Issue: GPU not available in container

**Symptoms:**
```
torch.cuda.is_available() = False
```

**Solution:**
```bash
# Install nvidia-docker2
sudo apt-get install nvidia-docker2

# Restart Docker
sudo systemctl restart docker

# Run with --gpus flag
docker run --gpus all gricebench:latest
```

---

### Data Issues

#### Issue: Data file corrupted

**Symptoms:**
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Solution:**
```bash
# Re-download data
python scripts/download_data.py --all

# Or validate JSON
python -m json.tool data_processed/detector_data/detector_train.json
```

---

#### Issue: Missing data fields

**Symptoms:**
```
KeyError: 'input_text'
```

**Solution:**
```python
# Check data structure
with open('data_processed/detector_data/detector_train.json') as f:
    sample = json.load(f)[0]
    print(list(sample.keys()))

# Ensure expected fields exist
assert 'input_text' in sample
assert 'labels' in sample
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profile GPU Memory

```bash
python scripts/profile_memory.py --model detector --batch_sizes 1,8,16
```

### Profile Latency

```bash
python scripts/profile_latency.py --model detector --num_samples 1000 --batch_size 16
```

### Check CUDA Version

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
```

### Verify Model Outputs

```python
# Test with known input
test_input = "[CONTEXT] Hello [RESPONSE] Hi there"
inputs = tokenizer(test_input, return_tensors="pt")
outputs = model(**inputs)
print(outputs['probs'])
# Should be 4 probabilities between 0-1
```

---

## Getting Help

If you continue to experience issues:

1. **Check GitHub Issues:** [github.com/yourusername/GriceBench/issues](https://github.com/yourusername/GriceBench/issues)
2. **Discussions:** [github.com/yourusername/GriceBench/discussions](https://github.com/yourusername/GriceBench/discussions)
3. **Email:** your.email@university.edu

**When reporting issues, include:**
- Python version
- PyTorch version
- CUDA version (if using GPU)
- Full error traceback
- Steps to reproduce

---

**Last Updated:** 2026-01-23
