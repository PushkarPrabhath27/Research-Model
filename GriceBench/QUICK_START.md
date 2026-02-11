# GriceBench Quick Start Guide

Get started with GriceBench in 5 minutes!

---

## Prerequisites

- **Python 3.10+**
- **CUDA 11.8+** (optional, for GPU acceleration)
- **16GB RAM** (32GB recommended)

---

## Installation (2 minutes)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/GriceBench.git
cd GriceBench

# 2. Create virtual environment
python -m venv grice_env
source grice_env/bin/activate  # Windows: grice_env\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm
```

---

## Quick Evaluation (3 minutes)

### Option 1: Run with Sample Data (Fastest)

```bash
# Download sample data
python scripts/download_data.py --sample

# Run quick evaluation
python scripts/quick_eval_simple.py --num_samples 10
```

**Expected Output:**
```
Cooperative Rate: ~95%
Violations: Q=4%, Ql=0%, R=0%, M=16%
```

### Option 2: With Pre-trained Models

```bash
# Download models (~4GB)
python scripts/download_models.py --all

# Run evaluation
python scripts/quick_eval_simple.py \\
    --detector models/detector/best_model.pt \\
    --num_samples 100
```

---

## API Server (1 minute)

```bash
# Start API server
uvicorn scripts.api:app --reload --port 8000

# Test in another terminal
curl http://localhost:8000/health

# Interactive docs
open http://localhost:8000/docs
```

---

## Docker Deployment (1 minute)

```bash
# Build image
docker build -t gricebench:latest .

# Run container
docker run --gpus all -p 8000:8000 gricebench:latest

# Or with docker-compose
docker-compose up -d
```

---

## Example Usage

### Detect Violations

```python
from scripts.train_detector import ViolationDetector
from transformers import AutoTokenizer

# Load model
detector = ViolationDetector.from_pretrained("models/detector")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# Prepare input
context = "What is quantum computing?"
response = "Quantum computing uses qubits for processing."
text = f"[CONTEXT] {context} [RESPONSE] {response}"

# Detect
inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = detector(inputs['input_ids'], inputs['attention_mask'])

print(f"Probabilities: {outputs['probs']}")
# Output: tensor([[0.023, 0.015, 0.008, 0.031]])
# [Quantity, Quality, Relation, Manner]
```

### API Request

```bash
curl -X POST http://localhost:8000/detect \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your-api-key-here" \\
  -d '{
    "context": "What is AI?",
    "response": "AI is artificial intelligence",
    "threshold": 0.5
  }'
```

---

## Next Steps

### Reproduce Experiments

1. **Part 1-2 (Local):**
   ```bash
   python scripts/run_all_experiments.py --parts 1,2
   ```

2. **Part 3-5 (Kaggle):**
   - Upload notebooks from `kaggle_notebooks/`
   - Follow instructions in README.md

### Train Models

```bash
# Train detector
python scripts/train_detector.py \\
    --train_data data_processed/detector_data/detector_train.json \\
    --val_data data_processed/detector_data/detector_val.json \\
    --output_dir models/detector

# Train repair
python scripts/train_repair.py \\
    --train_data data_processed/repair_data/repair_train.json \\
    --output_dir models/repair
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=scripts --cov-report=html
```

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python scripts/quick_eval_simple.py --batch_size 8
```

### Model Not Found

```bash
# Download models
python scripts/download_models.py --all
```

### See Full Guide

For detailed troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## Resources

- **Full Documentation:** [README.md](README.md)
- **API Docs:** http://localhost:8000/docs
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Issues:** https://github.com/yourusername/GriceBench/issues

---

**Need Help?**

- GitHub Discussions: [/discussions](https://github.com/yourusername/GriceBench/discussions)
- Email: your.email@university.edu

---

**Last Updated:** 2026-01-23
