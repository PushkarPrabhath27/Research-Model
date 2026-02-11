# GriceBench: Operationalizing Gricean Maxims for Dialogue Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**GriceBench** is a comprehensive framework for training, evaluating, and deploying dialogue systems that adhere to Gricean Maxims of Cooperative Conversation.

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Experiment Reproduction](#experiment-reproduction)
- [Project Structure](#project-structure)
- [Model Zoo](#model-zoo)
- [Citation](#citation)

---

## 🎯 Overview

GriceBench implements a three-component system for cooperative dialogue:

1. **Detector** (DeBERTa-v3-base): Multi-label classifier for 4 Gricean maxims
2 **Repair** (T5): Edits responses to fix violations of Quantity, Quality, and Manner
3. **Generator** (GPT-2 + DPO): Preference-optimized language model

### Key Results

| Configuration | Cooperative Rate | Improvement |
|---------------|------------------|-------------|
| Baseline (GPT-2) | 83.8% | - |
| DPO Only | 83.2% | -0.6pp |
| Detect + Repair | 93.0% | +9.2pp |
| **Full System** | **95.0%** | **+11.2pp** |

### Detector Performance (500 validation examples)

| Maxim | F1 Score | Precision | Recall |
|-------|----------|-----------|--------|
| Quantity | 1.000 | 100% | 100% |
| Quality | 0.930 | 91.7% | 94.3% |
| Relation | 1.000 | 100% | 100% |
| Manner | 0.940 | 94.9% | 93.1% |
| **Overall** | **0.968** | **96.6%** | **96.9%** |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+ required
python --version

# CUDA 11.8+ for GPU acceleration (optional but recommended)
nvidia-smi
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/PushkarPrabhath27/Research-Model.git
cd Research-Model/GriceBench

# 2. Create virtual environment
python -m venv grice_env
source grice_env/bin/activate  # On Windows: grice_env\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model (required)
python -m spacy download en_core_web_sm
```

### Download Pretrained Models

```bash
# Download all three models (~4GB total)
python scripts/download_models.py --all

# Or download individually:
python scripts/download_models.py --detector  # ~735MB
python scripts/download_models.py --repair    # ~900MB
python scripts/download_models.py --dpo       # ~2.5GB
```

### Quick Evaluation

```bash
# Run the full system on 100 test examples
python scripts/quick_eval_simple.py \\
    --detector models/detector/best_model.pt \\
    --repair models/repair \\
    --dpo models/dpo \\
    --num_samples 100 \\
    --output quick_eval_results.json

# Expected output:
# Cooperative Rate: ~95%
# Violations: Q=4%, Ql=0%, R=0%, M=16%
```

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    User Input Context                     │
└────────────────────┬─────────────────────────────────────┘
                     │
         ┌───────────▼──────────┐
         │   DPO Generator      │
         │   (GPT-2 + DPO)      │
         └───────────┬──────────┘
                     │
          Generated Response
                     │
         ┌───────────▼──────────┐
         │  Violation Detector  │
         │   (DeBERTa-v3)       │
         └───────────┬──────────┘
                     │
      ┌──────────────▼──────────────┐
      │   Violations Detected?      │
      └──┬──────────────────────┬───┘
         │ No                   │ Yes
         │                      │
         │              ┌───────▼────────┐
         │              │ Relation Issue?│
         │              └───┬────────┬───┘
         │                  │ Yes    │ No
         │          ┌───────▼────┐   │
         │          │ Retrieval  │   │
         │          │ + Regen    │   │
         │          └───────┬────┘   │
         │                  │    ┌───▼────┐
         │                  │    │ Repair │
         │                  │    │  (T5)  │
         │                  │    └───┬────┘
         │                  └────────┘
         │                      │
    ┌────▼──────────────────────▼────┐
    │      Final Response             │
    └─────────────────────────────────┘
```

---

## 📦 Installation

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS (12.0+), Windows 10/11
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with 8GB+ VRAM (T4, V100, A100, RTX 3090, etc.)
- **Storage**: 10GB for code + models + data

### Detailed Setup

```bash
# 1. Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip git

# 2. Clone and setup
git clone https://github.com/PushkarPrabhath27/Research-Model.git
cd Research-Model/GriceBench

# 3. Create virtual environment
python3.10 -m venv grice_env
source grice_env/bin/activate

# 4. Upgrade pip
pip install --upgrade pip setuptools wheel

# 5. Install PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 6. Install remaining dependencies
pip install -r requirements.txt

# 7. Download NLP resources
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

### Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

---

## 🔬 Experiment Reproduction

Reproduce all results from the paper in 5 parts (~8 hours total on Kaggle free tier).

### Part 1: Relation Repair System (1 hour)

**Goal:** Build retrieval-augmented repair for relation violations.

```bash
# 1. Create topical corpus
python scripts/create_response_corpus.py \\
    --data_dir data_processed \\
    --output data_processed/relation_repair/response_corpus.json

# 2. Build FAISS index
python scripts/build_retrieval_system.py \\
    --corpus data_processed/relation_repair/response_corpus.json \\
    --output data_processed/relation_repair/

# 3. Evaluate relevance
python scripts/evaluate_relation_repair.py \\
    --retrieval_dir data_processed/relation_repair \\
    --output results/part1_relation_repair.json
```

**Expected Output:**
- Corpus: ~50K responses
- Retrieval MRR: >0.7
- Top-1 accuracy: >60%

---

### Part 2: Human Evaluation Setup (30 min)

**Goal:** Prepare blinded evaluation interface for human annotators.

```bash
# 1. Prepare evaluation samples
python scripts/prepare_human_eval_samples.py \\
    --detector_results results/detector_predictions.json \\
    --num_samples 100 \\
    --output data_processed/human_eval/samples_blinded.json

# 2. Launch Gradio interface (local)
python scripts/human_eval_gradio.py \\
    --samples data_processed/human_eval/samples_blinded.json \\
    --port 7860

# Access at http://localhost:7860

# 3. Analyze inter-annotator agreement (after annotation)
python scripts/analyze_human_eval.py \\
    --annotations data_processed/human_eval/annotations.json \\
    --output results/part2_human_eval_report.md
```

**Metrics:** Krippendorff's α, Cohen's κ, Agreement tables

---

### Part 3: Baseline Comparisons (2 hours on Kaggle)

**Goal:** Compare GriceBench against Mistral-7B and Qwen2.5-7B.

**Option 1: Local (requires 24GB+ VRAM)**

```bash
python scripts/evaluate_baselines.py \\
    --models mistralai/Mistral-7B-Instruct-v0.2 Qwen/Qwen2.5-7B-Instruct \\
    --test_data data_processed/detector_data/detector_val.json \\
    --output results/part3_baselines.json
```

**Option 2: Kaggle (Free Tier)**

Upload `kaggle_notebooks/GRICEBENCH_PART_3_BASELINES.ipynb` to Kaggle with:
- Dataset: `gricebench-test-data`
- GPU: T4 x1
- Run time: ~2 hours

**Expected Results:**
- Mistral-7B: 89.1% cooperative
- Qwen2.5-7B: 84.2% cooperative
- GriceBench: 95.0% cooperative

---

### Part 4: Ablation Studies (3 hours on Kaggle)

**Goal:** Measure contribution of each system component.

**Kaggle Notebook:** `GRICEBENCH_PART_4_ABLATIONS.ipynb`

**Datasets Required:**
1. `gricean-maxim-detector-model` (detector checkpoint)
2. `gricebench-repair-model` (T5 repair model)
3. `dpo-generator-model` (DPO generator)
4. `gricebench-test-data` (evaluation data)

**Ablation Configurations:**
1. **full_system**: DPO + Detector + Repair
2. **dpo_only**: DPO without detection/repair
3. **detect_repair**: Baseline + Detector + Repair (no DPO)
4. **baseline**: GPT-2 without any optimization

**Run on Kaggle:**
- GPU: T4
- Expected runtime: ~3 hours
- Output: `ablation_report.md`, `ablation_results.json`

**Expected Results:**

| Config | Cooperative Rate |
|--------|------------------|
| full_system | 95.0% |
| detect_repair | 93.0% |
| baseline | 83.8% |
| dpo_only | 83.2% |

---

### Part 5: Error Analysis (30 min on Kaggle)

**Goal:** Identify failure patterns and generate confusion matrices.

**Kaggle Notebook:** `GRICEBENCH_PART_5_ERROR_ANALYSIS.ipynb`

**Datasets Required:**
1. `gricean-maxim-detector-model`
2. `gricebench-detector-validation` (detector_val.json)

**Outputs:**
- Confusion matrices (4 heatmaps)
- Hardest examples (FP/FN with high confidence)
- Error categorization by violation type
- Markdown report for paper

**Key Findings:**
- Quantity/Relation: Perfect F1 (1.0)
- Quality: F1=0.930 (6 FP, 4 FN)
- Manner: F1=0.940 (8 FP, 11 FN)
- Total error rate: 5.8%

---

## 📂 Project Structure

```
GriceBench/
├── data_raw/                    # Raw datasets (Wizard, TopicalChat, LIGHT)
│   ├── wizard_of_wikipedia/
│   ├── topicalchat/
│   └── light/
├── data_processed/              # Processed datasets
│   ├── detector_data/           # Multi-label classification data
│   │   ├── detector_train.json  # 4,012 examples
│   │   └── detector_val.json    # 500 examples
│   ├── repair_data/             # Seq2seq repair pairs
│   │   ├── repair_train.json    # 3,210 examples
│   │   └── repair_val.json      # 401 examples
│   ├── dpo_data/                # Preference pairs
│   │   ├── dpo_train.json       # 8,120 pairs
│   │   └── dpo_val.json         # 1,015 pairs
│   └── relation_repair/         # Retrieval corpus + FAISS
├── models/                      # Trained model checkpoints
│   ├── detector/
│   │   └── best_model.pt        # DeBERTa detector
│   ├── repair/                  # T5 repair model
│   └── dpo/                     # DPO-optimized GPT-2
├── scripts/                     # Python scripts (34 total)
│   ├── train_detector.py
│   ├── train_repair.py
│   ├── prepare_dpo_data.py
│   ├── evaluate_detector.py
│   ├── human_eval_gradio.py
│   └── ...
├── kaggle_notebooks/            # Jupyter notebooks for Kaggle
│   ├── GRICEBENCH_PART_3_BASELINES.ipynb
│   ├── GRICEBENCH_PART_4_ABLATIONS.ipynb
│   └── GRICEBENCH_PART_5_ERROR_ANALYSIS.ipynb
├── results/                     # Experiment outputs
│   ├── part3_baselines/
│   ├── part4output/             # Ablation study results
│   └── part5output/             # Error analysis
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🤖 Model Zoo

All models available on Hugging Face: `https://huggingface.co/PushkarPrabhath27/GriceBench`

| Model | Base Architecture | Parameters | Size | Use Case |
|-------|-------------------|------------|------|----------|
| **Detector** | DeBERTa-v3-base | 184M | 735MB | Multi-label maxim classification |
| **Repair** | T5-base | 220M | 900MB | Violation repair (non-Relation) |
| **Generator (DPO)** | GPT-2-medium | 355M | 1.4GB | Cooperative response generation |
| **Baseline Generator** | GPT-2-medium | 355M | 1.4GB | Unoptimized baseline |

### Download Individual Models

```bash
# Detector
wget https://huggingface.co/PushkarPrabhath27/GriceBench-Detector/resolve/main/best_model.pt \\
    -O models/detector/best_model.pt

# Repair
git lfs install
git clone https://huggingface.co/PushkarPrabhath27/GriceBench-Repair models/repair

# DPO Generator
git clone https://huggingface.co/PushkarPrabhath27/GriceBench-DPO models/dpo
```

---

## 📊 Performance Benchmarks

### Inference Speed (T4 GPU)

| Component | Batch Size | Throughput | Latency (avg) |
|-----------|------------|------------|---------------|
| Detector | 64 | 450 samples/sec | 2.2ms |
| Repair | 32 | 180 samples/sec | 5.5ms |
| Generator | 16 | 25 samples/sec | 40ms |
| **Full Pipeline** | 16 | 12 samples/sec | 83ms |

### Resource Usage

| Configuration | GPU Memory | CPU RAM |
|---------------|------------|---------|
| Detector only | 2.0GB | 4GB |
| Repair only | 3.5GB | 6GB |
| Generator only | 4.2GB | 8GB |
| **Full System** | 9.8GB | 16GB |

---

## 🔧 Training Your Own Models

### 1. Train Detector (2 hours on V100)

```bash
python scripts/train_detector.py \\
    --train_data data_processed/detector_data/detector_train.json \\
    --val_data data_processed/detector_data/detector_val.json \\
    --model_name microsoft/deberta-v3-base \\
    --output_dir models/detector \\
    --epochs 10 \\
    --batch_size 16 \\
    --learning_rate 2e-5
```

**Expected F1:** ~0.97 (macro-average)

### 2. Train Repair Model (3 hours on V100)

```bash
python scripts/train_repair.py \\
    --train_data data_processed/repair_data/repair_train.json \\
    --val_data data_processed/repair_data/repair_val.json \\
    --model_name t5-base \\
    --output_dir models/repair \\
    --epochs 5 \\
    --batch_size 8 \\
    --learning_rate 3e-5
```

**Expected BLEU:** Quantity=45.2, Quality=38.7, Manner=52.1

### 3. Train DPO Generator (Kaggle recommended)

Use `kaggle_notebooks/KAGGLE_DPO_TRAINING.ipynb` (free tier: ~6 hours)

```python
# Key hyperparameters
config = {
    "base_model": "gpt2-medium",
    "beta": 0.1,  # DPO temperature
    "learning_rate": 5e-7,
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation": 8
}
```

---

## 🧪 Testing

Run unit tests:

```bash
# Test all components
pytest tests/ -v

# Test specific modules
pytest tests/test_detector.py
pytest tests/test_repair.py
pytest tests/test_dpo.py
```

---

## 📝 Citation

If you use GriceBench in your research, please cite:

```bibtex
@inproceedings{gricebench2024,
  title={GriceBench: Operationalizing Gricean Maxims for Cooperative Dialogue Systems},
  author={Your Name},
  booktitle={Proceedings of the Conference},
  year={2024}
}
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/PushkarPrabhath27/Research-Model/issues)
- **Email**: your.email@university.edu
- **Paper**: [ArXiv Link](https://arxiv.org/abs/...)

---

## 🙏 Acknowledgments

- **Datasets**: Wizard of Wikipedia, TopicalChat, LIGHT
- **Infrastructure**: Kaggle Free Tier (T4 GPUs)
- **Models**: Hugging Face Transformers

---

**Last Updated:** 2026-01-23
