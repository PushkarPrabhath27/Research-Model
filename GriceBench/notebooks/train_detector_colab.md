# GriceBench Detector Training - Google Colab Notebook

This notebook trains the Gricean violation detector on Google Colab with GPU.

## Instructions:
1. Upload `GriceBench/` folder to Google Drive
2. Open this notebook in Google Colab
3. Enable GPU: Runtime → Change runtime type → T4 GPU
4. Run all cells

---

## Step 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Step 2: Navigate to Project

```python
import os
os.chdir('/content/drive/MyDrive/GriceBench')
print(f"Working directory: {os.getcwd()}")
!ls
```

## Step 3: Install Dependencies

```python
!pip install transformers datasets torch scikit-learn tqdm --quiet
```

## Step 4: Check GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

## Step 5: Prepare Data (if not already done)

```python
!python scripts/prepare_detector_data.py
```

## Step 6: Train Detector (Phase 1 - Weak Supervision)

```python
!python scripts/train_detector.py \
    --train data_processed/detector_data/detector_train.json \
    --val data_processed/detector_data/detector_val.json \
    --output models/detector_weak \
    --model microsoft/deberta-v3-base \
    --epochs 3 \
    --batch-size 16 \
    --lr 2e-5
```

## Step 7: Fine-tune on Gold Data (Phase 2)

```python
# After annotating gold examples, run:
!python scripts/train_detector.py \
    --train data_processed/gold_train.json \
    --val data_processed/gold_val.json \
    --output models/detector_gold \
    --model models/detector_weak \
    --epochs 5 \
    --batch-size 8 \
    --lr 1e-5
```

## Step 8: Evaluate

```python
import json
from pathlib import Path

# Load training history
history_path = Path('models/detector_weak/training_history.json')
if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)
    
    print("Training Results:")
    print(f"  Best Val F1: {max(history['val_f1']):.4f}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
```

## Step 9: Download Model

After training, download the model folder to your local machine or keep on Drive.

```python
# Zip for download
!zip -r models/detector_final.zip models/detector_weak/
print("Model saved! Download from Files panel or keep on Drive.")
```
