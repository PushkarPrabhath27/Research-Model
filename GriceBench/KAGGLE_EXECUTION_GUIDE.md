# GriceBench Kaggle Execution Guide

## Overview

This guide explains how to run the GriceBench scientific improvement plan on Kaggle.

---

## Required Datasets

### 1. gricebench-scientific-fix (Private Dataset)

Create a Kaggle dataset called `gricebench-scientific-fix` containing the following files.

**Files to upload (with their local paths):**

| File | Local Path |
|------|------------|
| `repair_test.json` | `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\data_processed\repair_data\repair_test.json` |
| `gold_annotation_set.json` | `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\data_processed\gold_annotation_set.json` |
| `val_examples.json` | `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\data_processed\val_examples.json` |
| `topical_corpus.json` | `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\data_processed\topical_corpus.json` |
| `faiss_index.pkl` (optional) | `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\data_processed\faiss_index.pkl` |

**Folder structure in Kaggle dataset:**
```
gricebench-scientific-fix/
├── repair_data/
│   └── repair_test.json
├── gold_annotation_set.json
├── val_examples.json
├── topical_corpus.json
└── faiss_index.pkl (optional)
```

**How to create the Kaggle dataset:**
1. Go to [Kaggle](https://www.kaggle.com) → Datasets → New Dataset
2. Name it `gricebench-scientific-fix`
3. Click "Upload Files" and navigate to the paths above
4. For `repair_test.json`, create a folder called `repair_data` and put it inside
5. Set visibility to **Private**
6. Click "Create"

---

## Phase 1: Data Preparation

**Notebook:** `KAGGLE_PHASE1_PREPARATION.ipynb`

**Datasets to Add:**
- `gricebench-scientific-fix` (your private dataset)

**Run order:**
1. Import notebook to Kaggle
2. Add dataset (right panel → Add Data → Your Work → gricebench-scientific-fix)
3. Run all cells
4. Download outputs:
   - `relation_eval_set.json` (200 examples for MRR eval)
   - `annotation_sample_1000.json` (for human annotation)

**After completion:**
- Add downloaded files back to your `gricebench-scientific-fix` dataset

---

## Phase 2: MRR Evaluation

**Notebook:** `KAGGLE_PHASE2_MRR_EVALUATION.ipynb`

**Datasets to Add:**
- `gricebench-scientific-fix` (with relation_eval_set.json)
- GPU recommended for faster encoding

**Run order:**
1. Import notebook to Kaggle
2. Add dataset
3. Enable GPU (Settings → Accelerator → GPU T4 x2)
4. Run all cells
5. Download `relation_repair_mrr.json`

**Decision Point:**
- If MRR ≥ 0.5 → Proceed to Phase 3 (Annotation)
- If MRR < 0.5 → Run improvement notebook

---

## Phase 3: Human Annotation

**Notebook:** Run locally using `scripts/annotation_interface.py`

OR create Kaggle notebook with Gradio interface for annotation.

**Output:** `self_annotations.json`

---

## Quick Reference

| Phase | Notebook | GPU? | Time | Output |
|-------|----------|------|------|--------|
| 1 | KAGGLE_PHASE1_PREPARATION | No | ~5 min | relation_eval_set.json, annotation_sample_1000.json |
| 2 | KAGGLE_PHASE2_MRR_EVALUATION | Yes | ~15 min | relation_repair_mrr.json |
| 3 | Local annotation interface | No | ~10 hrs | self_annotations.json |
| 4 | KAGGLE_DETECTOR_AGREEMENT | No | ~5 min | detector_human_agreement.json |

---

## Troubleshooting

### Dataset not found
- Check mount path: `/kaggle/input/gricebench-data/`
- Ensure dataset is added to notebook

### Out of memory
- Reduce corpus sample size (MAX_CORPUS = 5000)
- Use smaller batch size (batch_size=32)

### Model download slow
- Pre-download sentence-transformers model as dataset
- Mount from `/kaggle/input/all-minilm-l6-v2/`
