# DPO Training Quick Start Guide - 411 Clean Pairs

## Files Ready

| File | Location | Purpose |
|------|----------|---------|
| `clean_dpo_pairs.json` | `dpo_datacleaning_outcomes/` | Training data (411 pairs) |
| `clean_dpo_pairs.csv` | `dpo_datacleaning_outcomes/` | Same data, CSV format |
| `KAGGLE_DPO_411_CLEAN_PAIRS.ipynb` | `GriceBench/` | Training notebook |

---

## Step 1: Create Kaggle Dataset

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **"+ New Dataset"**
3. Name it: `gricebench-clean-dpo`
4. Upload `clean_dpo_pairs.json`
5. Click **"Create"**

---

## Step 2: Create Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"+ New Notebook"**
3. Upload `KAGGLE_DPO_411_CLEAN_PAIRS.ipynb`
4. **Enable GPU:** Settings → Accelerator → GPU T4 x2
5. **Add your dataset:** + Add Data → Your Datasets → gricebench-clean-dpo

---

## Step 3: Run Training

1. **Verify data path** in Cell 2 matches your dataset path:
   ```python
   DATA_PATH = "/kaggle/input/gricebench-clean-dpo/clean_dpo_pairs.json"
   ```
2. Click **"Run All"**
3. Training takes ~30-45 minutes on T4 GPU

---

## Step 4: Download Model

After training completes:
1. Go to **Output** tab
2. Download `dpo_411_clean_model.zip`

---

## Expected Outcomes

| Metric | Expected |
|--------|----------|
| Training loss | Decreasing over epochs |
| Eval loss | Stable or decreasing |
| Training time | ~30-45 mins |

---

## Next Steps After Training

Based on results, choose your path:

- **✅ If manner improves** → Relax manner threshold, get more data
- **⚠️ If weak improvement** → Move to synthetic generation
- **❌ If no effect** → Skip directly to synthetic generation

---

## Troubleshooting

**"File not found" error:**
- Check dataset is added to notebook
- Verify path matches: `/kaggle/input/YOUR-DATASET-NAME/clean_dpo_pairs.json`

**Out of memory:**
- Reduce `per_device_train_batch_size` from 2 to 1
- Reduce `max_length` from 512 to 384

**Training too slow:**
- Reduce `num_train_epochs` from 5 to 3
