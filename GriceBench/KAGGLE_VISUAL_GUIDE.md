# Kaggle Dataset Upload - Visual Walkthrough

## Your Detector Folder

Here's what you actually have in your detector folder:

![Your Detector Files](C:/Users/pushk/.gemini/antigravity/brain/954b83cb-cc9f-4839-937b-c35a79bab4d9/uploaded_image_1767277448291.png)

**Files you see:**
- ✅ `best_model.pt` - **THIS IS THE ONE YOU NEED** (735 MB)
- `history.json` - Optional (training metrics)
- `README.md` - Optional (documentation)

---

## What to Upload

### For Detector Model Dataset:
**Upload ONLY:**
- `best_model.pt` (the 735 MB file)

**You can skip:**
- `history.json` (not needed for evaluation)
- `README.md` (not needed for evaluation)

---

## Step-by-Step with Your Actual Files

### 1. Open Kaggle Datasets
- Go to: https://www.kaggle.com/datasets
- Click "New Dataset"

### 2. Upload the Model File

**Using your File Explorer (like in your screenshot):**

1. You're already in the right folder: `GriceBench > models > detector`
2. You can see the 3 files
3. **Click on `best_model.pt`** (the PT File, 735 MB)
4. Drag it to the Kaggle page
5. **OR** if using file picker:
   - Select `best_model.pt`
   - Click "Open" button at bottom

### 3. Fill in Details

```
Title: Gricean Maxim Detector Model
Tags: nlp, detector, deberta
License: Apache 2.0
Visibility: Private
```

### 4. Click "Create"

Wait for upload (5-10 minutes for 735 MB file)

### 5. Get Your Dataset Path

After creation, you'll see your dataset page.

**The path is:**
```
/kaggle/input/YOUR-DATASET-NAME
```

**How to find YOUR-DATASET-NAME:**
- Look at the URL in your browser
- It will be: `https://www.kaggle.com/datasets/YOUR-USERNAME/gricean-maxim-detector-model`
- The last part (`gricean-maxim-detector-model`) is your dataset name
- So your path is: `/kaggle/input/gricean-maxim-detector-model`

**Write it down here:**
```
My detector path: /kaggle/input/___________________________
```

---

## For DPO Model

Same process, but upload ALL files from `dpo_final` folder:
- All 9 files (adapter_config.json, adapter_model.safetensors, etc.)

---

## For Test Data

Same process, but upload:
- `dpo_val.json` from `GriceBench/data_processed/dpo_data/`

**Important:** The path for test data includes the filename:
```
/kaggle/input/YOUR-TEST-DATASET-NAME/dpo_val.json
```

---

## Summary

You need to create 3 datasets:

| Dataset | Files to Upload | Example Path |
|---------|----------------|--------------|
| Detector | `best_model.pt` only | `/kaggle/input/gricean-maxim-detector-model` |
| DPO Model | All 9 files from `dpo_final` | `/kaggle/input/dpo-generator-model` |
| Test Data | `dpo_val.json` only | `/kaggle/input/dpo-test-data/dpo_val.json` |

---

## Next Steps

After uploading all 3 datasets:
1. Write down all 3 paths
2. Move to creating the Kaggle notebook
3. You'll paste these paths into the notebook code

**See `KAGGLE_UPLOAD_GUIDE.md` for more detailed instructions!**
