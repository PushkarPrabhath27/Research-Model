# Quick Upload Instructions

## What You Have

You have `dpo_train_filtered.json` with **1,970 pairs** (conflict-filtered version).

## What to Do (5 minutes)

### Step 1: Upload to Kaggle

1. Go to: https://www.kaggle.com/datasets
2. Click **"New Dataset"**
3. Upload: `c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\dpo_train_filtered.json`
4. **Title:** `gricebench-dpo-with-scores`
5. **Subtitle:** "DPO training data with detector scores (1,970 pairs)"
6. Click **"Create"**

### Step 2: Use Multi-Stage Guide

Open `KAGGLE_MULTISTAGE_DPO_GUIDE.md` and in **CELL 2**, change the path to:

```python
# Load your uploaded data
with open('/kaggle/input/gricebench-dpo-with-scores/dpo_train_filtered.json') as f:
    original_data = json.load(f)
```

### Step 3: Adjust Thresholds in CELL 3

Since your data has weaker signals, use lower thresholds:

```python
# Adjust for your data's signal strength
strong_quantity = abs(item['quantity_margin']) > 0.10  # Lower from 0.3
strong_relation = abs(item['relation_margin']) > 0.10  # Lower from 0.3
```

### Step 4: Run Training

Follow the rest of the guide as-is. Expected results:
- Stage 1: ~600-800 content pairs
- Stage 2: ~300-400 manner pairs
- Final cooperative: 55-65% (still good!)

---

## Alternative: Get 3,551-Pair Version (Better Results)

If you want the better version with stronger signals:

1. Go to your Kaggle DPO scoring notebook
2. Find the cell output that says "3,551 pairs"
3. The data is in a variable - add this cell:

```python
# List all large variables
for name in dir():
    if not name.startswith('_'):
        obj = globals()[name]
        if isinstance(obj, list) and len(obj) > 3000:
            print(f"{name}: {len(obj)} items")
            if len(obj) == 3551:
                # Save it!
                with open('/kaggle/working/dpo_train_3551.json', 'w') as f:
                    json.dump(obj, f, indent=2)
                print(f"✅ Saved {name} as dpo_train_3551.json")
```

4. Download and upload as `gricebench-dpo-scored-original`
5. Use original thresholds (0.3) in the guide

---

**Recommendation: Just use the 1,970-pair file - it's faster and still works!**
