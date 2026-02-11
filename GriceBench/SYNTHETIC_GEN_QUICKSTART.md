# Synthetic Generation Quick Start (Kaggle)

## Files Ready

| File | Location | Purpose |
|------|----------|---------|
| `KAGGLE_SYNTHETIC_GEN.ipynb` | `GriceBench/` | Generation script |
| `scored_data.json` | `GriceBench/dpo_datacleaning_outcomes/` | Input data |

---

## Step 1: Create Dataset
1. Go to Kaggle Datasets.
2. Create New Dataset named `gricebench-scored`.
3. Upload `scored_data.json`.

## Step 2: Setup Notebook
1. create New Notebook.
2. File -> Import Notebook -> Upload `KAGGLE_SYNTHETIC_GEN.ipynb`.
3. **Add Data** -> `gricebench-scored`.
4. **Secrets** (Add-ons menu):
   - Add new secret: `GEMINI_API_KEY`
   - Value: Your Google Gemini API Key

## Step 3: Run
1. Run All Cells.
2. It will process ~4000 prompts.
3. Autosaves to `synthetic_candidates.json`.

## Step 4: Download
- Download `synthetic_candidates.json` from the Output tab.

---

## Next Steps
Once you have the JSON:
1. We will **score** these new responses (to ensure they are actually good).
2. We will **filter** for all-positive margins.
3. We will **train** the final DPO model.
