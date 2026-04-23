# Quick Evaluation - How to Run

## ✅ What's Ready

Your repair model has been moved to the correct location:
```
GriceBench/models/repair/
├── repair_model/        ← Your trained T5 model
└── history.json         ← Training metrics
```

I've created a quick evaluation script:
```
GriceBench/scripts/quick_eval_repair.py
```

## 🚀 How to Run It

### Option 1: Command Line (Windows)

1. **Open Command Prompt** (search "cmd" in Start menu)

2. **Navigate to your project:**
   ```
   cd "c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench"
   ```

3. **Find your Python executable** (try each until one works):
   ```
   where python
   ```
   OR
   ```
   where python3
   ```
   OR
   ```
   py --version
   ```

4. **Run the evaluation** (use whichever Python command worked above):
   ```
   python scripts\quick_eval_repair.py
   ```
   OR
   ```
   python3 scripts\quick_eval_repair.py
   ```
   OR
   ```
   py scripts\quick_eval_repair.py
   ```

### Option 2: Anaconda/Miniconda (If you have it installed)

```bash
conda activate your_environment_name
cd "c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench"
python scripts\quick_eval_repair.py
```

### Option 3: VS Code or PyCharm

1. Open the file `scripts/quick_eval_repair.py`
2. Click the "Run" button (▶️ icon)
3. Make sure you're using the correct Python environment

### Option 4: Jupyter Notebook

Create a new notebook and run:
```python
%cd "c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench"
%run scripts/quick_eval_repair.py
```

---

## 📊 What to Expect

The script will:
1. Load your trained repair model
2. Test on 10 random examples
3. Show you:
   - The violated input
   - The generated repair
   - The reference (target) repair
   - How similar they are

**Expected runtime:** 1-3 minutes (depending on CPU/GPU)

---

## 🔍 What to Look For

**Good signs:**
- ✅ Generated repairs make sense
- ✅ Violations appear fixed
- ✅ Meaning preserved from input
- ✅ Word overlap > 50% with reference

**Bad signs:**
- ❌ Repairs are gibberish
- ❌ Violations still present
- ❌ Meaning completely changed
- ❌ Word overlap < 20% with reference

---

## ⚠️ If You Get Errors

### Error: "No module named 'transformers'"
**Fix:** Install dependencies first:
```
pip install transformers torch datasets
```

### Error: "FileNotFoundError: models/repair/repair_model"
**Fix:** Check the model was moved correctly:
```
dir models\repair
```
You should see a `repair_model` folder inside.

### Error: "CUDA out of memory"
**Fix:** The script will automatically use CPU if no GPU available. It'll just be slower (2-5 min instead of 30 sec).

---

## 📝 After Running

Once you run the script and see the results:

1. **If results look good (repairs make sense):**
   - ✅ Ready for full evaluation
   - Tell me and I'll help with comprehensive evaluation

2. **If results look bad (repairs don't make sense):**
   - ❌ Model may need retraining or debugging
   - Share some example outputs with me
   - We'll diagnose the issue

---

**Let me know when you've run it and what the results look like!** 🚀
