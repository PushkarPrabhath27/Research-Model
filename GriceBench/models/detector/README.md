# Detector Model Storage

## 📁 Place Your Downloaded Kaggle Files Here

Put the 2 files you downloaded from Kaggle in this folder:

1. **best_model.pt** (~500MB)
   - Your trained DeBERTa-v3-base violation detector
   - Contains model weights and configuration

2. **history.json** (rename from `history.json` if needed)
   - Training metrics (loss, F1 scores per epoch)
   - Used for analysis and plotting

## ✅ After Placing Files

Your directory should look like:
```
models/detector/
├── best_model.pt
├── training_history.json (renamed from history.json)
└── README.md (this file)
```

Once files are here, you're ready for Chapter 9: Detector Evaluation!
