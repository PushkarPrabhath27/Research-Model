import json, sys

nb = json.load(open(r'c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\KAGGLE_PHASE5_DPO_ANNOTATED.ipynb', 'r', encoding='utf-8'))
fmt = nb["nbformat"]
fmt_m = nb["nbformat_minor"]
print(f"Format: nbformat {fmt}.{fmt_m}")
print(f"Cells: {len(nb['cells'])}")
kaggle = nb["metadata"]["kaggle"]
print(f"GPU enabled: {kaggle['isGpuEnabled']}")
print(f"Accelerator: {kaggle['accelerator']}")
print(f"Internet: {kaggle['isInternetEnabled']}")
print()
for i, c in enumerate(nb['cells']):
    snippet = c['source'][:90].strip().replace('\n', ' ')
    print(f"  Cell {i}: {c['cell_type']:8s} | {snippet}")

# Validate JSON structure
for i, c in enumerate(nb['cells']):
    assert 'cell_type' in c, f"Cell {i} missing cell_type"
    assert 'source' in c, f"Cell {i} missing source"
    if c['cell_type'] == 'code':
        assert 'outputs' in c, f"Cell {i} missing outputs"

print("\n✅ Notebook structure valid")
