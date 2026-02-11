import json

nb = json.load(open(r'c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\KAGGLE_PHASE6_DETECTOR_V2.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']
print(f'Valid JSON: OK')
print(f'Cells: {len(cells)}')

kaggle = nb['metadata']['kaggle']
print(f'GPU: {kaggle["isGpuEnabled"]}')
print(f'Internet: {kaggle["isInternetEnabled"]}')

for i, c in enumerate(cells):
    print(f'  Cell {i}: {c["cell_type"]} ({len(c["source"])} chars)')

src = ''.join(c['source'] for c in cells)

print('\nKey feature checks:')
print(f'  MANDATORY DATA VERIFICATION: {"MANDATORY DATA VERIFICATION" in src}')
print(f'  assert len(violations): {"assert len(violations)" in src}')
print(f'  DATA LEAKAGE: {"DATA LEAKAGE" in src}')
print(f'  phase4_violation source: {"phase4_violation" in src}')
print(f'  generation_method: {"generation_method" in src}')
print(f'  Error Analysis: {"Error Analysis" in src or "ERROR ANALYSIS" in src}')
print(f'  Per-Generation-Method: {"PER-GENERATION-METHOD" in src}')
print(f'  HELD-OUT TEST: {"HELD-OUT" in src}')
print(f'  natural_violations.json: {"natural_violations.json" in src}')
print(f'  total_memory (not total_mem): {"total_memory" in src}')
print(f'  total_mem bug: {src.count("total_mem") - src.count("total_memory")}')
print(f'  deberta-v3-small: {"deberta-v3-small" in src}')
