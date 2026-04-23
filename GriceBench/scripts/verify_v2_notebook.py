import json

nb = json.load(open(r'c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\KAGGLE_PHASE5_DPO_V2.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']
print(f'Valid JSON: OK')
print(f'Cells: {len(cells)}')

kaggle_meta = nb['metadata']['kaggle']
print(f'GPU enabled: {kaggle_meta["isGpuEnabled"]}')
print(f'Internet enabled: {kaggle_meta["isInternetEnabled"]}')

for i, c in enumerate(cells):
    print(f'  Cell {i}: {c["cell_type"]} ({len(c["source"])} chars)')

src = ''.join(c['source'] for c in cells)

print('\nKey checks:')
print(f'  total_mem (bug): {src.count("total_mem") - src.count("total_memory")} occurrences')
print(f'  total_memory (correct): {src.count("total_memory")} occurrences')
print(f'  beta = 0.3: {"beta: float = 0.3" in src}')
print(f'  num_epochs = 3: {"num_epochs: int = 3" in src}')
print(f'  batch = 8: {"per_device_batch: int = 8" in src}')
print(f'  max_length = 256: {"max_length: int = 256" in src}')
print(f'  EarlyStoppingCallback: {"EarlyStoppingCallback" in src}')
print(f'  DPODiagnosticCallback: {"DPODiagnosticCallback" in src}')
print(f'  load_best_model_at_end: {"load_best_model_at_end=True" in src}')
print(f'  dpo_results_v2: {"dpo_results_v2" in src}')
