import json
from collections import Counter

# Load Phase 4 natural violations
data = json.load(open(r'c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\results\phase4output\natural_violations.json', 'r', encoding='utf-8'))

print(f"Type: {type(data)}")
if isinstance(data, list):
    print(f"Length: {len(data)}")
    print(f"First entry keys: {list(data[0].keys())}")
    print(f"\nFirst 3 entries:")
    for i, entry in enumerate(data[:3]):
        print(f"\n--- Entry {i} ---")
        for k, v in entry.items():
            val_str = str(v)[:200]
            print(f"  {k}: {val_str}")
    
    # Count by maxim / violation type
    if 'maxim' in data[0]:
        maxim_counts = Counter(d['maxim'] for d in data)
        print(f"\nMaxim distribution: {dict(maxim_counts)}")
    if 'violation_type' in data[0]:
        vt_counts = Counter(d['violation_type'] for d in data)
        print(f"\nViolation type distribution: {dict(vt_counts)}")
    if 'source' in data[0]:
        src_counts = Counter(d['source'] for d in data)
        print(f"\nSource distribution: {dict(src_counts)}")
    if 'strategy' in data[0]:
        strat_counts = Counter(d['strategy'] for d in data)
        print(f"\nStrategy distribution: {dict(strat_counts)}")
    if 'label' in data[0]:
        label_counts = Counter(str(d['label']) for d in data)
        print(f"\nLabel distribution: {dict(label_counts)}")
    
    # Check for 'text' field lengths
    if 'text' in data[0]:
        lens = [len(d['text']) for d in data]
        print(f"\nText lengths: min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.0f}")
    elif 'violated_text' in data[0]:
        lens = [len(d['violated_text']) for d in data]
        print(f"\nViolated text lengths: min={min(lens)}, max={max(lens)}, avg={sum(lens)/len(lens):.0f}")

elif isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")
    for k, v in data.items():
        if isinstance(v, list):
            print(f"  {k}: list of {len(v)}")
        elif isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())[:5]}")
        else:
            print(f"  {k}: {str(v)[:100]}")
