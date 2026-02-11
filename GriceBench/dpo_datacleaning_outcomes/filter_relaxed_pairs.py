import json
import csv
import os

# Load original scored data
with open('scored_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total original pairs: {len(data)}")

# Filter criteria (Branch A: Relaxed Manner)
# Quantity > 0 (Strict)
# Quality > 0 (Strict)
# Relation > 0 (Strict)
# Manner > -0.1 (Relaxed)

relaxed_pairs = []

for entry in data:
    margins = entry.get('margins', {})
    
    qty = margins.get('quantity', 0)
    qlt = margins.get('quality', 0)
    rel = margins.get('relation', 0)
    man = margins.get('manner', 0)
    
    # The Relaxed Rule
    if (qty > 0 and 
        qlt > 0 and 
        rel > 0 and 
        man > -0.1): # Relaxed from 0 to -0.1
        
        # Add metadata for DPO
        entry['prompt'] = entry['prompt']
        entry['chosen'] = entry['chosen']
        entry['rejected'] = entry['rejected']
        relaxed_pairs.append(entry)

print(f"Relaxed filter pairs: {len(relaxed_pairs)}")

# Save JSON
with open('relaxed_dpo_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(relaxed_pairs, f, indent=2)

# Save CSV for inspection
with open('relaxed_dpo_pairs.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['prompt', 'chosen', 'rejected', 'quantity_margin', 'quality_margin', 'relation_margin', 'manner_margin'])
    for p in relaxed_pairs:
        m = p['margins']
        writer.writerow([
            p['prompt'], 
            p['chosen'], 
            p['rejected'], 
            m['quantity'], 
            m['quality'], 
            m['relation'], 
            m['manner']
        ])

print("Saved to relaxed_dpo_pairs.json and relaxed_dpo_pairs.csv")
