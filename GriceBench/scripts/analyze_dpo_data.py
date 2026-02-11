import json

data = json.load(open(r'c:\Users\pushk\OneDrive\Documents\Research Model\GriceBench\scripts\annotation\tier1_hard_pairs_FULLY_ANNOTATED.json', 'r', encoding='utf-8'))
non_eq = [d for d in data if d['preference'] != 'equal']
print('Total:', len(data))
print('Non-equal (DPO):', len(non_eq))

strong = [d for d in non_eq if 'much' in d['preference']]
weak = [d for d in non_eq if 'slight' in d['preference']]
print('Strong (much):', len(strong))
print('Weak (slight):', len(weak))

avg_ctx = sum(len(d['context']) for d in non_eq) / len(non_eq)
avg_a = sum(len(d['response_A']) for d in non_eq) / len(non_eq)
avg_b = sum(len(d['response_B']) for d in non_eq) / len(non_eq)
max_ctx = max(len(d['context']) for d in non_eq)
max_resp = max(max(len(d['response_A']), len(d['response_B'])) for d in non_eq)

print(f'Avg context length: {avg_ctx:.0f} chars')
print(f'Avg resp_A length: {avg_a:.0f} chars')  
print(f'Avg resp_B length: {avg_b:.0f} chars')
print(f'Max context: {max_ctx} chars')
print(f'Max response: {max_resp} chars')

# Estimate token counts (rough: 1 token ~ 4 chars)
print(f'\nEstimated tokens:')
print(f'  Avg prompt: ~{int(avg_ctx/4)} tokens')
print(f'  Avg response: ~{int((avg_a+avg_b)/(2*4))} tokens')
print(f'  Max total: ~{int((max_ctx+max_resp)/4)} tokens')

# Train/val split calculation
from collections import Counter
import random
random.seed(42)

# Simulate the same split as the notebook
strong_list = [d for d in non_eq if 'much' in d['preference']]
weak_list = [d for d in non_eq if 'slight' in d['preference']]
random.shuffle(strong_list)
random.shuffle(weak_list)

val_strong = max(1, int(len(strong_list) * 0.15))
val_weak = max(1, int(len(weak_list) * 0.15)) if weak_list else 0

train_n = len(non_eq) - val_strong - val_weak
val_n = val_strong + val_weak
print(f'\nTrain: {train_n}, Val: {val_n}')
print(f'Steps/epoch at bs=8, grad_accum=4: {train_n // 32}')
print(f'Steps/epoch at bs=4, grad_accum=8: {train_n // 32}')
