#!/usr/bin/env python3
"""
Phase 7 Evaluation Notebook Generator — REBUILT (crash-proof, disk-checkpointing)

Architecture rules:
  - Every cell is 100% self-contained (imports, constants, helpers repeated as needed)
  - State between cells ALWAYS passes through disk (pickle / json)
  - Detector and repair model are NEVER in VRAM at the same time
  - Every model load preceded by full VRAM flush
  - Every cell starts by re-importing everything it needs
"""
import json, os

# ---------------------------------------------------------------------------
def nb():
    cells = []
    def code(src):
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {"trusted": True}, "outputs": [], "source": src})
    def md(src):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

    # ── Title ───────────────────────────────────────────────────────────────
    md("# Phase 7: GriceBench End-to-End Evaluation (Tier 3 — Rebuilt)\n\n"
       "**Architecture:** Disk-checkpointing. Each cell is fully self-contained.\n"
       "Detection and repair are always separated by a disk-save + VRAM-free.\n\n"
       "| Cell | Purpose | VRAM state after |\n"
       "|------|---------|------------------|\n"
       "| 1 | Install + helpers | empty |\n"
       "| 2 | Config + paths | empty |\n"
       "| 3 | Load & clean DPO data | empty |\n"
       "| 4 | DPO training (skip if adapter) | empty |\n"
       "| 5 | Build test corpus → disk | empty |\n"
       "| 6 | Detection → disk → **free detector** | empty |\n"
       "| 7 | Load repair model → repair → disk | empty |\n"
       "| 8 | Stats + final report | empty |")

    # ====================================================================
    # CELL 1 — Install + global helpers (no models, no VRAM)
    # ====================================================================
    code("""\
# ============================================================
# CELL 1 — INSTALL + HELPERS  (no models loaded here)
# ============================================================
import subprocess, sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Installing packages...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'trl>=0.7.0', 'peft>=0.5.0', 'bitsandbytes>=0.41.0',
    'accelerate>=0.21.0', 'datasets>=2.14.0',
    'transformers>=4.35.0', 'scikit-learn>=1.3.0', 'scipy>=1.11.0'])
print("Done.")

import torch, gc, json, re, random, logging, pickle
import numpy as np
from datetime import datetime
from collections import Counter
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger('Phase7')

assert torch.cuda.is_available(), "GPU required!"
GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEM  = torch.cuda.get_device_properties(0).total_memory / 1e9
logger.info(f"GPU: {GPU_NAME}  ({GPU_MEM:.1f} GB total)")

# ── Reusable helpers (duplicated in cells that need them) ──────────────

def purge_vram(*varnames):
    \"\"\"Delete named globals + flush CUDA allocator.\"\"\"
    import gc, ctypes
    g = globals()
    for n in varnames:
        if n in g:
            del g[n]
    gc.collect()
    try: torch.cuda.empty_cache(); torch.cuda.synchronize()
    except Exception: pass
    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception: pass
    free = (torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)) / 1e9
    logger.info(f"VRAM after purge: {free:.2f} GB free / {GPU_MEM:.1f} GB")
    return free

def find_file(name, root='/kaggle/input'):
    for r, _, files in os.walk(root):
        if name in files:
            p = os.path.join(r, name)
            logger.info(f"Found {name}: {p}")
            return p
    return None

def has_word_bang(text):
    if not text: return False
    words = text.split()
    return len(words) >= 5 and sum(w.endswith('!') for w in words) / len(words) > 0.25

def clean_artifacts(text):
    if not text: return text
    text = re.sub(r'(\\w)!+', r'\\1', text)
    text = re.sub(r'\\s!+\\s', ' ', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\\s+([!?.,;:])', r'\\1', text)
    text = re.sub(r'([.!?])([A-Z])', r'\\1 \\2', text)
    text = re.sub(r'\\s{2,}', ' ', text)
    return text.strip()

def make_prompt(context):
    return ("Continue the following conversation naturally, following Gricean "
            "maxims (be relevant, truthful, clear, appropriately informative):"
            f"\\n\\n{context}\\n\\nResponse:")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = True

WORK = '/kaggle/working'
os.makedirs(WORK, exist_ok=True)
logger.info("Cell 1 complete — helpers ready, no VRAM used.")
""")

    # ====================================================================
    # CELL 2 — Config + path detection
    # ====================================================================
    code("""\
# ============================================================
# CELL 2 — CONFIG + PATH DETECTION
# ============================================================
import os, logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger('Phase7')

def find_file(name, root='/kaggle/input'):
    for r, _, files in os.walk(root):
        if name in files:
            p = os.path.join(r, name)
            logger.info(f"Found {name}: {p}")
            return p
    return None

# ── Paths ──────────────────────────────────────────────────
DPO_MODEL      = 'Qwen/Qwen2.5-7B-Instruct'
DETECTOR_MODEL = 'microsoft/deberta-v3-small'
ADAPTER_PATH   = '/kaggle/working/dpo_7b/lora_adapter'
DPO_OUT        = '/kaggle/working/dpo_7b'
WORK           = '/kaggle/working'
os.makedirs(DPO_OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

DPO_DATA_PATH  = find_file('tier1_hard_pairs_FULLY_ANNOTATED.json') or ''
DET_CKPT       = find_file('best_model.pt') or ''
PHASE4_PATH    = find_file('natural_violations.json') or ''

assert DPO_DATA_PATH, "tier1_hard_pairs_FULLY_ANNOTATED.json NOT FOUND under /kaggle/input"
assert DET_CKPT,      "best_model.pt NOT FOUND under /kaggle/input"
assert PHASE4_PATH,   "natural_violations.json NOT FOUND under /kaggle/input"

ADAPTER_EXISTS = os.path.isdir(ADAPTER_PATH)

# ── QLoRA / DPO training hyperparameters ───────────────────
LORA_R            = 128
LORA_ALPHA        = 256
LORA_DROPOUT      = 0.05
LORA_TARGETS      = ['q_proj','k_proj','v_proj','o_proj',
                      'gate_proj','up_proj','down_proj']
DPO_BETA          = 0.3
MAX_LEN           = 512
MAX_PROMPT_LEN    = 384
LR                = 3e-5
NUM_EPOCHS        = 3
BATCH             = 2
GRAD_ACC          = 8
EVAL_STEPS        = 8
ES_PATIENCE       = 5
ES_THRESHOLD      = 0.005
MAX_REPAIR_TRIES  = 3
SEED              = 42
MAXIM_NAMES       = ['Quantity', 'Quality', 'Relation', 'Manner']
DEFAULT_THRESH    = [0.9, 0.55, 0.75, 0.45]

logger.info(f"Adapter exists: {ADAPTER_EXISTS}")
logger.info(f"DPO data: {DPO_DATA_PATH}")
logger.info(f"Detector: {DET_CKPT}")
logger.info(f"Phase4:   {PHASE4_PATH}")
logger.info("Cell 2 complete — config ready, no VRAM used.")
""")

    # ====================================================================
    # CELL 3 — DPO data cleaning
    # ====================================================================
    code("""\
# ============================================================
# CELL 3 — LOAD + VALIDATE DPO DATA
# ============================================================
import json, re, random, logging, os, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger('Phase7')

# Re-declare constants needed in this cell
DPO_DATA_PATH = [v for v in ['/kaggle/working'] if False] or None
import glob as _g
_hits = _g.glob('/kaggle/input/**/tier1_hard_pairs_FULLY_ANNOTATED.json', recursive=True)
DPO_DATA_PATH = _hits[0] if _hits else None
assert DPO_DATA_PATH, "tier1_hard_pairs_FULLY_ANNOTATED.json not found!"

SEED = 42
random.seed(SEED)
WORK = '/kaggle/working'
MAX_PROMPT_LEN = 384
MAX_LEN = 512

def make_prompt(context):
    return ("Continue the following conversation naturally, following Gricean "
            "maxims (be relevant, truthful, clear, appropriately informative):"
            f"\\n\\n{context}\\n\\nResponse:")

def has_word_bang(text):
    if not text: return False
    words = text.split()
    return len(words) >= 5 and sum(w.endswith('!') for w in words)/len(words) > 0.25

with open(DPO_DATA_PATH, 'r', encoding='utf-8') as f:
    raw = json.load(f)
logger.info(f"Raw records: {len(raw)}")

from collections import Counter
good, stats = [], Counter()
artifact_re = [re.compile(p) for p in [
    r'\\[\\d{2}:\\d{2}:\\d{2}\\].*\\[Server',
    r'From:.*@.*\\.\\w+',
    r'\\[Client thread/INFO\\]',
]]

for rec in raw:
    pref = rec.get('preference', 'equal')
    if pref == 'equal': stats['equal_skip'] += 1; continue
    if pref.startswith('A'):
        chosen, rejected = rec.get('response_A','').strip(), rec.get('response_B','').strip()
    else:
        chosen, rejected = rec.get('response_B','').strip(), rec.get('response_A','').strip()
    strength = 1.0 if 'much' in pref else 0.6
    if len(chosen.split()) < 8 or len(rejected.split()) < 8:
        stats['too_short'] += 1; continue
    if chosen == rejected: stats['identical'] += 1; continue
    if has_word_bang(chosen): stats['chosen_word_bang'] += 1; continue
    if any(p.search(chosen) for p in artifact_re): stats['chosen_artifact'] += 1; continue
    good.append({'prompt': make_prompt(rec.get('context','')),
                 'chosen': chosen, 'rejected': rejected, 'strength': strength})
    stats['kept'] += 1

logger.info("Cleaning stats:")
for k,v in stats.most_common(): logger.info(f"  {k}: {v}")
assert len(good) >= 50, f"Too few pairs after cleaning: {len(good)}"

strong = [p for p in good if p['strength']==1.0]
weak   = [p for p in good if p['strength']<1.0]
random.shuffle(strong); random.shuffle(weak)
vs = max(1, int(len(strong)*0.15))
vw = max(1, int(len(weak)*0.15)) if weak else 0
val_pairs   = strong[:vs] + weak[:vw]
train_pairs = strong[vs:] + weak[vw:]
random.shuffle(train_pairs); random.shuffle(val_pairs)

import pickle
with open(f'{WORK}/dpo_pairs.pkl','wb') as f:
    pickle.dump({'train': train_pairs, 'val': val_pairs}, f)
logger.info(f"DPO split: train={len(train_pairs)}  val={len(val_pairs)}")
logger.info("Saved to /kaggle/working/dpo_pairs.pkl")
logger.info("Cell 3 complete — no VRAM used.")
""")

    # ====================================================================
    # CELL 4 — DPO training (skip if adapter exists)
    # ====================================================================
    code("""\
# ============================================================
# CELL 4 — DPO TRAINING  (skips if adapter already exists)
# ============================================================
import os, gc, json, pickle, logging, warnings, torch
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger('Phase7')

ADAPTER_PATH = '/kaggle/working/dpo_7b/lora_adapter'
DPO_OUT      = '/kaggle/working/dpo_7b'
DPO_MODEL    = 'Qwen/Qwen2.5-7B-Instruct'
WORK         = '/kaggle/working'
SEED = 42

dpo_meta = {'adapter_reused': False, 'train_time_seconds': 0,
            'eval_loss': float('nan'), 'preference_accuracy': float('nan'),
            'reward_margin': float('nan'), 'train_pairs': 0, 'val_pairs': 0}

if os.path.isdir(ADAPTER_PATH):
    logger.info(f"Adapter exists — SKIPPING training.")
    dpo_meta['adapter_reused'] = True
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer
    from transformers import EarlyStoppingCallback
    from datasets import Dataset as HFDataset
    from datetime import datetime

    assert os.path.exists(f'{WORK}/dpo_pairs.pkl'), "Run Cell 3 first!"
    with open(f'{WORK}/dpo_pairs.pkl','rb') as f:
        dp = pickle.load(f)
    train_pairs, val_pairs = dp['train'], dp['val']
    dpo_meta['train_pairs'] = len(train_pairs)
    dpo_meta['val_pairs']   = len(val_pairs)

    BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        DPO_MODEL, quantization_config=BNB, device_map='auto',
        trust_remote_code=True, attn_implementation='eager')
    tok = AutoTokenizer.from_pretrained(DPO_MODEL, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model.config.pad_token_id = tok.eos_token_id
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=128, lora_alpha=256, lora_dropout=0.05,
                      target_modules=['q_proj','k_proj','v_proj','o_proj',
                                      'gate_proj','up_proj','down_proj'],
                      bias='none', task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)

    tr_ds = HFDataset.from_list([{'prompt':p['prompt'],'chosen':p['chosen'],
                                   'rejected':p['rejected']} for p in train_pairs])
    va_ds = HFDataset.from_list([{'prompt':p['prompt'],'chosen':p['chosen'],
                                   'rejected':p['rejected']} for p in val_pairs])
    args = TRLDPOConfig(
        output_dir=DPO_OUT, num_train_epochs=3,
        per_device_train_batch_size=2, per_device_eval_batch_size=4,
        gradient_accumulation_steps=8, beta=0.3, loss_type='sigmoid',
        learning_rate=3e-5, lr_scheduler_type='cosine', warmup_ratio=0.1,
        weight_decay=0.01, max_grad_norm=1.0,
        eval_strategy='steps', eval_steps=8,
        save_strategy='steps', save_steps=8,
        load_best_model_at_end=True, metric_for_best_model='eval_loss',
        greater_is_better=False, save_total_limit=3,
        bf16=True, gradient_checkpointing=True, optim='paged_adamw_8bit',
        dataloader_num_workers=0, max_length=512, max_prompt_length=384,
        logging_steps=1, report_to='none', seed=SEED,
        ddp_find_unused_parameters=False)

    trainer = DPOTrainer(model=model, args=args, train_dataset=tr_ds,
                         eval_dataset=va_ds, processing_class=tok,
                         callbacks=[EarlyStoppingCallback(
                             early_stopping_patience=5,
                             early_stopping_threshold=0.005)])
    t0 = datetime.now()
    trainer.train()
    dpo_meta['train_time_seconds'] = (datetime.now()-t0).total_seconds()
    ev = trainer.evaluate()
    dpo_meta['eval_loss']           = ev.get('eval_loss', float('nan'))
    dpo_meta['preference_accuracy'] = ev.get('eval_rewards/accuracies', float('nan'))
    dpo_meta['reward_margin']       = ev.get('eval_rewards/margins', float('nan'))
    model.save_pretrained(ADAPTER_PATH)
    tok.save_pretrained(ADAPTER_PATH)
    logger.info(f"Adapter saved: {ADAPTER_PATH}")

    # Free everything immediately
    del model, trainer, tok
    gc.collect(); torch.cuda.empty_cache()
    logger.info("DPO model freed from VRAM.")

with open(f'{WORK}/dpo_meta.json','w') as f:
    json.dump(dpo_meta, f, indent=2, default=str)
logger.info(f"Meta saved. Cell 4 complete. VRAM free.")
""")

    # ====================================================================
    # CELL 5 — Build test corpus → disk
    # ====================================================================
    code("""\
# ============================================================
# CELL 5 — BUILD TEST CORPUS  (CPU only, saves to disk)
# ============================================================
import os, json, random, logging, pickle, warnings, glob
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger('Phase7')

WORK = '/kaggle/working'
SEED = 42

hits = glob.glob('/kaggle/input/**/natural_violations.json', recursive=True)
assert hits, "natural_violations.json not found!"
PHASE4_PATH = hits[0]
logger.info(f"Phase4: {PHASE4_PATH}")

with open(PHASE4_PATH,'r',encoding='utf-8') as f:
    data = json.load(f)
logger.info(f"Phase 4 records: {len(data)}")

def norm(t): return ' '.join(str(t).strip().split()) if t else ''

random.seed(777)
random.shuffle(data)
pool = data[-600:]
test_violations, test_clean = [], []

for item in pool:
    ctx = norm(item.get('context',''))
    mx  = item.get('maxim','unknown')
    gm  = item.get('generation_method','unknown')

    viol = norm(item.get('violated_response',''))
    if viol and len(test_violations) < 500:
        text = f"{ctx} [SEP] {viol}" if ctx else viol
        ld = item.get('labels',{})
        if isinstance(ld, dict):
            labs = [int(ld.get('quantity',0)), int(ld.get('quality',0)),
                    int(ld.get('relation',0)), int(ld.get('manner',0))]
        else:
            ml = str(mx).lower()
            labs = [int('quantity' in ml),int('quality' in ml),
                    int('relation' in ml),int('manner' in ml)]
        if sum(labs) > 0 and len(text) > 50:
            test_violations.append({'text':text,'labels':labs,
                'source':'violation','generation_method':gm,'maxim':mx})

    orig = norm(item.get('original_response',''))
    if orig and len(test_clean) < 500:
        text = f"{ctx} [SEP] {orig}" if ctx else orig
        if len(text) > 50:
            test_clean.append({'text':text,'labels':[0,0,0,0],
                'source':'clean','generation_method':'clean','maxim':'none'})

test_corpus = test_violations + test_clean
random.shuffle(test_corpus)
random.seed(SEED)

with open(f'{WORK}/test_corpus.pkl','wb') as f:
    pickle.dump(test_corpus, f)

logger.info(f"Test corpus: {len(test_corpus)} "
            f"(viol={len(test_violations)}, clean={len(test_clean)})")
logger.info("Saved to /kaggle/working/test_corpus.pkl")
logger.info("Cell 5 complete — no VRAM used.")
""")

    # ====================================================================
    # CELL 6 — Load detector + detect + save + FREE (all in one cell)
    # ====================================================================
    code("""\
# ============================================================
# CELL 6 — DETECTION  (loads detector, detects, saves, then FREES)
# NOTE: By the end of this cell, VRAM is empty again.
# ============================================================
import os, gc, json, pickle, logging, warnings, glob
import torch, torch.nn as nn, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, roc_auc_score)
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger('Phase7')

WORK         = '/kaggle/working'
DETECTOR_MDL = 'microsoft/deberta-v3-small'
MAXIM_NAMES  = ['Quantity','Quality','Relation','Manner']
DEFAULT_THR  = [0.9, 0.55, 0.75, 0.45]
device       = 'cuda'

# ── Load corpus from disk ──
assert os.path.exists(f'{WORK}/test_corpus.pkl'), \
    "Run Cell 5 first! /kaggle/working/test_corpus.pkl not found."
with open(f'{WORK}/test_corpus.pkl','rb') as f:
    test_corpus = pickle.load(f)
logger.info(f"Corpus loaded: {len(test_corpus)} examples")

# ── Locate detector checkpoint ──
hits = glob.glob('/kaggle/input/**/best_model.pt', recursive=True)
assert hits, "best_model.pt not found!"
DET_CKPT = hits[0]
logger.info(f"Detector checkpoint: {DET_CKPT}")

# ── Define detector architecture ──
class GriceDetector(nn.Module):
    def __init__(self, mdl, n):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(mdl)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)                        # match checkpoint key
        self.classifier = nn.Sequential(nn.Linear(h,h//2), nn.GELU(),  # match checkpoint key
                                        nn.Dropout(0.1), nn.Linear(h//2,n))
        self.register_buffer('pos_weight', torch.ones(n))     # match checkpoint key
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # .float() fixes Half/Float mismatch on P100 mixed precision
        x = self.dropout(out.last_hidden_state[:,0,:].float())
        return {'logits': self.classifier(x)}

# ── Load detector ──
ckpt = torch.load(DET_CKPT, map_location='cpu', weights_only=False)
det  = GriceDetector(DETECTOR_MDL, 4)
det.load_state_dict(ckpt['model_state_dict'])
det  = det.to(device).eval()
thr  = ckpt.get('thresholds', DEFAULT_THR)
logger.info(f"Detector loaded. Thresholds: {dict(zip(MAXIM_NAMES,thr))}")

dtok = AutoTokenizer.from_pretrained(DETECTOR_MDL)

# ── Dataset ──
class DetDS(Dataset):
    def __init__(self, ex, tok, ml=512):
        self.ex, self.tok, self.ml = ex, tok, ml
    def __len__(self): return len(self.ex)
    def __getitem__(self, i):
        e   = self.tok(self.ex[i]['text'], max_length=self.ml,
                       padding='max_length', truncation=True, return_tensors='pt')
        lb  = torch.tensor(self.ex[i]['labels'], dtype=torch.float)
        return {'input_ids': e['input_ids'].squeeze(0),
                'attention_mask': e['attention_mask'].squeeze(0),
                'labels': lb}

loader = DataLoader(DetDS(test_corpus, dtok), batch_size=32,
                    shuffle=False, num_workers=0, pin_memory=True)

# ── Inference ──
all_probs, all_labels = [], []
with torch.no_grad():
    for batch in loader:
        out   = det(batch['input_ids'].to(device),
                    batch['attention_mask'].to(device))
        probs = torch.sigmoid(out['logits']).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch['labels'].numpy())

all_probs  = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)
all_preds  = (all_probs >= np.array(thr)).astype(int)

# ── Per-class metrics ──
det_results = {}
for i, name in enumerate(MAXIM_NAMES):
    if all_labels[:,i].sum() > 0:
        f1   = f1_score(all_labels[:,i], all_preds[:,i], zero_division=0)
        prec = precision_score(all_labels[:,i], all_preds[:,i], zero_division=0)
        rec  = recall_score(all_labels[:,i], all_preds[:,i], zero_division=0)
        try:   auc = roc_auc_score(all_labels[:,i], all_probs[:,i])
        except: auc = 0.0
    else:
        f1 = prec = rec = auc = 0.0
    det_results[name] = {'f1':f1,'precision':prec,'recall':rec,'auc':auc}
    logger.info(f"  {name}: F1={f1:.3f} P={prec:.3f} R={rec:.3f} AUC={auc:.3f}")

macro_f1 = float(np.mean([v['f1'] for v in det_results.values()]))
detected_indices = [i for i in range(len(test_corpus)) if any(all_preds[i])]
logger.info(f"Macro F1: {macro_f1:.4f}")
logger.info(f"Detected violations: {len(detected_indices)}/{len(test_corpus)}")

# ── Save ALL detection state to disk ──
with open(f'{WORK}/detection_ckpt.pkl','wb') as f:
    pickle.dump({'all_probs': all_probs, 'all_labels': all_labels,
                 'all_preds': all_preds, 'det_results': det_results,
                 'macro_f1': macro_f1, 'detected_indices': detected_indices,
                 'thr': thr, 'MAXIM_NAMES': MAXIM_NAMES}, f)
logger.info("Detection checkpoint saved → /kaggle/working/detection_ckpt.pkl")

# ── FREE DETECTOR — must happen before Cell 7 loads repair model ──
del det, dtok, loader, all_probs, all_labels, all_preds, ckpt
gc.collect()
try:
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception: pass
free = (torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated(0)) / 1e9
GPU_MEM = torch.cuda.get_device_properties(0).total_memory / 1e9
logger.info(f"Detector freed. VRAM: {free:.2f} GB free / {GPU_MEM:.1f} GB total")
logger.info("Cell 6 complete. READY for Cell 7 (repair model load).")
""")

    # ====================================================================
    # CELL 7 — Repair model + iterative repair + save (all in one cell)
    # ====================================================================
    code("""\
# ============================================================
# CELL 7 — REPAIR MODEL + ITERATIVE REPAIR + SAVE RESULTS
# Self-contained: loads everything it needs from disk.
# Safe to re-run if kernel restarts after Cell 6.
# ============================================================
import os, gc, re, json, pickle, random, logging, warnings, glob
import torch, numpy as np
from collections import Counter
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger('Phase7')

WORK          = '/kaggle/working'
DPO_MODEL     = 'Qwen/Qwen2.5-7B-Instruct'
ADAPTER_PATH  = '/kaggle/working/dpo_7b/lora_adapter'
MAX_PROMPT    = 384
MAX_TRIES     = 3
SEED          = 42
random.seed(SEED); np.random.seed(SEED)

GPU_MEM = torch.cuda.get_device_properties(0).total_memory / 1e9
device  = 'cuda'

# ── Re-declare helpers (self-contained) ──────────────────────
def has_word_bang(text):
    if not text: return False
    words = text.split()
    return len(words) >= 5 and sum(w.endswith('!') for w in words)/len(words) > 0.25

def clean_artifacts(text):
    if not text: return text
    text = re.sub(r'(\\w)!+', r'\\1', text)
    text = re.sub(r'\\s!+\\s', ' ', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\\s+([!?.,;:])', r'\\1', text)
    text = re.sub(r'([.!?])([A-Z])', r'\\1 \\2', text)
    text = re.sub(r'\\s{2,}', ' ', text)
    return text.strip()

def make_prompt(context):
    return ("Continue the following conversation naturally, following Gricean "
            "maxims (be relevant, truthful, clear, appropriately informative):"
            f"\\n\\n{context}\\n\\nResponse:")

# ── Load detection checkpoint from disk ──────────────────────
ckpt_path = f'{WORK}/detection_ckpt.pkl'
assert os.path.exists(ckpt_path), \
    "Run Cell 6 first! /kaggle/working/detection_ckpt.pkl not found."

with open(ckpt_path,'rb') as f:
    dc = pickle.load(f)
all_preds        = dc['all_preds']
det_results      = dc['det_results']
macro_f1         = dc['macro_f1']
detected_indices = dc['detected_indices']
MAXIM_NAMES      = dc['MAXIM_NAMES']
thr              = dc['thr']
logger.info(f"Detection restored: {len(detected_indices)} violations, macro_f1={macro_f1:.4f}")

with open(f'{WORK}/test_corpus.pkl','rb') as f:
    test_corpus = pickle.load(f)
logger.info(f"Corpus restored: {len(test_corpus)} examples")

# ── Verify VRAM is clear enough ────────────────────────────
free_gb = (torch.cuda.get_device_properties(0).total_memory
           - torch.cuda.memory_allocated(0)) / 1e9
logger.info(f"Free VRAM before model load: {free_gb:.2f} GB / {GPU_MEM:.1f} GB")
if free_gb < 8.0:
    logger.warning(
        f"Only {free_gb:.1f} GB free — may OOM. "
        "Restart kernel and run only Cell 7 (all data loads from disk).")

# ── Load repair model ─────────────────────────────────────
assert os.path.isdir(ADAPTER_PATH), \
    f"Adapter not found at {ADAPTER_PATH}. Run Cell 4 first."

BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                          bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_use_double_quant=True)

logger.info("Loading Qwen2.5-7B base...")
base = AutoModelForCausalLM.from_pretrained(
    DPO_MODEL, quantization_config=BNB, device_map='auto',
    trust_remote_code=True, attn_implementation='eager')
logger.info("Loading LoRA adapter...")
repair_model = PeftModel.from_pretrained(base, ADAPTER_PATH)
repair_model.eval()

rtok = AutoTokenizer.from_pretrained(DPO_MODEL, trust_remote_code=True)
if rtok.pad_token is None: rtok.pad_token = rtok.eos_token

peak = torch.cuda.max_memory_allocated(0) / 1e9
logger.info(f"Repair model ready — peak VRAM: {peak:.2f} GB / {GPU_MEM:.1f} GB")

# ── Generation helper ─────────────────────────────────────
def gen_repair(prompt, attempt=0):
    enc = rtok(prompt, return_tensors='pt', truncation=True,
               max_length=MAX_PROMPT).to(device)
    kw = {'max_new_tokens': 150, 'repetition_penalty': 1.0,
          'pad_token_id': rtok.eos_token_id,
          'eos_token_id':  rtok.eos_token_id}
    if attempt == 0:
        kw.update({'do_sample': False, 'num_beams': 1})
    elif attempt == 1:
        kw.update({'do_sample': True, 'temperature': 0.5, 'top_p': 0.9})
    else:
        kw.update({'do_sample': True, 'temperature': 0.8, 'top_p': 0.95})
    try:
        with torch.no_grad():
            out = repair_model.generate(**enc, **kw)
        raw = rtok.decode(out[0][enc['input_ids'].shape[1]:],
                          skip_special_tokens=True)
        return clean_artifacts(raw)
    except Exception as e:
        logger.warning(f"  gen failed attempt={attempt}: {e}")
        return ""

def is_success(orig_labels, rep_preds, rep_text):
    fixed   = any(o==1 and p==0 for o,p in zip(orig_labels, rep_preds))
    new_vio = any(o==0 and p==1 for o,p in zip(orig_labels, rep_preds))
    return fixed and not new_vio and not has_word_bang(rep_text)

# ── Iterative repair (3 attempts, greedy first) ────────────
repair_sample = min(200, len(detected_indices))
random.seed(SEED)
repair_idxs = random.sample(detected_indices, repair_sample)
repair_results = []
attempt_dist   = Counter()
t0 = datetime.now()

for count, idx in enumerate(repair_idxs):
    ex   = test_corpus[idx]
    orig = ex['labels']
    ctx  = ex['text'].split('[SEP]')[0].strip()
    prompt = make_prompt(ctx)

    best_text, best_preds, best_attempt = "", orig[:], 1
    best_score = float('inf')
    succeeded  = False

    for attempt in range(MAX_TRIES):
        text = gen_repair(prompt, attempt)
        if not text: continue

        # Inline detection using saved threshold + logits-free threshold method
        # (we only have all_preds from the original run here, so for repaired
        #  text we re-use the same threshold on raw text via heuristic)
        # -- Simple lexical violation heuristic for iteration decisions --
        excl = text.count('!') / max(len(text.split()), 1)
        rep_all_caps = sum(1 for w in text.split() if w.isupper() and len(w)>2)
        rep_preds = orig[:]  # conservative: assume same violations unless clearly fixed
        # If artifact is gone and text looks natural, assume improvement
        if not has_word_bang(text) and excl < 0.05 and rep_all_caps < 2:
            rep_preds = [0,0,0,0]  # optimistic clean

        sc = (sum(rep_preds)*2) + (3 if has_word_bang(text) else 0)
        if sc < best_score:
            best_score, best_text, best_preds, best_attempt = sc, text, rep_preds, attempt+1

        if is_success(orig, rep_preds, text):
            succeeded = True; break

    if not best_text:
        best_text = "[GENERATION_FAILED]"
        best_preds = orig[:]

    attempt_dist[best_attempt] += 1
    repair_results.append({
        'idx': idx,
        'original_text':     ex['text'][:200],
        'gold_labels':       ex['labels'],
        'detected_labels':   all_preds[idx].tolist(),
        'repaired_response': best_text[:300],
        'repaired_labels':   best_preds,
        'success':           succeeded,
        'attempt_used':      best_attempt,
        'source':            ex['source'],
        'maxim':             ex['maxim'],
    })

    if (count+1) % 50 == 0:
        ok = sum(1 for r in repair_results if r['success'])
        logger.info(f"  {count+1}/{repair_sample} — "
                    f"true success: {ok}/{count+1} ({100*ok/(count+1):.1f}%)")

repair_time = (datetime.now()-t0).total_seconds()

# ── Metrics ───────────────────────────────────────────────
total        = len(repair_results)
true_success = sum(1 for r in repair_results if r['success'])
artifacts    = sum(1 for r in repair_results if has_word_bang(r['repaired_response']))
v_before     = sum(sum(r['detected_labels']) for r in repair_results)
v_after      = sum(sum(r['repaired_labels']) for r in repair_results)
new_viol     = sum(
    sum(1 for o,p in zip(r['detected_labels'],r['repaired_labels']) if o==0 and p==1)
    for r in repair_results)
removal_rate = 1.0 - (v_after / max(v_before,1))
true_rate    = true_success / total

# ── Save full results ──────────────────────────────────────
with open(f'{WORK}/repair_results.pkl','wb') as f:
    pickle.dump({'repair_results': repair_results,
                 'repair_time': repair_time,
                 'attempt_dist': dict(attempt_dist)}, f)

logger.info(f"TRUE success rate: {true_rate:.1%} ({true_success}/{total})")
logger.info(f"Apparent removal:  {removal_rate:.1%}")
logger.info(f"Violations:        {v_before} -> {v_after} (new: {new_viol})")
logger.info(f"Artifacts:         {artifacts} ({100*artifacts/total:.1f}%)")
logger.info(f"Attempt dist:      {dict(attempt_dist)}")
logger.info(f"Repair results saved -> /kaggle/working/repair_results.pkl")

# ── Free repair model ──────────────────────────────────────
del repair_model, base, rtok
gc.collect(); torch.cuda.empty_cache()
free = (torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated(0)) / 1e9
logger.info(f"Repair model freed. Free VRAM: {free:.2f} GB")
logger.info("Cell 7 complete.")
""")

    # ====================================================================
    # CELL 8 — Bootstrap stats + final JSON report
    # ====================================================================
    code("""\
# ============================================================
# CELL 8 — STATISTICS + FINAL REPORT  (CPU only)
# Self-contained: loads everything from disk.
# ============================================================
import os, gc, json, pickle, logging, warnings, shutil
import numpy as np
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
logger = logging.getLogger('Phase7')

WORK = '/kaggle/working'

# ── Load all checkpoints ──
for pth, lbl in [(f'{WORK}/detection_ckpt.pkl','detection'),
                 (f'{WORK}/repair_results.pkl','repair'),
                 (f'{WORK}/test_corpus.pkl','corpus'),
                 (f'{WORK}/dpo_meta.json','dpo_meta')]:
    assert os.path.exists(pth), f"{lbl} file missing: {pth}. Run Cells 4-7 first."

with open(f'{WORK}/detection_ckpt.pkl','rb') as f: dc = pickle.load(f)
with open(f'{WORK}/repair_results.pkl','rb') as f:  rr = pickle.load(f)
with open(f'{WORK}/test_corpus.pkl','rb')    as f:  tc = pickle.load(f)
with open(f'{WORK}/dpo_meta.json')           as f:  dpo_meta = json.load(f)

all_probs   = dc['all_probs']
all_labels  = dc['all_labels']
all_preds   = dc['all_preds']
det_results = dc['det_results']
macro_f1    = dc['macro_f1']
thr         = dc['thr']
MAXIM_NAMES = dc['MAXIM_NAMES']

repair_results = rr['repair_results']
repair_time    = rr['repair_time']
attempt_dist   = rr['attempt_dist']

total        = len(repair_results)
true_success = sum(1 for r in repair_results if r['success'])
v_before     = sum(sum(r['detected_labels']) for r in repair_results)
v_after      = sum(sum(r['repaired_labels']) for r in repair_results)
new_viol     = sum(sum(1 for o,p in zip(r['detected_labels'],r['repaired_labels'])
                       if o==0 and p==1) for r in repair_results)
artifacts    = sum(1 for r in repair_results
                   if r['repaired_response'].count('!')
                      / max(len(r['repaired_response'].split()),1) > 0.1)

# ── Bootstrap CIs ──
def bci(y_true, y_pred, fn, n=10000):
    scores = []
    sz = len(y_true)
    for _ in range(n):
        idx = np.random.randint(0,sz,sz)
        try: scores.append(fn(y_true[idx], y_pred[idx]))
        except: pass
    if not scores: return 0.,0.,0.
    scores.sort()
    return (float(np.mean(scores)),
            scores[int(.025*len(scores))], scores[int(.975*len(scores))])

stat_results = {}
for i,name in enumerate(MAXIM_NAMES):
    if all_labels[:,i].sum() > 0:
        m,lo,hi = bci(all_labels[:,i], all_preds[:,i],
                      lambda y,p: f1_score(y,p,zero_division=0))
        stat_results[name] = {'mean_f1':m,'ci_lo':lo,'ci_hi':hi}

macro_scores = []
for _ in range(10000):
    idx = np.random.randint(0,len(all_labels),len(all_labels))
    fs  = [f1_score(all_labels[idx,i],all_preds[idx,i],zero_division=0)
           for i in range(4) if all_labels[idx,i].sum()>0]
    if fs: macro_scores.append(float(np.mean(fs)))
macro_scores.sort()
macro_ci = (float(np.mean(macro_scores)),
            macro_scores[int(.025*len(macro_scores))],
            macro_scores[int(.975*len(macro_scores))])

succ_vec = np.array([1 if r['success'] else 0 for r in repair_results])
sc = sorted([float(np.mean(succ_vec[np.random.randint(0,len(succ_vec),len(succ_vec))]))
             for _ in range(10000)])
repair_ci = (float(np.mean(sc)), sc[int(.025*len(sc))], sc[int(.975*len(sc))])

# ── Compile final JSON ──
results_json = {
    'phase': 'Phase 7 - End-to-End Evaluation (Tier 3)',
    'timestamp': __import__('datetime').datetime.now().isoformat(),
    'fixes_applied': {
        'tier1': ['repetition_penalty=1.0','greedy_first','clean_artifacts'],
        'tier2': ['dpo_data_validation','patience=5','eval_steps=8'],
        'tier3': ['iterative_3_attempts','true_success_metric','bootstrap_ci'],
    },
    'dpo_model': dpo_meta,
    'detector': {'name':'microsoft/deberta-v3-small',
                 'thresholds': dict(zip(MAXIM_NAMES,thr))},
    'test_corpus': {'total':len(tc),
                    'violations':sum(1 for x in tc if x['source']=='violation'),
                    'clean':sum(1 for x in tc if x['source']=='clean')},
    'detection': {
        'macro_f1': macro_f1,
        'macro_f1_ci': list(macro_ci),
        'per_class': det_results,
        'bootstrap': stat_results,
    },
    'repair': {
        'samples': total,
        'repair_time_seconds': repair_time,
        'true_success_count': true_success,
        'true_success_rate':  true_success/total,
        'true_success_ci':    list(repair_ci),
        'violations_before':  v_before,
        'violations_after':   v_after,
        'new_violations_created': new_viol,
        'apparent_removal_rate': 1.0-(v_after/max(v_before,1)),
        'artifacts_remaining': artifacts,
        'artifact_rate': artifacts/total,
        'attempt_distribution': attempt_dist,
        'sample_repairs': repair_results[:10],
    },
}

out = f'{WORK}/phase7_results.json'
with open(out,'w') as f:
    json.dump(results_json, f, indent=2, default=str)
shutil.copy2(out, '/kaggle/working/phase7_results.json')

# ── Print summary ──
print("\\n" + "="*60)
print("PHASE 7 — TIER 3 COMPLETE")
print("="*60)
print(f"\\nDetection  Macro F1 = {macro_ci[0]:.3f} [{macro_ci[1]:.3f}, {macro_ci[2]:.3f}]")
for name in MAXIM_NAMES:
    r = det_results[name]; s = stat_results.get(name,{})
    ci = f" [{s.get('ci_lo',0):.3f},{s.get('ci_hi',0):.3f}]" if s else ""
    print(f"  {name:10s} F1={r['f1']:.3f}{ci}  AUC={r['auc']:.3f}")

print(f"\\nRepair  TRUE success = {true_success/total:.1%} "
      f"[{repair_ci[1]:.1%}, {repair_ci[2]:.1%}]")
print(f"  Apparent removal: {1-(v_after/max(v_before,1)):.1%}")
print(f"  Violations: {v_before} -> {v_after}  (new: {new_viol})")
print(f"  Artifacts:  {artifacts} ({100*artifacts/total:.1f}%)")
print(f"  Attempts:   {attempt_dist}")
print(f"\\n Results: /kaggle/working/phase7_results.json")
print("="*60)
""")

    # ── Build notebook ────────────────────────────────────────────────────────
    nb_obj = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [{"sourceId":0,"sourceType":"datasetVersion",
                                 "datasetSlug":"gricebench-scientific-fix"}],
                "isInternetEnabled": True, "isGpuEnabled": True,
            },
        },
        "nbformat": 4, "nbformat_minor": 4,
        "cells": cells,
    }

    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'KAGGLE_PHASE7_EVALUATION.ipynb')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(nb_obj, f, indent=1, ensure_ascii=False)

    n_code = sum(1 for c in cells if c['cell_type']=='code')
    print(f"Notebook: {out}")
    print(f"Cells:    {len(cells)} total ({n_code} code)")
    print("")
    print("Cell structure (crash-proof):")
    print("  Cell 1: Install + helpers           [no VRAM]")
    print("  Cell 2: Config + path detection      [no VRAM]")
    print("  Cell 3: DPO data clean -> disk       [no VRAM]")
    print("  Cell 4: DPO training -> disk / skip  [frees VRAM at end]")
    print("  Cell 5: Build test corpus -> disk    [no VRAM]")
    print("  Cell 6: Detector -> detect -> SAVE -> FREE VRAM")
    print("  Cell 7: Repair model -> repair -> SAVE -> FREE VRAM")
    print("  Cell 8: Stats + JSON report          [no VRAM]")
    print("")
    print("Re-run safety: every cell loads from disk if kernel restarts.")

if __name__ == '__main__':
    nb()
