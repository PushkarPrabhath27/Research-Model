import json

notebook = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# DPO Training - 411 Clean Pairs\n", "\n", "**Steps:**\n", "1. Enable GPU T4 x2\n", "2. Add dataset: gricebench-clean-dpo\n", "3. Run Cell 1, then RESTART kernel\n", "4. Run Cell 2 (training + save)"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["# CELL 1: Install packages\n", "# Use latest trl to match Kaggle's transformers\n", "!pip install -q trl peft bitsandbytes\n", "print('==='*20)\n", "print('RESTART KERNEL NOW: Runtime -> Restart session')\n", "print('Then run Cell 2')\n", "print('==='*20)"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# CELL 2: Complete Training\n",
                "import warnings; warnings.filterwarnings('ignore')\n",
                "import os, json, torch, shutil\n",
                "os.environ['TRANSFORMERS_VERBOSITY'] = 'error'\n",
                "os.environ['TRL_USE_RICH'] = '0'\n",
                "\n",
                "from datasets import Dataset\n",
                "from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments\n",
                "from peft import LoraConfig, prepare_model_for_kbit_training\n",
                "from trl import DPOTrainer\n",
                "\n",
                "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\n",
                "\n",
                "# Load data\n",
                "DATA = '/kaggle/input/gricebench-clean-dpo/clean_dpo_pairs.json'\n",
                "if not os.path.exists(DATA):\n",
                "    print('ERROR: Dataset not found. Available:')\n",
                "    for d in os.listdir('/kaggle/input'): print(f'  /kaggle/input/{d}')\n",
                "    raise FileNotFoundError(DATA)\n",
                "\n",
                "with open(DATA) as f: pairs = json.load(f)\n",
                "print(f'Loaded {len(pairs)} pairs')\n",
                "\n",
                "formatted = [{'prompt':p['prompt'],'chosen':p['chosen'],'rejected':p['rejected']} for p in pairs]\n",
                "ds = Dataset.from_list(formatted).train_test_split(test_size=0.1, seed=42)\n",
                "print(f'Train: {len(ds[\"train\"])}, Eval: {len(ds[\"test\"])}')\n",
                "\n",
                "# Load model\n",
                "MODEL = 'HuggingFaceTB/SmolLM2-360M-Instruct'\n",
                "model = AutoModelForCausalLM.from_pretrained(MODEL, load_in_4bit=True, device_map='auto')\n",
                "model = prepare_model_for_kbit_training(model)\n",
                "tok = AutoTokenizer.from_pretrained(MODEL)\n",
                "tok.pad_token = tok.eos_token\n",
                "print('Model loaded')\n",
                "\n",
                "# Configure\n",
                "lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')\n",
                "args = TrainingArguments(\n",
                "    output_dir='/kaggle/working/dpo',\n",
                "    num_train_epochs=5,\n",
                "    per_device_train_batch_size=2,\n",
                "    gradient_accumulation_steps=4,\n",
                "    learning_rate=5e-5,\n",
                "    logging_steps=20,\n",
                "    save_steps=100,\n",
                "    fp16=True,\n",
                "    report_to='none'\n",
                ")\n",
                "\n",
                "# Train\n",
                "trainer = DPOTrainer(\n",
                "    model=model,\n",
                "    ref_model=None,\n",
                "    beta=0.1,\n",
                "    train_dataset=ds['train'],\n",
                "    eval_dataset=ds['test'],\n",
                "    tokenizer=tok,\n",
                "    peft_config=lora,\n",
                "    args=args,\n",
                "    max_length=512,\n",
                "    max_prompt_length=256\n",
                ")\n",
                "\n",
                "print('Starting training...')\n",
                "trainer.train()\n",
                "print('Training complete!')\n",
                "\n",
                "# Save\n",
                "trainer.save_model('/kaggle/working/final')\n",
                "tok.save_pretrained('/kaggle/working/final')\n",
                "shutil.make_archive('/kaggle/working/dpo_model', 'zip', '/kaggle/working/final')\n",
                "print('==='*20)\n",
                "print('DONE! Download: /kaggle/working/dpo_model.zip')\n",
                "print('==='*20)"
            ]
        }
    ]
}

path = 'c:/Users/pushk/OneDrive/Documents/Research Model/GriceBench/KAGGLE_DPO_FINAL.ipynb'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
print('Created:', path)
