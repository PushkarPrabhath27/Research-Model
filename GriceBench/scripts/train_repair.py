"""
GriceBench Repair Model - Chapter 10
====================================

T5-based model for repairing Gricean maxim violations.

Input: Violated response + violation type + context + evidence
Output: Repaired response (cooperative, adhering to maxims)

Based on Chapter 10-11 of the Implementation Guide.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AdamW,
    get_linear_schedule_with_warmup
)
from pathlib import Path
from typing import Dict, List
from tqdm.auto import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

class RepairConfig:
    """Repair model configuration."""
    # Model
    model_name = 't5-base'  # 220M parameters
    
    # Control tokens
    control_tokens_path = 'data_processed/repair_data/control_tokens.json'
    
    # Data
    train_data_path = 'data_processed/repair_data/repair_train.json'
    val_data_path = 'data_processed/repair_data/repair_val.json'
    
    # Training
    batch_size = 4  # T5 is memory-intensive
    learning_rate = 3e-4
    num_epochs = 5
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    
    # Sequence lengths
    max_input_length = 512
    max_target_length = 256
    
    # Device
    device = 'cuda' if torch.cuda is_available() else 'cpu'
    
    # Output
    model_save_path = 'models/repair/best_model'
    
    # Logging
    log_interval = 50


# ============================================================================
# DATASET
# ============================================================================

class RepairDataset(Dataset):
    """Dataset for repair model training."""
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer: T5Tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 256
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            example['input_text'],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            example['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Replace padding token id in labels with -100 (ignore in loss)
        labels = target_encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(0),
            'attention_mask': input_encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }


# ============================================================================
# REPAIR MODEL
# ============================================================================

class GriceBenchRepairModel:
    """T5-based repair model with custom control tokens."""
    
    def __init__(self, config: RepairConfig = None):
        self.config = config or RepairConfig()
        self.device = torch.device(self.config.device)
        
        # Load control tokens
        with open(self.config.control_tokens_path, 'r') as f:
            token_config = json.load(f)
        
        self.control_tokens = token_config['all_tokens']
        
        print(f"🤖 Initializing repair model on {self.device}")
    
    def load_model(self):
        """Load and initialize T5 model with custom tokens."""
        print(f"\n📥 Loading {self.config.model_name}...")
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
        
        # Add control tokens
        print(f"   Adding {len(self.control_tokens)} control tokens...")
        self.tokenizer.add_tokens(self.control_tokens)
        
        # Load model
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        
        # Resize embeddings to accommodate new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Move to device
        self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Model loaded: {total_params:,} parameters ({trainable_params:,} trainable)")
    
    def load_data(self):
        """Load training and validation data."""
        print(f"\n📂 Loading training data...")
        
        # Load train
        with open(self.config.train_data_path, 'r', encoding='utf-8') as f:
            train_examples = json.load(f)
        
        # Load val
        with open(self.config.val_data_path, 'r', encoding='utf-8') as f:
            val_examples = json.load(f)
        
        print(f"✅ Train: {len(train_examples):,} | Val: {len(val_examples):,}")
        
        # Create datasets
        self.train_dataset = RepairDataset(
            train_examples,
            self.tokenizer,
            self.config.max_input_length,
            self.config.max_target_length
        )
        
        self.val_dataset = RepairDataset(
            val_examples,
            self.tokenizer,
            self.config.max_input_length,
            self.config.max_target_length
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        print(f"   {len(self.train_loader)} train batches")
        print(f"   {len(self.val_loader)} val batches")
    
    def setup_training(self):
        """Setup optimizer and scheduler."""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"\n⚙️ Training setup:")
        print(f"   Total steps: {total_steps:,}")
        print(f"   Warmup steps: {warmup_steps:,}")
        print(f"   Learning rate: {self.config.learning_rate}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """Full training loop."""
        print("\n" + "="*70)
        print("🏋️ TRAINING REPAIR MODEL")
        print("="*70)
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'#'*70}")
            print(f"# EPOCH {epoch+1}/{self.config.num_epochs}")
            print(f"{'#'*70}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            history['train_loss'].append(train_loss)
            
            # Validate
            print(f"\n🔍 Validating...")
            val_loss = self.validate()
            history['val_loss'].append(val_loss)
            
            print(f"\n📊 Epoch {epoch+1} Results:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
                print(f"💾 SAVED! Best val loss: {val_loss:.4f}")
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"   Best val loss: {best_val_loss:.4f}")
        print("="*70)
        
        # Save history
        history_path = Path(self.config.model_save_path).parent / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n✅ Saved training history: {history_path}")
    
    def save_model(self):
        """Save model and tokenizer."""
        save_path = Path(self.config.model_save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def generate_repair(self, violated_text: str, max_length: int = 256) -> str:
        """Generate a repair for a violated response."""
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            violated_text,
            max_length=self.config.max_input_length,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        repair = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return repair


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Train the repair model."""
    print("="*70)
    print("GRICEBENCH REPAIR MODEL TRAINING - CHAPTER 10")
    print("="*70)
    
    # Initialize
    repair_model = GriceBenchRepairModel()
    
    # Load model
    repair_model.load_model()
    
    # Load data
    repair_model.load_data()
    
    # Setup training
    repair_model.setup_training()
    
    # Train
    repair_model.train()
    
    print("\n✅ Model saved to: models/repair/best_model/")
    print("\nNext steps:")
    print("  1. Evaluate repair model (scripts/evaluate_repair.py)")
    print("  2. Generate sample repairs")
    print("  3. Calculate repair metrics")


if __name__ == "__main__":
    main()
