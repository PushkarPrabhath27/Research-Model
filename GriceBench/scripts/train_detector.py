"""
GriceBench Violation Detector - Model and Training
===================================================

This module implements the Gricean violation detector:
- DeBERTa encoder + multi-label classification head
- Two-phase training: weak supervision → gold fine-tuning
- Evaluation metrics: per-maxim F1, exact match

Based on Chapter 8 of the GriceBench Implementation Guide.

IMPORTANT: This script is designed to run on Google Colab with GPU.
Write and test locally, then upload to Google Drive and run on Colab.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

# These imports require GPU environment (Colab)
try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        get_linear_schedule_with_warmup
    )
    # Use PyTorch's AdamW (transformers removed theirs in newer versions)
    from torch.optim import AdamW
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "microsoft/deberta-v3-base"  # or "roberta-base"
    num_labels: int = 4  # One per maxim
    max_length: int = 512
    
    # Training
    batch_size: int = 8  # Reduce if OOM
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Paths
    train_path: str = "data_processed/detector_data/detector_train.json"
    val_path: str = "data_processed/detector_data/detector_val.json"
    output_dir: str = "models/detector"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    maxims: Tuple[str, ...] = ('quantity', 'quality', 'relation', 'manner')


# ============================================================================
# DATASET
# ============================================================================

class ViolationDataset(Dataset):
    """Dataset for violation detection."""
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 512,
        maxims: Tuple[str, ...] = ('quantity', 'quality', 'relation', 'manner')
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.maxims = maxims
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        encoding = self.tokenizer(
            example['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Extract labels
        labels = example.get('labels', {})
        label_tensor = torch.tensor(
            [labels.get(m, 0) for m in self.maxims],
            dtype=torch.float
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label_tensor
        }


# ============================================================================
# MODEL
# ============================================================================

class ViolationDetector(nn.Module):
    """
    Multi-label classifier for Gricean violation detection.
    
    Architecture:
    - Pre-trained encoder (DeBERTa/RoBERTa)
    - Pooler (CLS token representation)
    - Classification head (linear layer with sigmoid)
    """
    
    def __init__(self, model_name: str, num_labels: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Load pre-trained encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Sigmoid for multi-label
        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Dict with 'logits', 'probabilities', and optionally 'loss'
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool (use CLS token)
        pooled = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        
        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [batch, num_labels]
        probs = self.sigmoid(logits)
        
        result = {'logits': logits, 'probabilities': probs}
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            result['loss'] = loss
        
        return result


# ============================================================================
# TRAINING
# ============================================================================

class DetectorTrainer:
    """Trainer for the violation detector."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Initialize model
        self.model = ViolationDetector(
            model_name=config.model_name,
            num_labels=config.num_labels
        ).to(self.device)
        
        print(f"Model loaded: {config.model_name}")
    
    def load_data(self, train_path: str, val_path: str) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare data."""
        # Load JSON
        with open(train_path, 'r', encoding='utf-8') as f:
            train_examples = json.load(f)
        with open(val_path, 'r', encoding='utf-8') as f:
            val_examples = json.load(f)
        
        print(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples")
        
        # Create datasets
        train_dataset = ViolationDataset(
            train_examples,
            self.tokenizer,
            max_length=self.config.max_length,
            maxims=self.config.maxims
        )
        val_dataset = ViolationDataset(
            val_examples,
            self.tokenizer,
            max_length=self.config.max_length,
            maxims=self.config.maxims
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Run training loop."""
        # Optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0.0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print('='*50)
            
            # Train
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            print(f"Train loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_metrics = self._validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_metrics['macro_f1'])
            
            print(f"Val loss: {val_loss:.4f}")
            print(f"Val F1 (macro): {val_metrics['macro_f1']:.4f}")
            for maxim, f1 in val_metrics['per_maxim_f1'].items():
                print(f"  {maxim}: {f1:.4f}")
            
            # Save best model
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
                self._save_checkpoint(epoch, val_metrics)
                print(f"  → New best model saved!")
        
        print(f"\nTraining complete! Best F1: {best_val_f1:.4f}")
        return history
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer,
        scheduler
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress = tqdm(train_loader, desc="Training")
        for batch in progress:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                total_loss += outputs['loss'].item()
                
                # Threshold at 0.5
                preds = (outputs['probabilities'] > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_preds, all_labels)
        
        return total_loss / len(val_loader), metrics
    
    def _calculate_metrics(self, preds, labels) -> Dict:
        """Calculate F1 scores per maxim and macro."""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        per_maxim_f1 = {}
        for i, maxim in enumerate(self.config.maxims):
            f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
            per_maxim_f1[maxim] = f1
        
        macro_f1 = sum(per_maxim_f1.values()) / len(per_maxim_f1)
        
        # Exact match
        exact_match = (preds == labels).all(axis=1).mean()
        
        return {
            'per_maxim_f1': per_maxim_f1,
            'macro_f1': macro_f1,
            'exact_match': exact_match
        }
    
    def _save_checkpoint(self, epoch: int, metrics: Dict) -> None:
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = output_dir / f"checkpoint_best.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics
        }, checkpoint_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Train the violation detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train violation detector')
    parser.add_argument('--train', type=str, default='data_processed/detector_data/detector_train.json')
    parser.add_argument('--val', type=str, default='data_processed/detector_data/detector_val.json')
    parser.add_argument('--output', type=str, default='models/detector')
    parser.add_argument('--model', type=str, default='microsoft/deberta-v3-base')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    
    args = parser.parse_args()
    
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers library not available!")
        return
    
    # Create config
    config = TrainingConfig(
        model_name=args.model,
        train_path=args.train,
        val_path=args.val,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print("="*50)
    print("GRICEBENCH VIOLATION DETECTOR TRAINING")
    print("="*50)
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    
    # Initialize trainer
    trainer = DetectorTrainer(config)
    
    # Load data
    train_loader, val_loader = trainer.load_data(args.train, args.val)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Save history
    history_path = Path(args.output) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining history saved to {history_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
