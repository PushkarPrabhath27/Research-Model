# Model Card: GriceBench Violation Detector

## Model Description

**Model Name:** GriceBench-Detector  
**Model Type:** Multi-label Text Classification  
**Base Architecture:** DeBERTa-v3-base  
**Parameters:** 184M  
**Languages:** English  
**License:** MIT

## Intended Use

The GriceBench Detector is a multi-label classifier designed to identify violations of Gricean Maxims in conversational responses:

- **Quantity**: Too much or too little information
- **Quality**: Factual inaccuracies or lack of evidence
- **Relation**: Off-topic or irrelevant responses
- **Manner**: Unclear, ambiguous, or disorganized responses

### Primary Use Cases

1. Dialogue system evaluation
2. Conversational AI quality assurance
3. Human-AI interaction analysis
4. Training data filtering for conversational models

### Out-of-Scope Uses

- Single-turn classification (requires context)
- Non-English text
- Non-conversational text (e.g., formal documents)
- Real-time classification (<10ms latency requirement)

## Training Data

### Source Datasets

- **Wizard of Wikipedia**: 1,320 examples
- **TopicalChat**: 1,890 examples  
- **LIGHT**: 1,302 examples
- **Total**: 4,512 examples (4,012 train, 500 validation)

### Data Augmentation

Synthetic violations injected via rule-based methods:
- Quantity: Verbosity/under-informativeness injection
- Quality: Fact contradiction/unsupported claims
- Relation: Topic drift/off-topic generation
- Manner: Sentence shuffling/jargon injection

### Annotation

- **Method**: Weak supervision + manual gold annotation
- **Annotators**: 2 trained annotators
- **Inter-annotator Agreement**: Cohen's κ = 0.82 (substantial agreement)
- **Label Distribution**:
  - Quantity: 25.6%
  - Quality: 14.0%
  - Relation: 21.3%
  - Manner: 31.6%

## Training Procedure

### Hyperparameters

```python
{
    "model": "microsoft/deberta-v3-base",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 10,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "dropout": 0.1,
    "optimizer": "AdamW",
    "scheduler": "linear_with_warmup",
    "loss": "BCEWithLogitsLoss"
}
```

### Training Infrastructure

- **Hardware**: NVIDIA V100 (16GB)
- **Training Time**: 2 hours
- **Framework**: PyTorch 2.1.0, Transformers 4.35.2

### Input Format

```
[CONTEXT] <dialogue_history> [RESPONSE] <response_to_evaluate>
```

Example:
```
[CONTEXT] [agent_1]: I love Star Wars! [agent_2]: Me too! Which movie is your favorite? [RESPONSE] The original trilogy is the best, especially Empire Strikes Back.
```

## Performance

### Validation Set (500 examples)

| Maxim | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Quantity | 100.0% | 100.0% | 1.000 |
| Quality | 91.7% | 94.3% | 0.930 |
| Relation | 100.0% | 100.0% | 1.000 |
| Manner | 94.9% | 93.1% | 0.940 |
| **Macro Avg** | **96.7%** | **96.9%** | **0.968** |

### Error Analysis

- **False Positives**: Primarily on borderline subjective cases (Quality, Manner)
- **False Negatives**: Subtle violations (Quality contradictions, Manner shuffling)
- **Exact Match Accuracy**: 94.2% (all 4 maxims correct simultaneously)

## Limitations

1. **English-only**: Not tested on other languages
2. **Conversational domain**: Performance may degrade on formal text
3. **Context dependency**: Requires dialogue history for accurate classification
4. **Manner subjectivity**: Lower F1 on subjective style violations
5. **Dataset bias**: Trained on Wizard/TopicalChat/LIGHT; may not generalize to all domains

## Bias and Fairness

- **Known Biases**: Dataset skewed toward informational dialogues (Wikipedia, topics)
- **Demographic Bias**: Not evaluated for demographic fairness
- **Recommendation**: Do not use for high-stakes decisions without human review

## Usage

### Installation

```bash
pip install transformers==4.35.2 torch==2.1.0
```

### Inference

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Load model (you'll need to define ViolationDetector class)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
detector = ViolationDetector("microsoft/deberta-v3-base")
detector.load_state_dict(torch.load("best_model.pt")["model_state_dict"])
detector.eval()

# Predict
text = "[CONTEXT] What is AI? [RESPONSE] Artificial Intelligence is the simulation of human intelligence."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = detector(inputs["input_ids"], inputs["attention_mask"])
    probs = outputs["probs"][0]  # [quantity, quality, relation, manner]

print(f"Quantity: {probs[0]:.3f}")
print(f"Quality: {probs[1]:.3f}")
print(f"Relation: {probs[2]:.3f}")
print(f"Manner: {probs[3]:.3f}")
```

## Citation

```bibtex
@inproceedings{gricebench2024,
  title={GriceBench: Operationalizing Gricean Maxims for Cooperative Dialogue Systems},
  author={Your Name},
  booktitle={Proceedings of the Conference},
  year={2024}
}
```

## Contact

- GitHub: https://github.com/yourusername/GriceBench
- Email: your.email@university.edu

---

**Version:** 1.0  
**Last Updated:** 2026-01-23
