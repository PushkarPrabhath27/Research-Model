# Model Card: GriceBench Repair Model

## Model Description

**Model Name:** GriceBench-Repair  
**Model Type:** Seq2Seq Text-to-Text Transformation  
**Base Architecture:** T5-base  
**Parameters:** 220M  
**Languages:** English  
**License:** MIT

## Intended Use

The GriceBench Repair Model is a sequence-to-sequence model that corrects violations of Gricean Maxims in conversational responses (excluding Relation violations, which require regeneration).

### Supported Repairs

- **Quantity**: Condense verbose responses or expand under-informative ones
- **Quality**: Fix factual inaccuracies given evidence  
- **Manner**: Improve clarity, remove jargon, reorder shuffled sentences

### Out-of-Scope

- **Relation repairs**: Use retrieval + regeneration instead
- **Multi-violation repair**: Currently handles one primary violation type at a time
- **Real-time streaming**: Model requires full response for editing

## Training Data

### Source

- **Total**: 3,611 repair pairs (3,210 train, 401 validation)
- **Derived from**: Wizard of Wikipedia, TopicalChat, LIGHT

### Violation Type Distribution

| Type | Count | Percentage |
|------|-------|------------|
| Quantity | 920 | 28.7% |
| Quality | 780 | 24.3% |
| Manner | 1,510 | 47.0% |

### Input Format

```
repair violation: <VIOLATION_TYPE> context: <context> response: <faulty_response>
```

Example:
```
repair violation: MANNER context: What is quantum computing? response: Quantum comp. uses q-bits for parallel proc. via superpos. and entang.
```

### Output Format

The corrected response:
```
Quantum computing uses quantum bits (qubits) for parallel processing through superposition and entanglement.
```

## Training Procedure

### Hyperparameters

```python
{
    "model": "t5-base",
    "max_input_length": 512,
    "max_target_length": 256,
    "batch_size": 8,
    "learning_rate": 3e-5,
    "epochs": 5,
    "warmup_steps": 300,
    "weight_decay": 0.01,
    "optimizer": "AdamW",
    "label_smoothing": 0.1
}
```

### Training Infrastructure

- **Hardware**: NVIDIA V100 (16GB)
- **Training Time**: 3 hours
- **Framework**: PyTorch 2.1.0, Transformers 4.35.2

## Performance

### Validation Set (401 examples)

| Violation Type | BLEU | ROUGE-L | Edit Distance | Success Rate |
|----------------|------|---------|---------------|--------------|
| Quantity | 45.2 | 62.8 | 8.3 tokens | 91.2% |
| Quality | 38.7 | 58.1 | 12.1 tokens | 87.5% |
| Manner | 52.1 | 68.4 | 6.7 tokens | 93.8% |
| **Overall** | **46.8** | **64.2** | **8.9** | **91.3%** |

**Success Rate:** Percentage of repaired responses re-classified as cooperative by the detector.

### Qualitative Examples

**Quantity (Verbose → Concise):**
- **Input**: "Sharks are fascinating creatures that live in the ocean and have been around for millions of years evolving into apex predators with incredible hunting abilities using electroreception and acute senses..."
- **Output**: "Sharks are apex predators that have evolved for millions of years with electroreception and acute senses."

**Manner (Jargon → Clear):**
- **Input**: "NLP utilizes ML algos for text proc."
- **Output**: "Natural Language Processing uses machine learning algorithms for text processing."

## Limitations

1. **Relation violations**: Cannot repair off-topic responses (requires regeneration)
2. **Context dependence**: Quality repairs may fail without sufficient context/evidence
3. **Over-correction**: Sometimes removes stylistic variation in Manner repairs
4. **Multi-violation**: Performance degrades when multiple maxims violated simultaneously
5. **Domain shift**: Trained on informational dialogues; may not generalize to all domains

## Bias and Fairness

- **Known Biases**: Training data skewed toward informational/factual corrections
- **Not evaluated for**: Demographic fairness, cultural sensitivity
- **Recommendation**: Human review for public-facing applications

## Usage

### Installation

```bash
pip install transformers==4.35.2 torch==2.1.0
```

### Inference

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model
tokenizer = T5Tokenizer.from_pretrained("models/repair")
model = T5ForConditionalGeneration.from_pretrained("models/repair")
model.eval()

# Prepare input
context = "What is quantum computing?"
response = "Quantum comp. uses q-bits for parallel proc."
violation = "MANNER"

input_text = f"repair violation: {violation} context: {context} response: {response}"
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate repair
outputs = model.generate(**inputs, max_new_tokens=150, num_beams=4)
repaired = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(repaired)
# Output: "Quantum computing uses quantum bits (qubits) for parallel processing."
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
