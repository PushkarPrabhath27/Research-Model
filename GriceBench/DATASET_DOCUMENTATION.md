# GriceBench Datasets Documentation

## Overview

GriceBench uses three source datasets to create training data for detecting, repairing, and generating cooperative dialogue responses.

## Source Datasets

### 1. Wizard of Wikipedia

- **Version**: 1.0
- **Source**: https://parl.ai/projects/wizard_of_wikipedia/
- **License**: CC BY-SA 4.0
- **Size**: ~22,000 dialogues
- **Domain**: Knowledge-grounded conversations
- **Usage in GriceBench**: 28% of training data

**Characteristics:**
- Information-seeking conversations
- Wikipedia-grounded responses
- High factual density
- Formal tone

**Example:**
```json
{
  "context": ["I love Star Wars!", "Me too! Which is your favorite?"],
  "response": "The original trilogy is considered the best by many fans. Empire Strikes Back won multiple awards.",
  "evidence": ["FS1: The Empire Strikes Back won the Hugo Award for Best Dramatic Presentation."]
}
```

### 2. TopicalChat

- **Version**: 1.0
- **Source**: https://github.com/alexa/Topical-Chat
- **License**: Amazon Research License
- **Size**: ~11,000 conversations
- **Domain**: Open-domain chitchat with facts
- **Usage in GriceBench**: 41% of training data

**Characteristics:**
- Casual conversational tone
- Mix of facts and opinions
- Diverse topics (politics, entertainment, sports)
- Natural topic transitions

**Example:**
```json
{
  "context": ["Do you like basketball?", "Yes! I watch the NBA regularly."],
  "response": "The Lakers are my favorite team. They've won 17 championships, tied with the Celtics for most in NBA history.",
  "topic": "Sports"
}
```

### 3. LIGHT

- **Version**: 1.0
- **Source**: https://parl.ai/projects/light/
- **License**: MIT
- **Size**: ~11,000 dialogues
- **Domain**: Fantasy role-playing conversations
- **Usage in GriceBench**: 31% of training data

**Characteristics:**
- Character-based dialogue
- Creative/fictional settings
- Less factual, more narrative
- Informal language

**Example:**
```json
{
  "context": ["Greetings, traveler!", "Hello, kind wizard!"],
  "response": "I have been studying the ancient texts in this tower for many years. What brings you here?",
  "character": "Wizard",
  "location": "Tower"
}
```

---

## Processed Datasets

### Detector Training Data

**File:** `data_processed/detector_data/detector_train.json`  
**Size:** 4,012 examples  
**Format:** Multi-label classification

**Schema:**
```json
{
  "id": "unique_identifier",
  "input_text": "[CONTEXT] <dialogue_history> [RESPONSE] <response>",
  "labels": {
    "quantity": 0,  // 0 or 1
    "quality": 1,
    "relation": 0,
    "manner": 0
  },
  "violation_type": "quality_contradiction",  // Specific violation subtype
  "source_dataset": "wizard",
  "is_synthetic": true  // Whether violation was injected
}
```

**Label Distribution:**
| Maxim | Positive Examples | Percentage |
|-------|-------------------|------------|
| Quantity | 1,028 | 25.6% |
| Quality | 562 | 14.0% |
| Relation | 854 | 21.3% |
| Manner | 1,268 | 31.6% |
| Clean (all 0) | 1,300 | 32.4% |

**Violation Types:**
- **Quantity**: `verbose`, `underinformative`
- **Quality**: `quality_contradiction`, `quality_unsupported`
- **Relation**: `offtopic`, `topic_drift`
- **Manner**: `manner_jargon`, `manner_shuffled`, `manner_ambiguous`

---

### Repair Training Data

**File:** `data_processed/repair_data/repair_train.json`  
**Size:** 3,210 examples  
**Format:** Seq2seq (input → output pairs)

**Schema:**
```json
{
  "id": "unique_identifier",
  "input": "repair violation: MANNER context: What is AI? response: AI is comp. sys. that mimics hum. intell.",
  "target": "AI is computer systems that mimic human intelligence.",
  "violation_type": "MANNER",
  "source_dataset": "topicalchat",
  "bleu_score": 52.3  // Expected BLEU for this repair
}
```

**Distribution by Violation:**
| Violation | Examples | Percentage |
|-----------|----------|------------|
| Quantity | 920 | 28.7% |
| Quality | 780 | 24.3% |
| Manner | 1,510 | 47.0% |

**Note:** Relation violations excluded (require regeneration, not repair).

---

### DPO Preference Data

**File:** `data_processed/dpo_data/dpo_train.json`  
**Size:** 8,120 pairs  
**Format:** Preference pairs (chosen vs rejected)

**Schema:**
```json
{
  "id": "unique_identifier",
  "context": "What is quantum computing?",
  "chosen": "Quantum computing uses qubits that can exist in superposition, enabling parallel computation for certain problem classes.",
  "rejected": "Quantum comp. is diff. from classical and uses quantum mech. for processing info.",
  "chosen_violations": {
    "quantity": 0,
    "quality": 0,
    "relation": 0,
    "manner": 0
  },
  "rejected_violations": {
    "quantity": 0,
    "quality": 0,
    "relation": 0,
    "manner": 1  // Manner violation (abbreviations, unclear)
  },
  "source_dataset": "wizard"
}
```

**Preference Criteria:**
- **Chosen**: Cooperative (0 violations)
- **Rejected**: ≥1 violation detected

**Violation Distribution (Rejected):**
| Maxim | Count | Percentage |
|-------|-------|------------|
| Quantity | 2,340 | 28.8% |
| Quality | 1,628 | 20.0% |
| Relation | 1,792 | 22.1% |
| Manner | 2,360 | 29.1% |

---

## Relation Repair Corpus

**File:** `data_processed/relation_repair/response_corpus.json`  
**Size:** ~50,000 responses  
**Format:** Retrieval corpus for Relation violation handling

**Schema:**
```json
{
  "id": "unique_identifier",
  "response": "The Eiffel Tower was completed in 1889 for the World's Fair.",
  "topic": "Architecture",
  "keywords": ["Eiffel Tower", "1889", "Paris", "engineering"],
  "source_dataset": "wizard",
  "embedding": [0.123, -0.456, ...]  // 384-dim sentence-transformers embedding
}
```

**FAISS Index:**
- **File:** `data_processed/relation_repair/faiss_index.bin`
- **Dimension:** 384
- **Index Type:** Flat (exact search)
- **Size:** ~200MB

---

## Data Statistics

### Overall Summary

| Split | Detector | Repair | DPO | Total Unique |
|-------|----------|--------|-----|--------------|
| Train | 4,012 | 3,210 | 8,120 | ~12,000 |
| Validation | 500 | 401 | 1,015 | ~1,500 |
| **Total** | **4,512** | **3,611** | **9,135** | **~13,500** |

### Source Distribution

| Dataset | Examples | Percentage |
|---------|----------|------------|
| Wizard of Wikipedia | 3,780 | 28% |
| TopicalChat | 5,535 | 41% |
| LIGHT | 4,185 | 31% |

### Synthetic vs Organic

- **Synthetic Violations**: 8,200 (61%)
- **Organic Examples**: 5,300 (39%)

**Synthetic Generation Methods:**
1. Violation injectors (rule-based)
2. Weak supervision (heuristic labeling)
3. Manual gold annotation (500 examples)

---

## Data Access

### Download Instructions

```bash
# Clone repository
git clone https://github.com/yourusername/GriceBench.git
cd GriceBench

# Download processed data (~500MB)
python scripts/download_data.py --all

# Or download individually
python scripts/download_data.py --detector
python scripts/download_data.py --repair
python scripts/download_data.py --dpo
```

### Preprocessing Pipeline

To recreate from raw data:

```bash
# 1. Download raw datasets
python scripts/download_data.py --raw

# 2. Prepare detector data
python scripts/prepare_detector_data.py \\
    --data_dir data_raw \\
    --output data_processed/detector_data

# 3. Prepare repair data
python scripts/prepare_repair_data.py \\
    --data_dir data_raw \\
    --output data_processed/repair_data

# 4. Prepare DPO data
python scripts/prepare_dpo_data.py \\
    --data_dir data_raw \\
    --output data_processed/dpo_data

# 5. Build relation corpus
python scripts/create_response_corpus.py \\
    --data_dir data_processed \\
    --output data_processed/relation_repair
```

---

## Data Quality

### Annotation Quality

- **Inter-annotator Agreement**: Cohen's κ = 0.82
- **Annotation Time**: ~45 seconds per example (average)
- **Annotator Training**: 2 hours + 50 calibration examples

### Data Filtering

Applied filters:
1. **Length**: 10-150 tokens (response)
2. **Language**: English detection (langdetect)
3. **Toxicity**: Perspective API score <0.5
4. **Duplicates**: Removed exact/near duplicates

**Retention Rate**: 87% (post-filtering)

---

## Data Limitations

1. **Domain Bias**: Skewed toward informational/factual dialogue
2. **Synthetic Noise**: Injected violations may not reflect natural errors
3. **English-only**: No multilingual support
4. **Temporal Bias**: Data from 2018-2020
5. **Coverage**: Not exhaustive of all violation types

---

## Ethical Considerations

- **Privacy**: Datasets anonymized, no personal information
- **Bias**: Potential demographic biases not evaluated
- **Intended Use**: Research purposes only
- **Misuse Potential**: Not for malicious content generation

---

## Citation

```bibtex
@inproceedings{gricebench2024,
  title={GriceBench: Operationalizing Gricean Maxims for Cooperative Dialogue Systems},
  author={Your Name},
  booktitle={Proceedings of the Conference},
  year={2024}
}
```

---

**Last Updated:** 2026-01-23
