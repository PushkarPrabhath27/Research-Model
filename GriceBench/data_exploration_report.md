# GriceBench Data Exploration Report

## Dataset Overview

### Topical-Chat Statistics
| Metric | Value |
|--------|-------|
| Conversations | 8,628 |
| Total turns | 188,378 |
| Mean response length | 19.6 words |
| Median response length | 18.0 words |
| Vocabulary size | 86,108 unique words |
| Mean turns/conversation | 21.8 |
| Knowledge usage rate | 100.0% |

---

## Response Length Distribution (For Quantity Violations)

| Percentile | Words |
|------------|-------|
| 10th | 8 |
| 25th | 12 |
| 50th (median) | 18 |
| 75th | 25 |
| 90th | 33 |
| 95th | 38 |
| 99th | 51 |

**Recommended Quantity Violation Thresholds:**
- **Too short (under-informative):** < 8 words (10th percentile)
- **Too long (over-informative):** > 38 words (95th percentile)

---

## Data Structure

### Conversation Format
Each conversation in Topical-Chat contains:
- `article_url`: Source article URL
- `config`: Configuration metadata
- `content`: Array of turns (the actual dialogue)
- `conversation_rating`: Quality rating

### Turn Format
Each turn contains:
- `agent`: Speaker identifier (agent_1 or agent_2)
- `message`: The actual text response
- `knowledge_source`: Reference to knowledge snippet used

### Reading Sets (Evidence)
Reading sets contain background knowledge snippets that speakers can reference. Each conversation has associated reading sets with factual information used to ground responses.

---

## Key Observations for GriceBench

### Quantity Maxim
- Mean response is ~20 words, very consistent across the dataset
- Responses shorter than 8 words are unusually brief
- Responses longer than 38 words are unusually verbose
- Use these as thresholds for under/over-informative detection

### Quality Maxim  
- 100% of messages have knowledge_source field
- This enables objective Quality violation detection by comparing responses to evidence
- Can use NLI (Natural Language Inference) to check for contradictions

### Relation Maxim
- Conversations are topical (each has associated article)
- Can measure semantic similarity between questions and responses
- Off-topic responses should have low cosine similarity

### Manner Maxim
- Vocabulary is diverse (86K unique words)
- Can use readability scores (Flesch-Kincaid)
- Can track pronoun ambiguity and sentence complexity

---

## Next Steps
1. Create (context, evidence, response) extraction functions
2. Build violation injection pipeline for each maxim
3. Generate weak supervision heuristics
4. Create training dataset with synthetic violations
