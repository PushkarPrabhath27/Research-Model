# Annotation Decision Rules

## Purpose

This document defines clear rules for annotating GriceBench examples for Gricean maxim violations. Following these guidelines ensures consistency in human annotation per the scientific improvement plan.

## The Four Maxims

### 1. Quantity (Information Amount)

**Question**: Does the response provide the right amount of information?

| Rating | Definition | Examples |
|--------|------------|----------|
| **Too Little** | Missing key information needed to answer the question | "Yes.", "I'm not sure.", "That's interesting." |
| **Appropriate** | Provides sufficient information without excess | Clear, complete answer to the question |
| **Too Much** | Includes unnecessary tangents, repetition, or excessive detail | Repeating the same point multiple ways, irrelevant stories |

**Decision Rule**:
- If the user would need to ask a follow-up to get basic info → **Too Little**
- If the response answers the question completely → **Appropriate**
- If you could remove 30%+ of content without losing meaning → **Too Much**

---

### 2. Quality (Truthfulness)

**Question**: Does the response make unsupported or contradictory claims?

| Rating | Definition | Examples |
|--------|------------|----------|
| **Unsupported** | Makes claims without evidence, uses weasel words, contradicts itself | "Studies show...", "Many believe...", "It's well known that..." |
| **Appropriate** | Claims are either supported or appropriately hedged | "I think...", "Based on [source]...", factual statements |

**Decision Rule**:
- If it cites "studies" or "experts" without specifics → **Unsupported**
- If it makes confident claims about uncertain things → **Unsupported**
- If claims are clearly common knowledge or properly hedged → **Appropriate**

---

### 3. Relation (Relevance)

**Question**: Does the response address the question asked?

| Rating | Definition | Examples |
|--------|------------|----------|
| **Off-topic** | Completely unrelated to the question | Answering about cooking when asked about programming |
| **Tangential** | Related topic but doesn't answer the specific question | Asked about X, answers about Y which is related to X |
| **Relevant** | Directly addresses the question | Clear answer to what was asked |

**Decision Rule**:
- If the topic is completely different → **Off-topic**
- If same general topic but wrong aspect → **Tangential**
- If it answers what was asked → **Relevant**

---

### 4. Manner (Clarity)

**Question**: Is the response clear and well-organized?

| Rating | Definition | Examples |
|--------|------------|----------|
| **Unclear** | Ambiguous, disorganized, jargon-heavy, hard to follow | Run-on sentences, unclear pronouns, unnecessary complexity |
| **Clear** | Easy to understand, well-structured | Readable prose, logical flow |

**Decision Rule**:
- If you had to re-read to understand → **Unclear**
- If important info is buried at the end → **Unclear**
- If pronouns are ambiguous (what does "it" refer to?) → **Unclear**
- If easily understood on first read → **Clear**

---

## Helpfulness Rating

**Scale**: 1-5

| Rating | Meaning |
|--------|---------|
| 1 | Not helpful at all - would frustrate a user |
| 2 | Minimally helpful - some relevant info but poor |
| 3 | Moderately helpful - answers partially |
| 4 | Helpful - good answer with minor issues |
| 5 | Very helpful - excellent answer |

---

## Edge Cases

### Multiple Violations
If a response has multiple issues, mark ALL that apply:
- Verbose AND unclear → Quantity: Too Much + Manner: Unclear
- Off-topic AND factually wrong → Relation: Off-topic + Quality: Unsupported

### Appropriate Hedging vs Unsupported
- "I think this might be true" → Appropriate (properly hedged)
- "Studies show this is true" → Unsupported (claims authority without source)

### Short vs Under-informative
- "No" when asked a yes/no question → Appropriate
- "No" when asked "Why did X happen?" → Too Little

---

## Self-Consistency Check

Every 50 annotations:
1. Re-annotate 5 random previous examples
2. Check if your judgments match
3. If >1 differs, review and update rules
4. Document any rule refinements

---

## Remember

- When in doubt, trust your first instinct
- Imagine a real user receiving this response
- Would YOU be satisfied with this answer?
