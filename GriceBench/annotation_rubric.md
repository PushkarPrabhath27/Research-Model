# GriceBench Annotation Rubric

## Purpose
This rubric defines how to annotate responses for Gricean maxim violations.
Use this guide to label examples consistently.

---

## Quantity Maxim
*"Say enough, but not too much"*

| Score | Label | Description |
|-------|-------|-------------|
| **0** | No Violation | Response provides appropriate amount of information |
| **1** | Too Little | Missing key information; user would need follow-ups |
| **2** | Too Much | Unnecessary repetition or tangents |

**Edge Case:** If question is ambiguous about detail level, default to 0 unless extreme.

---

## Quality Maxim
*"Only say what is true and supported"*

| Score | Label | Description |
|-------|-------|-------------|
| **0** | No Violation | All claims supported by evidence or common knowledge |
| **1** | Violation | Contains unsupported or contradicted claims |

**Edge Cases:**
- Opinions are NOT violations (not factual claims)
- Paraphrasing is OK if meaning preserved

---

## Relation Maxim
*"Stay on topic"*

| Score | Label | Description |
|-------|-------|-------------|
| **0** | No Violation | Response addresses the question |
| **1** | Violation | Response fails to answer what was asked |

**Edge Case:** Asking for clarification is NOT a violation.

---

## Manner Maxim
*"Be clear and organized"*

| Score | Label | Description |
|-------|-------|-------------|
| **0** | No Violation | Response is clear and appropriate for audience |
| **1** | Violation | Confusing, poor organization, ambiguous references |

**Edge Case:** Complex topics may require complex language. Judge relative to topic complexity.

---

## Annotation Workflow
1. Read the **context** (user's question)
2. Read the **evidence** (available facts)
3. Read the **response** (what to evaluate)
4. For each maxim, assign a score (0, 1, or 2 for Quantity)
5. If unsure, mark as "uncertain" in notes
