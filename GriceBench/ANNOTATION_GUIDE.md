# GriceBench Annotation Guide for Beginners

## What You're Doing

You have 500 pairs of AI responses. For each pair, you decide: **Which response is better?**

Your judgment trains the AI to be more "cooperative" in conversations.

---

## Step 1: Get Your Files

### Download from Kaggle
1. Open your Kaggle notebook (the one you ran earlier)
2. Look at the **right panel** → Click **"Output"** tab
3. Find `tier1_hard_pairs.json` (or `preference_pairs_1500.json`)
4. Click **⋮ (three dots)** → **Download**

### Save to Your Computer
Save it somewhere easy to find, like your Desktop.

---

## Step 2: Open the Annotation Interface

1. Find the file: `GriceBench/scripts/annotation_interface.html`
2. **Double-click** to open in your web browser
3. You'll see a dark interface with "GriceBench Fast Annotator"

---

## Step 3: Load Your Data

1. Click the green **"📂 Load JSON"** button
2. Select the `tier1_hard_pairs.json` file you downloaded
3. The first pair will appear on screen

---

## Step 4: How to Annotate

### What You See

```
┌─────────────────────────────────────────────────────────┐
│ 📝 Context                                               │
│ "What's your favorite movie?"                           │
│ "I really like action films!"                           │
│ "Me too! Any recommendations?"                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────┐  ┌─────────────────────┐
│ Response A          │  │ Response B          │
│                     │  │                     │
│ "You should watch   │  │ "Well, movies are   │
│ John Wick, it's     │  │ interesting. There  │
│ action-packed!"     │  │ are many types."    │
└─────────────────────┘  └─────────────────────┘
```

### Your Decision Process

**Read the Context first** - What is the conversation about? What was asked?

**Read Response A** - Ask yourself:
- Does it answer the question? ✓ or ✗
- Is it helpful? ✓ or ✗
- Is it clear? ✓ or ✗

**Read Response B** - Same questions

**Compare** - Which would you prefer if you were the person asking?

### Use Keyboard Shortcuts (FAST!)

| Key | Meaning | When to Use |
|-----|---------|-------------|
| `1` | A is MUCH better | A clearly answers, B doesn't |
| `2` | A is slightly better | A is a bit better |
| `3` | Equal | Both equally good/bad |
| `4` | B is slightly better | B is a bit better |
| `5` | B is MUCH better | B clearly answers, A doesn't |
| `→` | Go to next pair | After you decide |
| `←` | Go back | If you made a mistake |
| `S` | Save progress | Every 50 pairs or so |

---

## Step 5: Decision Criteria (The 4 Gricean Maxims)

### 1. QUANTITY - Is it the right amount of info?

**Too little** (Bad):
> Context: "How do I bake a cake?"
> Response: "You need ingredients."

**Too much** (Bad):
> Context: "What's the capital of France?"
> Response: "Paris is the capital of France, which brings to mind the French Revolution of 1789 when..."

**Just right** (Good):
> Context: "What's the capital of France?"
> Response: "Paris is the capital of France."

---

### 2. QUALITY - Is it truthful and supported?

**Bad - Unsupported claims**:
> "Studies show that 87% of people agree." (No source!)
> "Scientists have proven..." (Which scientists?)

**Bad - Overconfident**:
> "This is DEFINITELY the only way." (Usually things aren't certain)

**Good - Honest**:
> "I think Paris is the capital, but you might want to double-check."

---

### 3. RELATION - Is it on-topic?

**Bad - Off-topic**:
> Context: "Do you like pizza?"
> Response: "The weather is nice today!"

**Bad - Tangent**:
> Context: "Do you like pizza?"
> Response: "Speaking of food, did you know that the first restaurant opened in 1765 in Paris..."

**Good - On-topic**:
> Context: "Do you like pizza?"
> Response: "Yes! I especially like pepperoni pizza."

---

### 4. MANNER - Is it clear and easy to understand?

**Bad - Confusing**:
> "Well, regarding the aforementioned query, one might posit that the utilization of..."

**Bad - Disorganized**:
> "And also, but first, however, anyway, so basically..."

**Good - Clear**:
> "Yes, I think that's a great idea because..."

---

## Step 6: Quick Decision Flowchart

```
Start: Read Context + Both Responses
         │
         ▼
    Can you understand both responses?
         │
    No ──┼── Yes
         │      │
         ▼      ▼
  Pick the    Do they both answer the question?
  clearer one      │
                No ─┼── Yes
                    │     │
                    ▼     ▼
              Pick the   Are they both accurate?
              one that        │
              answers    No ──┼── Yes
                              │     │
                              ▼     ▼
                        Pick the   They're similar
                        accurate   → Press "3" (Equal)
                        one
```

---

## Step 7: Examples

### Example 1: Clear Winner

**Context**: "What's 2 + 2?"

**Response A**: "The answer is 4."

**Response B**: "Well, mathematics is a fascinating field..."

**Your choice**: Press `1` (A is MUCH better)
**Why**: A directly answers, B doesn't answer at all

---

### Example 2: Slight Difference

**Context**: "Do you like movies?"

**Response A**: "Yes, I enjoy watching movies, especially action films."

**Response B**: "Yes, I like movies."

**Your choice**: Press `2` (A is slightly better)
**Why**: Both answer, but A gives more helpful detail

---

### Example 3: Equal

**Context**: "What's your name?"

**Response A**: "I'm an AI assistant."

**Response B**: "I'm a helpful AI."

**Your choice**: Press `3` (Equal)
**Why**: Both are equally valid responses

---

### Example 4: B is Better

**Context**: "Can you explain quantum physics simply?"

**Response A**: "Quantum physics is the study of subatomic particles using Schrödinger equations and wave functions in Hilbert spaces..."

**Response B**: "Quantum physics is about how tiny particles behave in surprising ways. They can be in two places at once!"

**Your choice**: Press `4` or `5` (B is better)
**Why**: B is clearer and simpler, which is what was asked

---

## Step 8: Save and Export

### During Annotation
- Press `S` every 50 pairs to save to browser
- If you close the browser, your progress is saved!

### When Finished
1. Click **"⬇️ Download Annotated"**
2. This saves `tier1_annotated.json`
3. Upload this to your Kaggle dataset

---

## Tips for Speed

1. **Trust your gut** - First impression is usually right
2. **Don't overthink** - If you can't decide in 10 seconds, press `3` (Equal)
3. **Take breaks** - Do 100 pairs, then rest
4. **Consistency matters** - Use the same standards throughout

---

## Time Estimate

- 500 pairs
- ~30 seconds each
- = ~4 hours total
- Suggested: 100 pairs/day × 5 days

---

## Troubleshooting

**Can't load file?**
- Make sure it's a `.json` file
- Try a different browser (Chrome works best)

**Lost progress?**
- Check if browser localStorage has it (it auto-saves)
- Make sure to press `S` to save manually too

**Not sure how to decide?**
- Ask: "Which response would I prefer if I was having this conversation?"
- When in doubt, choose the shorter, clearer, more direct answer
