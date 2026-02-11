# Complete Annotation Guide for Helpers
## GriceBench Response Quality Annotation

---

## 📌 What Is This Project?

You're helping with **AI research**. We're training an AI to give **better, more helpful responses** in conversations.

**Your job**: Compare two AI responses and pick the better one. That's it!

Think of it like being a judge on a talent show - you see two contestants (responses) and pick which one did a better job answering the question.

---

## 📁 Files You'll Receive

You should receive these files from the project owner:

| File | What It Is |
|------|------------|
| `tier1_hard_pairs.json` | The 500 pairs you need to annotate |
| `annotation_interface.html` | The tool to do annotations (opens in browser) |
| `annotated.json` | Already-done examples for reference (optional) |

**Save all files to the same folder** on your computer (e.g., a folder called "Annotation" on your Desktop).

---

## 🚀 Step-by-Step: Getting Started

### Step 1: Open the Annotation Tool

1. Find `annotation_interface.html` on your computer
2. **Double-click it** - it will open in your web browser (Chrome works best)
3. You'll see a dark interface that says "GriceBench Fast Annotator"

### Step 2: Load Your Data

1. Click the big green **"📂 Load JSON"** button
2. Navigate to where you saved the files
3. Select `tier1_hard_pairs.json`
4. The first pair will appear on screen

### Step 3: Start Annotating!

Now you're ready to compare responses. See the next section for how to decide.

---

## 🎯 How to Make Decisions

### What You'll See on Screen

```
┌─────────────────────────────────────────────────────────────┐
│ 📝 CONTEXT (The Conversation So Far)                        │
│                                                             │
│ [agent_1]: Do you like movies?                              │
│ [agent_2]: Yes! I love action films!                        │
│ [agent_1]: What's your favorite?                            │
└─────────────────────────────────────────────────────────────┘

┌───────────────────────┐       ┌───────────────────────┐
│     RESPONSE A        │       │     RESPONSE B        │
│                       │       │                       │
│ "John Wick is my      │       │ "Movies are great.    │
│ favorite! It has      │       │ I like movies. Movies │
│ Keanu Reeves."        │       │ are very good."       │
└───────────────────────┘       └───────────────────────┘
```

### The Decision Process

**Step 1: Read the Context**
- What is the conversation about?
- What question was asked?

**Step 2: Read Both Responses**
- Which one actually answers the question?
- Which one is clearer?
- Which one would YOU prefer if you were having this conversation?

**Step 3: Make Your Choice**

| Press | Meaning | When to Use |
|-------|---------|-------------|
| **1** | A is MUCH better | A clearly answers, B is off-topic or confusing |
| **2** | A is slightly better | A is a bit more helpful or clear |
| **3** | Equal / Can't decide | Both are similar quality |
| **4** | B is slightly better | B is a bit more helpful or clear |
| **5** | B is MUCH better | B clearly answers, A is off-topic or confusing |

**Step 4: Move to Next Pair**
- Press **→** (right arrow) to go to the next pair
- Press **←** (left arrow) if you made a mistake and want to go back

---

## 📋 The 4 Things That Make a Response Good

### 1. 📏 QUANTITY - Right Amount of Information

**BAD: Too Little**
> Question: "How do I bake a cake?"
> Answer: "Use ingredients."

**BAD: Too Much (Off-Topic Rambling)**
> Question: "What's Paris known for?"
> Answer: "Paris is in France. France fought in World War 2. My grandfather was in WW2. He had a dog named Rex..."

**GOOD: Just Right**
> Question: "What's Paris known for?"
> Answer: "Paris is known for the Eiffel Tower, great food, and art museums like the Louvre."

---

### 2. ✅ QUALITY - Truthful and Honest

**BAD: Made-Up Statistics**
> "Studies show that 87% of people agree with this." (What studies? This sounds fake!)

**BAD: Overconfident**
> "This is DEFINITELY the only correct answer." (Usually nothing is 100% certain)

**BAD: Vague Claims**
> "Many experts believe..." (Which experts? Be specific!)

**GOOD: Honest**
> "I think the Eiffel Tower was built around 1889, but you might want to verify that."

---

### 3. 🎯 RELATION - Stays On Topic

**BAD: Completely Off-Topic**
> Question: "Do you like pizza?"
> Answer: "The weather is nice today!"

**BAD: Goes on a Tangent**
> Question: "Do you like pizza?"
> Answer: "Speaking of food, did you know the first restaurant opened in 1765? It was in Paris, which reminds me of the French Revolution..."

**GOOD: Answers the Question**
> Question: "Do you like pizza?"
> Answer: "Yes! I love pepperoni pizza the most."

---

### 4. 💬 MANNER - Clear and Easy to Understand

**BAD: Confusing Jargon**
> "One might posit that the utilization of the aforementioned methodology..."

**BAD: Repetitive**
> "I like movies. I really like movies. Movies are good. I like movies a lot."

**BAD: Disorganized**
> "And also, but first, however, anyway, so basically, what I mean is..."

**GOOD: Clear and Simple**
> "Yes, I think that's a great idea. Here's why..."

---

## ⭐ Quick Reference: Decision Flowchart

```
START: Read Context + Both Responses
            │
            ▼
   Can you understand both responses?
            │
    NO ─────┼───── YES
            │       │
            ▼       ▼
    Pick the    Do both actually answer the question?
    clearer one       │
                NO ───┼─── YES
                      │     │
                      ▼     ▼
                Pick the   Is one more helpful or accurate?
                one that         │
                answers    NO ───┼─── YES
                                 │     │
                                 ▼     ▼
                          Press 3   Pick the better one
                          (Equal)   (1-2 or 4-5)
```

---

## 📚 Real Examples

### Example 1: Clear Winner - A is Much Better (Press 1)

**Context**: "What's 2 + 2?"

**Response A**: "The answer is 4."

**Response B**: "Mathematics is a fascinating subject that dates back thousands of years..."

**Decision**: Press **1** (A is MUCH better)
**Why**: A directly answers. B rambles and never gives the answer.

---

### Example 2: Slight Difference (Press 2)

**Context**: "Do you like movies?"

**Response A**: "Yes, I love movies! Action films are my favorite."

**Response B**: "Yes, I like movies."

**Decision**: Press **2** (A is slightly better)
**Why**: Both answer the question, but A provides more helpful detail.

---

### Example 3: Both Equal (Press 3)

**Context**: "What's your name?"

**Response A**: "I'm an AI assistant."

**Response B**: "I'm a helpful AI."

**Decision**: Press **3** (Equal)
**Why**: Both are equally valid and helpful responses.

---

### Example 4: B is Better (Press 4 or 5)

**Context**: "Can you explain quantum physics simply?"

**Response A**: "Quantum physics uses Schrödinger equations and wave functions in Hilbert spaces..."

**Response B**: "Quantum physics is about how tiny particles behave in surprising ways - they can be in two places at once!"

**Decision**: Press **5** (B is MUCH better)
**Why**: The question asked for a SIMPLE explanation. B is simple, A is too technical.

---

### Example 5: Both Are Bad (Press 3)

**Context**: "What's the capital of Japan?"

**Response A**: "[ERROR] [ERROR] [ERROR]"

**Response B**: "{ json: null, status: failed }"

**Decision**: Press **3** (Equal)
**Why**: Both are gibberish/broken. Neither is useful.

---

## 💾 Saving Your Work

### While Annotating
- Press **S** every ~50 pairs to save your progress
- The tool also auto-saves to your browser
- If you close the browser and reopen, your progress should still be there!

### When You're Done
1. Click the **"⬇️ Download Annotated"** button
2. This saves a file called `tier1_annotated.json`
3. **Send this file back** to the project owner

---

## ⏱️ Time Estimate

| | |
|---|---|
| Total pairs | 500 |
| Time per pair | ~20-30 seconds |
| Total time | ~3-4 hours |
| **Recommended pace** | 100 pairs/day × 5 days |

Take breaks! Doing too many at once leads to fatigue and inconsistent decisions.

---

## 💡 Pro Tips

1. **Trust your gut** - Your first impression is usually right
2. **Don't overthink** - If you can't decide in 10 seconds, press 3 (Equal)
3. **Ask yourself**: "Which response would I prefer if I was the person asking?"
4. **When in doubt**: Pick the shorter, clearer, more direct answer
5. **It's subjective** - There's no "wrong" answer, we want YOUR honest opinion

---

## ❓ Troubleshooting

**Can't load the JSON file?**
- Make sure the file ends in `.json`
- Try using Chrome browser
- Make sure you downloaded the file correctly

**Lost your progress?**
- Check if the browser still has it (localStorage)
- Always press S to save periodically
- If truly lost, just continue from where you remember

**Both responses look identical?**
- Look carefully - sometimes there are subtle differences
- Check capitalization, punctuation, or extra words
- If truly identical, press 3 (Equal)

**Response is cut off or looks weird?**
- Just evaluate based on what you can see
- If both are equally broken, press 3 (Equal)

**Not sure what the conversation is about?**
- Focus on the last message in the context (that's usually the question)
- Pick the response that best addresses that last message

---

## 📞 Questions?

If you're unsure about anything, ask the project owner! It's better to ask than to guess.

**Remember**: Your annotations directly help improve AI! Thank you for your help! 🙏
