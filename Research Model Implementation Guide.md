# Gricean Violation Detection & Repair (GriceBench): The Complete Implementation Guide

This guide assumes you have never built a machine learning model before. Every single step is explained in detail, from setting up your computer to training your final model. Follow this guide sequentially without skipping any section.

---

# Part I: Foundations and Preparation

---

## Chapter 1: Understanding What You Are Building

Before touching any code or data, you need a crystal-clear mental model of what you are creating.

### 1.1 The Big Picture

You are building a system that makes AI assistants better at communicating like cooperative humans. The philosopher Paul Grice identified four principles (called "maxims") that humans follow when communicating cooperatively. Your system will detect when an AI response violates these principles and then fix those violations.

Think of it like a spell-checker, but instead of checking spelling, you are checking whether responses follow good communication principles.

**The four maxims you will work with:**

**Quantity:** Say enough to be helpful, but not so much that you overwhelm or bore the listener. A response violates Quantity by being too brief (leaving out important information) or too verbose (including unnecessary repetition or tangents).

**Quality:** Only say things that are true and supported by evidence. A response violates Quality by stating false information, making unsupported claims, or contradicting known facts.

**Relation:** Stay on topic. A response violates Relation by drifting to unrelated subjects or failing to address what the user actually asked.

**Manner:** Be clear and organized. A response violates Manner by being confusing, using unexplained jargon, having poor structure, or containing ambiguous references.

### 1.2 The Three Components You Will Build

**Component A: The Violation Detector (Critic)**

This is a classifier that takes a conversation context, any available evidence, and a response, then outputs probabilities for whether each maxim is violated. For example, it might output: Quantity violation probability 0.15, Quality violation probability 0.85, Relation violation probability 0.10, Manner violation probability 0.20. In this case, the model thinks Quality is likely violated.

**Component B: The Repair Model (Editor)**

This is a text-rewriting model that takes a response with identified violations and rewrites it to fix those specific violations while preserving the core meaning. If the detector says "Quality is violated because the response contradicts the evidence," the repair model rewrites the response to align with the evidence.

**Component C: The Generator Training Loop (Optional Advanced Component)**

This uses the detector as a feedback signal to train a dialogue model to avoid violations in the first place. Think of it as teaching a student by showing them their mistakes until they learn not to make them.

### 1.3 Why This Specific Approach

You might wonder why you cannot just ask a large language model to "be cooperative." The problem is that "cooperative" is vague. By breaking it into four specific, measurable maxims, you can actually train models on concrete objectives and measure improvement scientifically. This is what makes your work publishable rather than just an application.

---

## Chapter 2: Setting Up Your Computing Environment

This chapter walks you through preparing your computer for machine learning research. Every step matters; do not skip anything.

### 2.1 Hardware Requirements and Recommendations

**Minimum requirements:**

You need a computer with at least 16 gigabytes of RAM (random access memory). RAM is the short-term memory your computer uses when running programs. Machine learning models need substantial RAM to load and process data.

You need at least 100 gigabytes of free storage space. Your datasets, model checkpoints, and intermediate files will consume significant space.

A dedicated graphics processing unit (GPU) is strongly recommended but not absolutely required for the initial stages. GPUs are specialized processors that can perform many calculations simultaneously, which dramatically speeds up model training. Without a GPU, training that takes one hour might take twenty hours.

**If you do not have a GPU:**

You have three free or low-cost options.

Google Colab provides free GPU access through your web browser. You get approximately twelve hours of continuous GPU use before needing to restart. This is sufficient for most of your experiments if you save checkpoints frequently.

Kaggle Notebooks offer thirty hours of GPU time per week for free. The interface is similar to Google Colab.

Your university may provide GPU cluster access. Contact your institution's research computing or IT department to ask about available resources.

**Recommended setup for this project:**

Use Google Colab for initial experiments and model development. Once your code is working, transfer to a more stable environment (university cluster or Kaggle) for final training runs. This approach costs nothing and is how many published papers are developed.

### 2.2 Installing Python

Python is the programming language used for virtually all modern machine learning. You need Python version 3.8, 3.9, 3.10, or 3.11. Version 3.10 is recommended for maximum compatibility with current libraries.

**On Windows:**

Download Python from the official website (python.org). During installation, you must check the box that says "Add Python to PATH." This is critical; if you skip this, none of the subsequent commands will work. After installation, open Command Prompt (search for "cmd" in the Start menu) and type "python --version" to verify the installation worked.

**On Mac:**

Macs come with Python pre-installed, but it is usually an old version. Install Homebrew first (a package manager for Mac) by following instructions at brew.sh. Then use Homebrew to install Python by opening Terminal and running the appropriate command. Verify with "python3 --version" in Terminal.

**On Linux:**

Python is typically pre-installed. Verify your version with "python3 --version" in the terminal. If you need a newer version, use your distribution's package manager.

**If using Google Colab:**

Python is already installed and configured. You do not need to do anything for this step. Simply go to colab.research.google.com and create a new notebook.

### 2.3 Understanding Virtual Environments

A virtual environment is an isolated space where you install packages for a specific project without affecting other projects on your computer. This prevents conflicts between different projects that might need different versions of the same library.

**Why this matters:**

Imagine Project A needs version 1.0 of a library, but Project B needs version 2.0. Without virtual environments, installing version 2.0 for Project B would break Project A. Virtual environments keep each project's dependencies separate.

**Creating a virtual environment:**

On your local machine, navigate to where you want your project folder using the command line. Create a new folder for your project, then create a virtual environment inside it. You activate the environment before working on your project and deactivate it when done. When the environment is active, any packages you install go only into that environment.

**If using Google Colab:**

Each Colab notebook is already an isolated environment. You do not need to create virtual environments manually. However, packages you install in Colab do not persist between sessions, so you will need to reinstall them each time you open your notebook.

### 2.4 Installing Required Libraries

You need several Python libraries for this project. Each serves a specific purpose.

**Transformers:** This library from Hugging Face provides pre-trained language models and tools for fine-tuning them. It is the foundation of modern NLP research. The library gives you access to models like BERT, RoBERTa, DeBERTa, T5, and many others without needing to build them from scratch.

**Datasets:** Also from Hugging Face, this library provides easy access to thousands of datasets and tools for processing them efficiently. It handles loading, preprocessing, and batching data.

**PyTorch:** This is a deep learning framework that provides the low-level operations for neural networks. Transformers is built on top of PyTorch. You rarely interact with PyTorch directly, but it must be installed.

**Scikit-learn:** This library provides traditional machine learning tools and evaluation metrics. You will use it for calculating F1 scores, confusion matrices, and other metrics.

**Pandas:** This library is for data manipulation. It provides DataFrames, which are like spreadsheets in Python, making it easy to organize and analyze your data.

**NumPy:** This library provides efficient numerical operations. Many other libraries depend on it.

**NLTK and SpaCy:** These are natural language processing toolkits with tools for tokenization (splitting text into words), part-of-speech tagging, named entity recognition, and other text processing tasks.

**Textstat:** This library calculates readability metrics, which you will use for detecting Manner violations.

**Sentence-Transformers:** This library provides models for creating sentence embeddings (numerical representations of sentences), which you will use for measuring semantic similarity in Relation violation detection.

**Installation process:**

On your local machine with your virtual environment activated, you use pip (Python's package installer) to install each library. On Google Colab, you add installation commands at the top of your notebook, and they run each time you open the notebook.

---

## Chapter 3: Obtaining and Understanding Your Data

Your dataset is the foundation of your entire project. This chapter explains exactly how to get the data and understand its structure.

### 3.1 Primary Dataset: Topical-Chat

Topical-Chat is a dataset of human-human conversations where participants discuss various topics using provided "reading sets" (background knowledge). This is perfect for your project because the reading sets serve as evidence, allowing you to measure Quality violations objectively.

**Dataset statistics:**

Topical-Chat contains approximately 11,000 conversations with over 235,000 individual turns. Each conversation has associated reading sets containing facts that participants can use in their responses. Topics span entertainment, sports, science, politics, and other domains.

**How to access it:**

The dataset is hosted on GitHub by Amazon Alexa. You need to download it from the official repository. The repository contains JSON files organized by conversation splits (train, valid_frequent, valid_rare, test_frequent, test_rare). The "frequent" splits contain conversations on topics that appear often in the data, while "rare" splits contain less common topics.

**Understanding the data structure:**

Each conversation in the JSON files has a unique identifier and contains multiple turns. Each turn has a speaker identifier (agent_1 or agent_2), the message content, and a reference to which knowledge snippet (if any) was used. The reading sets are in separate JSON files, organized by topic and containing individual fact snippets.

**What you will extract:**

For each training example, you need to create a tuple of (context, evidence, response). The context is the conversation history up to and including the user's question. The evidence is the knowledge snippet(s) available for that turn. The response is what the assistant said.

### 3.2 Secondary Dataset: FaithDial

FaithDial is specifically designed for studying hallucination in knowledge-grounded dialogue. It was created by taking conversations from Wizard-of-Wikipedia and editing responses to fix hallucinations, creating paired examples of hallucinated versus faithful responses.

**Why this helps:**

FaithDial provides natural examples of Quality violations (the original hallucinated responses) and their corrections (the edited faithful responses). This is valuable for training your repair model because the edits were made by humans specifically to fix faithfulness issues.

**How to access it:**

FaithDial is available through the Hugging Face Datasets library. You can load it directly without manual downloading by using the datasets library's load_dataset function.

**What you will use it for:**

Use FaithDial primarily for Quality violation examples and repair training pairs. The original responses are Quality violations; the edited responses are the repairs.

### 3.3 Understanding Data Formats

**JSON format:**

JSON (JavaScript Object Notation) is a text-based format for storing structured data. It uses curly braces for objects and square brackets for lists. Understanding JSON is essential because most NLP datasets use this format. When you open a JSON file, you see nested structures of keys and values. Python can read JSON files directly into dictionaries and lists.

**Data loading in Python:**

You will use Python's json library to read JSON files from Topical-Chat and the Hugging Face datasets library to load FaithDial. The datasets library handles downloading, caching, and efficient loading automatically.

### 3.4 Data Exploration Before Processing

Before building anything, spend time exploring your data manually. This step is crucial and often skipped by beginners.

**Questions to answer through exploration:**

How long are typical responses in Topical-Chat? This informs what "too long" and "too short" mean for Quantity violations.

How often do responses directly quote or paraphrase the knowledge snippets? This shows what grounded responses look like for Quality.

What is the vocabulary like? Are there domain-specific terms? This affects how you think about Manner violations.

How many turns are typically in a conversation? This affects how much context to include.

**How to explore:**

Load a sample of conversations (perhaps 50-100) and read through them manually. Take notes on patterns you observe. Calculate basic statistics like average response length, vocabulary size, and topic distribution. This qualitative understanding will make all subsequent decisions more informed.

---

# Part II: Building the Dataset (GriceBench)

---

## Chapter 4: Designing Your Violation Injection Pipeline

This is where your original contribution begins. You will create a systematic method for generating examples of each violation type from clean, cooperative responses.

### 4.1 The Philosophy of Controlled Violation Injection

The fundamental insight is this: creating violations synthetically allows you to know the ground truth labels with certainty. If you take a good response and deliberately add a contradicting fact, you know with 100% certainty that it now violates Quality. This is far more efficient than paying annotators to find and label naturally occurring violations.

**The key constraint:**

Each transformation must change only one maxim at a time (for single-violation examples). If your Quantity transformation accidentally also makes the response off-topic, your labels are wrong and your detector will learn confused patterns.

### 4.2 Quantity Violation Transformations

**Under-informative (too little) transformation:**

Starting from a good response, you want to create a version that says too little to be helpful while still being superficially on-topic.

Identify the key informational content of the response. This includes specific facts, numbers, names, and explanations. Replace specific content with vague acknowledgments like "That's interesting" or "It depends on various factors." Keep topic words so the response appears related but lacks substance.

**Example:**

Original: "The Great Wall of China is approximately 13,171 miles long and was built over many centuries, starting in the 7th century BC. It was constructed primarily to protect against invasions from nomadic groups."

Under-informative: "The Great Wall of China is quite long and has an interesting history of construction."

The transformed version is on-topic (Relation satisfied), not false (Quality satisfied), and clear (Manner satisfied), but lacks the specific information that would make it helpful (Quantity violated).

**Over-informative (too much) transformation:**

Add content that is true and related but unnecessary for answering the question. The key is redundancy or excessive detail beyond what was asked.

Techniques include adding paraphrases of the same fact using different words, including tangentially related facts from the evidence that were not asked about, and repeating information in multiple ways.

**Example:**

User question: "How long is the Great Wall?"

Original: "The Great Wall of China is approximately 13,171 miles long."

Over-informative: "The Great Wall of China is approximately 13,171 miles long, which is to say about 21,196 kilometers in metric units. That's an incredibly long distance—roughly 13,171 miles of wall stretching across northern China. To put that in perspective, 13,171 miles is longer than the distance from New York to Beijing and back. The wall, measuring some 13,171 miles, represents one of the longest structures ever built."

The transformed version answers the question (Relation satisfied), is true (Quality satisfied), and is clear (Manner mostly satisfied), but says far more than needed through redundancy (Quantity violated).

**Anti-shortcut measures:**

You must also create counterexamples to prevent the detector from learning spurious correlations.

Long-but-good examples: Find responses that are naturally long because the question is complex and requires detailed explanation. Label these as NOT violating Quantity.

Short-but-sufficient examples: Find responses to simple factual questions where a brief answer is appropriate. Label these as NOT violating Quantity.

### 4.3 Quality Violation Transformations

Quality violations occur when the response says something unsupported by or contradicting the available evidence.

**Unsupported claim injection:**

Add a plausible-sounding fact that is not in the evidence. The fact should be related to the topic (so Relation is preserved) and stated clearly (so Manner is preserved), but not actually supported by the knowledge provided.

**Example:**

Evidence: "Mount Everest is 29,032 feet tall and was first summited in 1953 by Edmund Hillary and Tenzing Norgay."

Original response (using evidence): "Mount Everest stands at 29,032 feet and was first climbed in 1953."

Quality violation (unsupported): "Mount Everest stands at 29,032 feet, was first climbed in 1953, and experiences approximately 150 climbing attempts per year."

The added claim about 150 attempts might be true in reality, but it is not in the provided evidence, so the response violates Quality in a knowledge-grounded setting.

**Contradiction injection:**

Alter a fact to directly contradict the evidence.

**Example:**

Evidence: "The Eiffel Tower was completed in 1889."

Original: "The Eiffel Tower was completed in 1889 for the World's Fair."

Quality violation (contradiction): "The Eiffel Tower was completed in 1901 for the World's Fair."

The date has been changed to contradict the evidence directly.

**Subtlety considerations:**

Make some contradictions subtle (changing 1889 to 1887) and some obvious (changing 1889 to 1950). This ensures your detector learns to catch both easy and hard cases.

### 4.4 Relation Violation Transformations

Relation violations occur when the response does not address what the user asked about.

**Hard negative substitution:**

Replace the response with a fluent, cooperative-sounding response from a different conversation about a different topic.

**Example:**

User question: "What's the population of Tokyo?"

Original response: "Tokyo has a population of approximately 14 million people in the city proper, making it one of the most populous cities in the world."

Relation violation: "The Mediterranean diet emphasizes olive oil, fish, and vegetables, and has been associated with numerous health benefits."

The substituted response is well-formed, true, and clear—it just has nothing to do with the question.

**Same-domain drift:**

A more subtle version keeps the general domain but answers a different question.

User question: "What's the population of Tokyo?"

Relation violation (subtle): "Tokyo is the capital of Japan and has been an important political center since the Meiji Restoration in 1868."

This response is about Tokyo, so it seems related at first glance, but it does not answer the population question.

**Length matching:**

When substituting responses, match the length approximately. Otherwise, the detector might learn "different length = Relation violation," which is not the concept you want it to learn.

### 4.5 Manner Violation Transformations

Manner violations make the response hard to understand without changing its truth or relevance.

**Ambiguous reference injection:**

Replace specific nouns with pronouns that lack clear antecedents.

**Example:**

Original: "Marie Curie won two Nobel Prizes. The first Nobel Prize was for Physics in 1903, and the second Nobel Prize was for Chemistry in 1911."

Manner violation: "She won two of them. The first one was for it in 1903, and the second one was for that in 1911."

**Sentence shuffling:**

Randomize the order of sentences to break logical flow.

**Example:**

Original: "First, preheat your oven to 350 degrees. Then, mix the flour and sugar. Finally, bake for 30 minutes."

Manner violation: "Bake for 30 minutes. First, preheat your oven to 350 degrees. Then, mix the flour and sugar."

**Jargon injection:**

Replace common words with technical equivalents without explanation, especially when the context suggests a general audience.

**Example:**

User: "I'm new to cooking. How do I make bread?" 

Original: "Mix flour, water, yeast, and salt. Let the dough rise for an hour, then bake at 400 degrees."

Manner violation: "Combine triticum aestivum powder, H2O, Saccharomyces cerevisiae, and NaCl. Allow fermentation-induced CO2 production to create alveolar structures for 60 minutes, then apply thermal energy at 477 Kelvin."

**Run-on creation:**

Join multiple sentences into one without proper punctuation.

**Example:**

Original: "The experiment worked. We found significant results. The hypothesis was confirmed."

Manner violation: "The experiment worked we found significant results the hypothesis was confirmed."

### 4.6 Multi-Maxim Violation Generation

After your single-maxim generators are working reliably, create examples that violate multiple maxims simultaneously.

**Quantity + Manner (verbose and unclear):**

Take an over-informative response and apply sentence shuffling plus ambiguous reference injection.

**Quality + Quantity (hallucinated and too long):**

Take a response, add multiple unsupported claims, and also add redundant paraphrasing.

**Relation + Quantity (off-topic and brief):**

Substitute an off-topic response that is also very short.

**Why this matters:**

In real model outputs, violations often co-occur. A model that only sees single violations will not learn to recognize the complex patterns in real data.

### 4.7 Implementing Validation Checks

Every generated violation must pass validation to ensure it actually represents the intended violation and does not accidentally violate other maxims.

**For Quantity violations:**

Run your Relation heuristic on the generated response. If it shows significant topic drift, reject the example. Run your Quality heuristic. If it shows unsupported claims, reject the example.

**For Quality violations:**

Run your Quantity heuristic. If the response became unusually long or short due to your injection, reject the example. Run your Relation heuristic. If topic drift occurred, reject the example.

**For Relation violations:**

Run your Quantity heuristic on both the original and substituted response. If their lengths are very different, reject the example. Run your Manner heuristic. If clarity changed substantially, reject the example.

**For Manner violations:**

Run your Quality and Relation heuristics. The truth and topic should be preserved; only clarity should change.

---

## Chapter 5: Implementing Weak Supervision Heuristics

Before training neural models, you will create rule-based heuristics that can label examples automatically. These labels are "weak" because they are noisy—not perfectly accurate—but they allow you to generate large amounts of training data without manual annotation.

### 5.1 Quantity Heuristics

**Length ratio method:**

Calculate the expected response length based on the question complexity. A simple yes/no question expects a short response; a complex "explain how" question expects a longer response.

Measure question complexity by counting question words (who, what, when, where, why, how), the number of entities mentioned, and the presence of multi-part questions (questions with "and" or multiple question marks).

Compare the actual response length to the expected length. If the actual length is less than 30% of expected, flag as "too little." If more than 300% of expected, flag as "too much."

**Redundancy detection:**

Use self-BLEU to measure how much a response repeats itself. BLEU is normally used to compare a candidate text to reference texts; self-BLEU compares different parts of the same text to each other.

Split the response into sentences. Calculate how similar each sentence is to every other sentence. High self-similarity indicates redundancy.

**Information density:**

Count unique named entities and specific facts per sentence. A response with many sentences but few unique entities is likely padding.

### 5.2 Quality Heuristics

**Natural Language Inference (NLI) for contradiction detection:**

Use a pre-trained NLI model to check if the response contradicts the evidence. NLI models take two texts (premise and hypothesis) and classify their relationship as entailment (hypothesis follows from premise), neutral (no clear relationship), or contradiction (hypothesis conflicts with premise).

For Quality checking, use the evidence as the premise and each sentence of the response as a hypothesis. If any sentence has high contradiction probability, flag the response for Quality violation.

**Retrieval-based fact verification:**

For each factual claim in the response, try to find support in the evidence. If a claim cannot be matched to any evidence sentence (low semantic similarity to all evidence), it may be unsupported.

Use sentence embeddings to compute similarity between response sentences and evidence sentences. Claims with similarity below a threshold to all evidence sentences are potentially unsupported.

**Entity consistency checking:**

Extract named entities from both the evidence and the response. If the response mentions entities not present in the evidence (for factual claims, not opinions), this suggests potential hallucination.

### 5.3 Relation Heuristics

**Semantic similarity:**

Compute the embedding of the user question and the embedding of the response. Use cosine similarity to measure how related they are. Responses with very low similarity to the question are potentially off-topic.

**Keyword overlap:**

Extract keywords from the question and check how many appear in the response. Complete absence of question keywords (aside from function words) suggests Relation violation.

**Topic model comparison:**

If you have many conversations, train a simple topic model (like LDA) on the corpus. Compare the topic distribution of the question to the topic distribution of the response. Large divergence suggests topic drift.

### 5.4 Manner Heuristics

**Readability scores:**

Use the Flesch-Kincaid readability formula, which estimates the grade level needed to understand a text based on sentence length and syllable counts. If a response to a general-audience question has a very high reading level (say, grade 16+), flag potential Manner violation.

**Discourse coherence:**

Track entity mentions across sentences. Coherent text tends to mention entities in the first sentence and then refer back to them. Many pronouns without antecedents established in recent sentences indicates poor coherence.

**Syntactic complexity:**

Parse sentences and measure parse tree depth. Very deeply nested sentences are harder to understand. Average parse tree depth above a threshold (determined empirically from your data) suggests potential Manner violation.

**Ambiguous reference counting:**

Count pronouns (it, they, this, that, these, those) and check if each has a clear antecedent within the previous two sentences. High ambiguous pronoun ratio indicates potential Manner violation.

### 5.5 Combining Heuristics

For each maxim, you have multiple heuristics. Combine them using voting or averaging.

**Voting approach:**

If 2 out of 3 heuristics flag a violation, label it as violated. This reduces noise from any single heuristic.

**Score averaging:**

Each heuristic outputs a score between 0 and 1. Average the scores. If the average exceeds a threshold (tuned on a small validation set), label as violated.

**Confidence-based filtering:**

Only keep examples where heuristics agree strongly. If all heuristics say the same thing with high confidence, the label is probably correct. Discard ambiguous examples where heuristics disagree.

---

## Chapter 6: Creating the Gold Standard Annotation Set

Weak supervision gets you far, but you need a smaller, carefully annotated dataset to evaluate your models accurately and to fine-tune after weak supervision pre-training.

### 6.1 Designing the Annotation Rubric

Your rubric is a document that explains exactly how annotators should label each maxim. The rubric must be detailed enough that two different people would make the same judgment on the same example.

**Quantity rubric:**

Score 0 (No violation): The response provides an appropriate amount of information for the question. It is neither noticeably incomplete nor redundant.

Score 1 (Too little): The response is missing key information that a cooperative speaker would include. The user would likely need to ask follow-up questions to get a complete answer.

Score 2 (Too much): The response includes unnecessary repetition, excessive tangents, or information beyond what the question asked for, to the point that it becomes burdensome.

Edge cases: If the question is genuinely ambiguous about how much detail is wanted, annotators should default to "no violation" unless the response is extreme.

**Quality rubric:**

Score 0 (No violation): All factual claims in the response are supported by the provided evidence, or are common knowledge not requiring evidence.

Score 1 (Violation): The response contains at least one factual claim that is either unsupported by the provided evidence or directly contradicts it.

Edge cases: Opinions and subjective statements are not Quality violations (they are not factual claims). Paraphrasing evidence in different words is not a violation as long as meaning is preserved.

**Relation rubric:**

Score 0 (No violation): The response directly addresses the user's question or statement. Even if it includes additional related information, the core query is answered.

Score 1 (Violation): The response fails to address the user's actual question. It may be on a related topic but does not answer what was asked.

Edge cases: If the user's question is unclear, a response that asks for clarification is not a Relation violation.

**Manner rubric:**

Score 0 (No violation): The response is clear, well-organized, and appropriate for the apparent audience.

Score 1 (Violation): The response is difficult to understand due to poor organization, ambiguous references, unexplained jargon, or confusing sentence structure.

Edge cases: Complex topics may require complex language. Judge clarity relative to the topic's inherent complexity. Technical jargon is acceptable if the user appears to be a domain expert based on context.

### 6.2 Setting Up Annotation Infrastructure

**Free annotation tool: Label Studio**

Label Studio is a free, open-source annotation tool that you can run on your own computer. It provides a web interface where annotators can view examples and enter labels. It tracks who labeled what and allows you to export results easily.

To set it up, install Label Studio using pip. Start the server, which opens a web interface. Create a project for your annotation task. Define your labeling interface (four questions, one per maxim, with the score options from your rubric). Import your examples. Share the link with your annotators.

**Alternative: Google Sheets**

If you cannot set up Label Studio, a simple Google Sheet works. Create columns for the example ID, context, evidence, response, and then one column per maxim for the annotator's score. Share the sheet with annotators and have them fill in their judgments. This is less elegant but functional.

### 6.3 Selecting Examples for Annotation

Do not annotate random examples. Strategically select examples to maximize the value of your annotation budget.

**Include all violation types:**

Ensure you have examples for each single-maxim violation and for clean examples (no violations). If you only have 1000 examples to annotate, allocate roughly 200 per maxim for single violations and 200 for clean controls.

**Include synthetic and natural examples:**

Annotate some examples you generated through violation injection (to verify your injection worked) and some natural examples from the original dataset (to find naturally occurring violations).

**Include edge cases:**

From your exploration of the data, identify ambiguous examples where you are unsure of the correct label. Annotating these helps calibrate your rubric and reveals where reasonable people disagree.

### 6.4 Running the Annotation Process

**Pilot phase:**

Before full annotation, do a pilot with 50 examples. Have all annotators label the same 50 examples. Calculate inter-annotator agreement (see next section). Discuss disagreements to clarify the rubric. Update the rubric based on pilot findings.

**Full annotation:**

After the rubric is stabilized, proceed with full annotation. For cost efficiency, have each example labeled by two annotators. For examples where they disagree, have a third annotator (or you) adjudicate.

**Quality control:**

Include some "check" examples where the correct answer is obvious. If an annotator frequently gets these wrong, their other labels may be unreliable.

### 6.5 Calculating Inter-Annotator Agreement

Inter-annotator agreement measures how consistently different annotators label the same examples. High agreement suggests your task is well-defined and labels are reliable. Low agreement suggests the task is ambiguous or the rubric needs improvement.

**Cohen's Kappa:**

For two annotators, Cohen's Kappa measures agreement beyond chance. A kappa of 0 means agreement is what you would expect by random guessing. A kappa of 1 means perfect agreement. Generally, kappa above 0.6 is considered acceptable for NLP tasks; above 0.8 is excellent.

**Fleiss' Kappa:**

If you have more than two annotators labeling the same examples, use Fleiss' Kappa, which generalizes Cohen's Kappa to multiple annotators.

**What to do with low agreement:**

If agreement is low (below 0.4), do not proceed with training. Instead, examine the disagreements to understand why annotators differ. Revise your rubric to address ambiguities. Re-train annotators on the revised rubric. Re-annotate the problematic examples.

---

# Part III: Training the Violation Detector

---

## Chapter 7: Preparing Data for Model Training

Before you can train a neural model, your data must be in the right format.

### 7.1 Creating Input-Output Pairs

Each training example for your detector will have:

Input: A concatenation of context, evidence, and response with special separator tokens.

Output: Four labels (one per maxim), either binary (0/1) or severity-based.

**Input formatting:**

Use a consistent format with special tokens to delineate different parts. For example:

[CONTEXT] User: What is photosynthesis? [EVIDENCE] Photosynthesis is the process by which plants convert sunlight into energy. [RESPONSE] Photosynthesis is how plants make food from sunlight.

The special tokens [CONTEXT], [EVIDENCE], and [RESPONSE] help the model understand the structure of the input.

**Output formatting:**

For multi-label classification, output is a vector of four numbers. For binary labels: [0, 0, 0, 1] might mean only Manner is violated. For severity labels, you would have more classes per maxim.

### 7.2 Tokenization

Neural language models do not understand text directly. They work with "tokens," which are numeric representations of text pieces. Tokenization is the process of converting text into tokens.

**Subword tokenization:**

Modern models use subword tokenization, which breaks words into smaller units. For example, "unhappiness" might become ["un", "happiness"] or ["un", "hap", "piness"]. This allows models to handle words they have never seen before by combining familiar pieces.

**Using the right tokenizer:**

Each pre-trained model comes with a specific tokenizer that was used during its training. You must use the same tokenizer. If you are using DeBERTa, use the DeBERTa tokenizer. If you are using RoBERTa, use the RoBERTa tokenizer. The Transformers library makes this easy—you load the tokenizer with the same model name.

**Handling long inputs:**

Models have a maximum input length (often 512 tokens). If your concatenated input exceeds this, you must truncate. A smart truncation strategy keeps the most important parts. For your task, the response is most critical (it is what you are classifying), so truncate the context first if needed, keeping the most recent turns.

### 7.3 Creating Train/Validation/Test Splits

Never evaluate your model on data it was trained on. You need separate sets.

**Training set (80%):** Used to update model weights. The model learns patterns from this data.

**Validation set (10%):** Used during training to check if the model is improving. You do not train on this directly, but you use it to decide when to stop training and to select hyperparameters.

**Test set (10%):** Used only once at the end to report final performance. Never look at test set results until you are done with all model development.

**Ensuring no leakage:**

If you have multiple examples from the same conversation, keep them all in the same split. Otherwise, the model might memorize conversation-specific patterns and appear to perform better than it actually would on new conversations.

### 7.4 Handling Class Imbalance

Your dataset might have more examples of some violation types than others. This imbalance can cause the model to be biased toward predicting the majority class.

**Strategies for handling imbalance:**

Oversampling: Duplicate examples from underrepresented classes so all classes have similar counts.

Undersampling: Remove examples from overrepresented classes. This discards data, so use it cautiously.

Class weights: During training, weight the loss function to penalize mistakes on underrepresented classes more heavily.

For your project, class weighting is usually the best approach because it does not discard data or create exact duplicates.

---

## Chapter 8: Training the Detector Model

Now you will actually train the neural network classifier.

### 8.1 Understanding Fine-Tuning

You will not train a model from scratch. Instead, you will "fine-tune" a pre-trained model. Pre-trained models like DeBERTa have already learned general language understanding from massive text corpora. Fine-tuning adapts this general knowledge to your specific task.

**Why fine-tuning works:**

Pre-training teaches the model about language structure, grammar, word meanings, and common patterns. Fine-tuning teaches it to apply this knowledge to your specific classification task. This is far more efficient than starting from random weights.

**What happens during fine-tuning:**

You add a classification head (new layers) on top of the pre-trained model. During training, both the new layers and the pre-trained layers are updated, but the pre-trained layers change only slightly (they are mostly preserving their knowledge).

### 8.2 Selecting a Pre-Trained Model

For multi-label text classification, encoder models are most suitable.

**DeBERTa (recommended):**

DeBERTa is currently one of the best-performing encoder models on many NLP benchmarks. It improves on BERT through disentangled attention mechanisms and enhanced mask decoder. The "base" size (140 million parameters) offers a good balance between performance and computational cost.

**RoBERTa:**

RoBERTa is a robustly optimized version of BERT. It is very well-understood and has strong performance. If you encounter issues with DeBERTa, RoBERTa is a reliable fallback.

**BERT:**

BERT is the original pre-trained encoder model. It is still effective but generally outperformed by DeBERTa and RoBERTa. Use it if you need maximum documentation and community support.

### 8.3 Setting Up the Classification Architecture

Your model will have two parts: the pre-trained encoder and a classification head.

**The encoder:**

Takes your tokenized input and produces "hidden states"—numeric representations of each token. The final hidden state of the [CLS] token (or pooled output) represents the entire input.

**The classification head:**

Takes the pooled representation and produces output logits for each class. For multi-label classification with four maxims, you typically have four separate output neurons, each with a sigmoid activation.

**Loss function:**

Use binary cross-entropy loss, calculated separately for each maxim and summed. This treats each maxim as an independent binary classification.

### 8.4 Understanding Hyperparameters

Hyperparameters are settings you choose before training that affect how training proceeds.

**Learning rate:**

Controls how much model weights change in response to each batch of data. Too high and the model learns nothing useful (overshoots good solutions). Too low and training takes forever or gets stuck. For fine-tuning pre-trained models, learning rates between 1e-5 and 5e-5 are typical.

**Batch size:**

The number of examples processed together before updating weights. Larger batches provide more stable gradient estimates but require more memory. For fine-tuning on a single GPU, batch sizes of 8 to 32 are common. If you run out of memory, reduce batch size.

**Number of epochs:**

One epoch means the model has seen every training example once. Multiple epochs allow the model to learn more thoroughly, but too many epochs cause overfitting (the model memorizes training data instead of learning generalizable patterns). Use validation performance to decide when to stop.

**Warmup steps:**

At the start of training, learning rate increases gradually from zero to the target value. This prevents early training instability. Typically, warmup covers 5-10% of total training steps.

### 8.5 The Training Loop

Training proceeds in iterations:

1. Load a batch of training examples.
2. Pass them through the model to get predictions.
3. Compare predictions to true labels using the loss function.
4. Calculate gradients (how to adjust weights to reduce loss).
5. Update weights using the optimizer.
6. Repeat until all batches in the epoch are processed.
7. After each epoch, evaluate on validation set.
8. If validation performance improved, save the model checkpoint.
9. If validation performance has not improved for several epochs, stop training (early stopping).

**Monitoring training:**

Track training loss and validation loss over time. Training loss should steadily decrease. Validation loss should decrease initially; if it starts increasing while training loss continues decreasing, you are overfitting.

Also track validation F1 score per maxim. This is your real performance metric.

### 8.6 Two-Phase Training Strategy

Your training will have two phases:

**Phase 1: Weak supervision pre-training**

Train on your large (50k+) weakly-labeled dataset. This gets the model familiar with the task. Train for 2-3 epochs. The model will learn imperfect but useful patterns.

**Phase 2: Gold fine-tuning**

Take the model from Phase 1 and continue training on your smaller (1k) gold-labeled dataset. Use a lower learning rate (perhaps 1e-5 instead of 2e-5) because you are refining rather than learning from scratch. Train for 5-10 epochs with early stopping based on validation performance.

**Why two phases:**

Weak labels provide volume; gold labels provide quality. Starting with weak supervision teaches the model approximate patterns. Fine-tuning on gold data corrects errors and sharpens decision boundaries.

### 8.7 Calibrating Your Detector

A calibrated model's probability outputs reflect true likelihoods. If the model says "80% probability of Quality violation," it should be correct about 80% of the time when it makes such predictions.

**Why calibration matters:**

You will use detector probabilities as rewards for training the generator. Miscalibrated probabilities lead to poor rewards and unstable training.

**Temperature scaling:**

A simple calibration method. After training, find a single "temperature" parameter that, when applied to logits before the sigmoid, minimizes calibration error on a held-out set. The calibrated probability is sigmoid(logit / temperature).

**Expected Calibration Error (ECE):**

A metric for calibration quality. Bin predictions by confidence level and compare average confidence to actual accuracy in each bin. Lower ECE means better calibration.

---

## Chapter 9: Evaluating the Detector

Before building the repair model, rigorously evaluate your detector.

### 9.1 Primary Metrics

**Per-maxim F1 score:**

For each maxim, calculate F1 score separately. F1 is the harmonic mean of precision (of predicted violations, how many were actually violations) and recall (of actual violations, how many did you predict). F1 balances both concerns.

Target: Aim for F1 above 0.7 for each maxim. Above 0.8 is strong.

**Macro-averaged F1:**

Average F1 across all four maxims. This single number summarizes overall detection performance.

**Exact match:**

For each example, did the model get all four labels correct? This is a strict metric. A model might have high per-maxim F1 but low exact match if it often gets one or two maxims right but rarely gets all four.

### 9.2 Error Analysis

Numbers alone do not tell you how to improve. You must analyze errors qualitatively.

**Confusion analysis:**

For each maxim, examine false positives (predicted violation but actually clean) and false negatives (predicted clean but actually violation). What patterns do you see?

**Maxim confusion:**

Does your model confuse Quantity with Manner? If it predicts Quantity violation when the true label is Manner, the model has not learned to distinguish verbosity from unclearness.

Create a confusion matrix between predicted and actual maxim combinations to see these patterns.

**Difficult examples:**

Identify examples where the model is confidently wrong. These are the most important to understand. Are they truly ambiguous, or is there a pattern the model missed?

### 9.3 Ablation Studies

Ablations systematically remove components to measure their contribution.

**No evidence ablation:**

Train and evaluate a version of your detector that does not see the evidence (only context and response). If Quality detection drops significantly, this confirms that evidence is important for Quality detection—a meaningful finding.

**Single maxim ablation:**

Train separate single-task models for each maxim and compare to your multi-task model. If the multi-task model outperforms single-task, this suggests the maxims share useful information.

---

# Part IV: Building the Repair Model

---

## Chapter 10: Designing the Repair Model

The repair model takes a response with identified violations and rewrites it to fix those violations.

### 10.1 Task Formulation

**Input:** Context, evidence, original response, violation labels

**Output:** Repaired response

The model learns to transform responses from cooperative to violating (actually the reverse—from violating to cooperative—using your injection pairs reversed).

### 10.2 Training Data Creation

You already have the data you need from the injection pipeline, just used in reverse.

For each (context, evidence, good response, violated response, violation type) tuple, create a training example where the input is the violated response with its violation label, and the target is the good response.

**Example:**

Violated response: "Photosynthesis is how plants make food from sunlight. Additionally, photosynthesis converts light into chemical energy. Plants use photosynthesis to convert sunlight into food."

Violation label: [Quantity: too_much]

Target (repaired): "Photosynthesis is how plants make food from sunlight."

### 10.3 Model Selection

**T5 (recommended for beginners):**

T5 treats all NLP tasks as text-to-text problems. It is very flexible and well-documented. The "base" size (220M parameters) is manageable on free GPU resources.

**BART:**

Similar to T5 but uses a different pre-training objective. Also a strong choice for sequence-to-sequence tasks.

### 10.4 Control Token Strategy

Make the repair conditional on the violation type by prepending special tokens to the input.

**Input format:**

[REPAIR] [VIOLATION=QUANTITY_OVER] [CONTEXT] User asked about photosynthesis [EVIDENCE] Photosynthesis converts sunlight to energy [RESPONSE] [Original violated response]

The [VIOLATION=...] token tells the model what to fix. During inference, you provide the violation types detected by your detector.

**Why control tokens work:**

During training, the model learns associations between violation tokens and the types of edits needed. When you provide a violation token at inference time, the model activates the corresponding editing behavior.

### 10.5 Handling Multiple Violations

When a response violates multiple maxims, include multiple violation tokens:

[REPAIR] [VIOLATION=QUANTITY_OVER] [VIOLATION=MANNER_UNCLEAR] [CONTEXT] ... [RESPONSE] ...

Your training data should include multi-violation examples (from your multi-maxim injection) so the model learns to handle these cases.

---

## Chapter 11: Training the Repair Model

### 11.1 Sequence-to-Sequence Training

For T5 and similar models, training is straightforward. You provide input-output pairs and the model learns to maximize the probability of generating the target output given the input.

**Loss function:**

Cross-entropy loss over the output tokens. The model learns to predict each token of the repaired response given the input and all previous tokens of the output.

### 11.2 Hyperparameters for Repair

**Learning rate:** 1e-4 to 3e-4 is typical for T5 fine-tuning.

**Batch size:** 4 to 16, depending on available memory. Sequence-to-sequence models with long inputs require significant memory.

**Maximum input length:** 512 tokens usually suffices for dialogue.

**Maximum output length:** Set to match typical response lengths in your data (perhaps 128-256 tokens).

**Epochs:** 3-5 epochs are often sufficient. Use early stopping based on validation performance.

### 11.3 Enforcing Constraints

The repair model must change as little as necessary. There are two concerns:

**Meaning preservation for non-Quality repairs:**

When fixing Quantity, Relation, or Manner, the core meaning should stay the same. If the original response said "Paris is the capital of France" and had a Manner issue, the repair should still say Paris is the capital of France, just more clearly.

**Evidence alignment for Quality repairs:**

When fixing Quality, the repair should become consistent with the evidence, even if this changes the meaning. If the original response said "Paris is the capital of Spain" (contradicting evidence), the repair should change this.

**Training signals for constraints:**

Include NLI-based filtering in your training data. Remove examples where the repair changes meaning inappropriately (for non-Quality fixes). This teaches the model to be conservative.

---

## Chapter 12: Evaluating the Repair Model

### 12.1 Automatic Metrics

**Targeted fix rate:**

Run your detector on the repaired responses. For what percentage did the targeted violation disappear? This is your primary success metric.

**No-regression rate:**

For what percentage did the repair NOT introduce new violations? A repair that fixes Quantity but creates a Quality violation is not successful.

**BLEU score:**

Measures n-gram overlap between the repair and the original good response (from your injection pairs). Higher BLEU suggests the repair is similar to what a human would produce.

**BERTScore:**

Uses BERT embeddings to measure semantic similarity between the repair and reference. More robust than BLEU for paraphrase scenarios.

### 12.2 Human Evaluation

Automatic metrics only go so far. You need human evaluation for your paper.

**Evaluation questions:**

1. Does this repair fix the indicated problem? (Yes/No)
2. Does this repair preserve the original meaning? (Yes/Partially/No) [For non-Quality repairs]
3. Does this repair align with the evidence? (Yes/No) [For Quality repairs]
4. Is this repair fluent and natural? (1-5 scale)
5. Would you prefer the repaired response over the original? (Prefer original / No preference / Prefer repaired)

**Evaluation setup:**

Create a set of 100-200 examples covering all violation types. For each example, show the evaluator the original response, the violation type, and the repaired response. Collect responses to the above questions.

**Reporting results:**

Report fix success rate (Q1), meaning preservation rate (Q2), evidence alignment rate (Q3), average fluency (Q4), and preference rate (Q5) in your paper.

---

# Part V: Closing the Loop with Generator Training

---

## Chapter 13: Training the Generator with Maxim Feedback

This optional but high-impact component uses your detector to train a dialogue generator that produces fewer violations in the first place.

### 13.1 Preference-Style Training with DPO

Direct Preference Optimization (DPO) is a technique for training language models on preference data without explicit reward modeling.

**How DPO works:**

Given a pair of responses (preferred and dispreferred), DPO directly adjusts the model to increase the probability of the preferred response and decrease the probability of the dispreferred response.

**Creating preference pairs:**

From your repair pipeline, you have natural preference pairs:

Preferred: Repaired response (cooperative)

Dispreferred: Original violated response

The "preference" comes from your linguistic principles (Gricean maxims), not human annotators.

### 13.2 DPO Training Procedure

**Starting point:**

Begin with a pre-trained dialogue model. Options include DialoGPT (open-domain), or a T5/BART model fine-tuned on Topical-Chat.

**Data preparation:**

For each example, you need: context, evidence, preferred response, dispreferred response. Format these as required by your DPO training code.

**Training:**

DPO fine-tunes the model to prefer cooperative responses. The loss function implicitly captures preference strength based on probability differences.

**Hyperparameters:**

Learning rate: 1e-6 to 5e-6 (low because you are making subtle preference adjustments)

Beta: A DPO-specific parameter controlling preference strength. Typical values are 0.1 to 0.5.

Epochs: 1-3 epochs. DPO can overfit quickly if run too long.

### 13.3 Alternative: Auxiliary Loss Training

Instead of DPO, you can add violation probabilities directly to the training loss.

**Approach:**

For each generated response during training:

1. Run the detector to get violation probabilities.
2. Add these probabilities (weighted) to the standard language modeling loss.
3. The generator learns to minimize both perplexity and predicted violations.

**Loss formulation:**

Total Loss = Language Model Loss + λ × (w_quantity × P(Quantity) + w_quality × P(Quality) + w_relation × P(Relation) + w_manner × P(Manner))

The weights (w_maxim) can be equal or tuned to prioritize certain maxims.

### 13.4 Adaptive Maxim Weighting

A publishable enhancement: learn to weight maxims differently based on context.

**Intuition:**

For a medical question, Quality (accuracy) matters most.

For a "explain simply" request, Manner matters most.

For a "be brief" request, Quantity matters most.

**Implementation:**

Train a small network that takes the context and outputs maxim weights. Use these weights in the auxiliary loss or as a conditional signal for DPO.

---

## Chapter 14: Evaluating the Full System

### 14.1 End-to-End Generation Evaluation

Generate responses from your trained generator on a held-out test set. Run your detector on these responses.

**Metrics:**

Violation rates: For each maxim, what percentage of responses are flagged as violations? Compare to a baseline generator (same model without maxim training).

Overall cooperative rate: What percentage of responses have no violations flagged?

### 14.2 Human Evaluation of Generation

Automatic detector scores are not enough—the detector might be wrong.

**Comparative evaluation:**

Show evaluators pairs of responses (one from baseline generator, one from your maxim-trained generator) for the same context. Ask which response is more cooperative/helpful.

**Attribute evaluation:**

Ask evaluators to rate each response on the four maxims separately. Are maxim-trained generator responses rated better?

### 14.3 Pragmatic Transfer Evaluation

Test whether your maxim-trained model generalizes to pragmatic understanding tasks.

**IMPPRES dataset:**

IMPPRES tests whether models understand implicatures (implied meanings) and presuppositions (assumed background). A model that truly understands cooperative communication should perform better on these tasks.

**Evaluation procedure:**

Take your trained generator and use it to answer IMPPRES questions. Compare accuracy to a baseline model. If your model shows gains, this suggests that learning from Gricean maxims transfers to pragmatic reasoning more broadly.

### 14.4 Ablation Studies for Generator Training

**Maxim removal ablations:**

Train versions where you remove each maxim's feedback. If removing Quality feedback hurts factual accuracy but not other dimensions, this confirms Quality feedback specifically teaches factuality.

**Detector quality ablation:**

Replace your trained detector with a random detector. If generator training no longer works, this confirms the detector is providing meaningful signal (not just noise that happens to regularize training).

---

# Part VI: Writing and Publishing Your Paper

---

## Chapter 15: Structuring Your Paper

A well-structured paper guides reviewers through your contribution clearly.

### 15.1 Title

The title should convey your core contribution in under 15 words.

Example: "GriceBench: Teaching Language Models Cooperative Communication via Maxim-Based Detection and Repair"

### 15.2 Abstract

The abstract is 200-300 words summarizing your entire paper. It should include:

Problem: One sentence on why current dialogue models fail at cooperative communication.

Gap: One sentence on why existing approaches do not solve this.

Your approach: Two to three sentences on your solution (detector, repair, generator training).

Results: Two sentences on key findings (detector F1, repair success rate, generation improvements).

Significance: One sentence on broader impact.

### 15.3 Introduction

The introduction expands the abstract to about one page. End with a bullet list of contributions:

"Our contributions are:

1. We formalize Gricean maxim violation detection as a multi-label classification task...
    
2. We release GriceBench, a dataset of X examples with controlled violations...
    
3. We demonstrate that maxim-based feedback improves dialogue generation by Y%..."
    

### 15.4 Related Work

Organize related work by themes, not just a list of papers. Suggested themes:

Pragmatics in NLP: Prior work discussing Gricean maxims conceptually. Contrast with your operational approach.

Dialogue evaluation: Metrics for dialogue quality. Position your maxim-based evaluation as complementary.

Controllable generation: Work on controlling style/length/etc. Position maxim control as a new, linguistically-grounded axis.

Preference learning: RLHF, DPO, etc. Contrast opaque preferences with your principle-grounded approach.

### 15.5 Task and Dataset

Clearly define:

Input/output specification for violation detection.

Your violation injection methodology with examples.

Annotation process and inter-annotator agreement.

Dataset statistics (number of examples, violation distribution, etc.).

### 15.6 Methods

Describe your three components:

Detector architecture and training procedure.

Repair model architecture and training procedure.

Generator training (if included).

Use equations sparingly and only if they add clarity.

### 15.7 Experiments

Organize by research question:

RQ1: Can we reliably detect maxim violations? (Detector results)

RQ2: Can we repair violations while preserving meaning? (Repair results)

RQ3: Does maxim feedback improve generation? (Generator results)

RQ4: Do improvements transfer to pragmatic reasoning? (IMPPRES results)

Include tables with clear captions and analysis of what the numbers mean.

### 15.8 Analysis

Go beyond numbers:

Error analysis: What types of errors does each component make?

Qualitative examples: Show good and bad cases.

Trade-off analysis: How do maxims interact? Do models learn to balance them?

### 15.9 Limitations and Ethical Considerations

Be honest about limitations:

Subjectivity of maxim judgments.

Limited to English.

Synthetic violations may not capture all natural violation patterns.

Potential misuse (gaming maxim detectors).

### 15.10 Conclusion

Summarize contributions and results. Suggest future work (multimodal maxims, cross-lingual studies, etc.).

---

## Chapter 16: Targeting Venues

### 16.1 Conference Options

**ACL, EMNLP, NAACL (Computational Linguistics):**

Most natural fit. Your work is fundamentally about language and pragmatics.

Submit to the Dialogue and Interactive Systems track or Semantics and Pragmatics track.

**NeurIPS, ICML (Machine Learning):**

Harder sell, but possible if you emphasize the novel training methodology (maxims as training signal).

**AAAI, IJCAI (Artificial Intelligence):**

Broader AI audience. Position as AI systems that communicate more effectively.

### 16.2 Timeline

Major conference deadlines are typically:

ACL: January

NAACL: December

EMNLP: June

NeurIPS: May

Plan to have a complete draft two weeks before the deadline to allow for polishing.

### 16.3 Pre-prints

Consider posting to arXiv after submission but before reviews return. This establishes priority and allows the community to see your work early. Be aware that some venues have policies about pre-prints.

---

## Chapter 17: Making Your Research Reproducible

Reproducibility is essential for scientific credibility and community impact.

### 17.1 Code Release

Create a GitHub repository with:

All code for data preprocessing, violation injection, training, and evaluation.

Clear README explaining how to run each component.

Requirements file listing all dependencies with version numbers.

Example scripts that can be run to reproduce key results.

### 17.2 Dataset Release

Release your GriceBench dataset:

Full violation-injected dataset with labels.

Gold-annotated test set.

Data documentation explaining each field.

### 17.3 Model Release

Upload trained model weights to Hugging Face Hub:

Violation detector.

Repair model.

Trained generator (if applicable).

Include model cards describing intended use and limitations.

### 17.4 Reproducibility Checklist

Before submission, verify:

Can someone else run your code from scratch?

Are random seeds specified for reproducibility?

Are all hyperparameters documented?

Are computational requirements (GPU hours, memory) reported?

---

# Part VII: Practical Timeline and Checklist

---

## Chapter 18: Twelve-Week Research Plan

### Week 1: Environment and Data Setup

Days 1-2: Set up Python environment, install all libraries, verify GPU access.

Days 3-5: Download Topical-Chat, explore data structure, write preprocessing code.

Days 6-7: Load sample conversations, calculate basic statistics, understand the domain.

Deliverable: Working data pipeline that extracts (context, evidence, response) tuples.

### Week 2: Violation Injection Implementation

Days 1-2: Implement Quantity violation transformations (under-informative, over-informative).

Days 3-4: Implement Quality violation transformations (unsupported, contradiction).

Day 5: Implement Relation violation transformations (topic substitution).

Days 6-7: Implement Manner violation transformations (ambiguity, shuffling, jargon).

Deliverable: Working violation injection code for all four maxims.

### Week 3: Weak Supervision Heuristics

Days 1-2: Implement Quantity heuristics (length ratio, redundancy detection).

Days 3-4: Implement Quality heuristics (NLI contradiction, retrieval matching).

Day 5: Implement Relation heuristics (semantic similarity, keyword overlap).

Days 6-7: Implement Manner heuristics (readability, coherence, ambiguity counting).

Deliverable: Working heuristic labeling code for all four maxims.

### Week 4: Dataset Generation

Days 1-3: Generate large weak-labeled dataset (50k+ examples).

Days 4-5: Run validation checks to filter bad injections.

Days 6-7: Create train/validation/test splits, verify balance.

Deliverable: Complete weak-labeled GriceBench dataset.

### Week 5: Gold Annotation Preparation and Pilot

Days 1-2: Finalize annotation rubric based on lessons from injection.

Days 3-4: Set up annotation tool (Label Studio or alternative).

Days 5-7: Run pilot annotation (50 examples), calculate agreement, revise rubric.

Deliverable: Validated annotation rubric and trained annotators.

### Week 6: Gold Annotation Completion

Days 1-5: Complete annotation of 1000 examples.

Days 6-7: Adjudicate disagreements, finalize gold labels.

Deliverable: Complete gold-labeled test set.
x`
### Week 7: Detector Training

Days 1-2: Implement detector architecture (multi-head DeBERTa classifier).

Days 3-4: Train on weak labels (Phase 1).

Days 5-6: Fine-tune on gold labels (Phase 2).

Day 7: Calibrate detector probabilities.

Deliverable: Trained and calibrated violation detector.

### Week 8: Detector Evaluation

Days 1-2: Calculate per-maxim F1, confusion matrices, calibration error.

Days 3-4: Conduct error analysis, identify failure patterns.

Days 5-7: Run ablation studies (no evidence, single-task).

Deliverable: Complete detector evaluation results and analysis.

### Week 9: Repair Model Training

Days 1-2: Prepare repair training data (violated → clean pairs).

Days 3-4: Train T5 repair model.

Days 5-7: Evaluate repair model (fix rate, no-regression, BLEU/BERTScore).

Deliverable: Trained repair model with automatic evaluation.

### Week 10: Human Evaluation

Days 1-3: Design human evaluation protocol, prepare evaluation set.

Days 4-6: Run human evaluation (100-200 examples).

Day 7: Analyze human evaluation results.

Deliverable: Human evaluation results for repair model.

### Week 11: Generator Training and Evaluation

Days 1-3: Prepare preference pairs, implement DPO training.

Days 4-5: Train generator with maxim feedback.

Days 6-7: Evaluate generator (violation rates, human preference, IMPPRES transfer).

Deliverable: Trained generator with complete evaluation.

### Week 12: Paper Writing and Submission

Days 1-3: Write Methods and Experiments sections.

Days 4-5: Write Introduction, Related Work, and Analysis.

Days 6: Write Abstract, Conclusion, and Limitations.

Day 7: Final polish, format check, submit.

Deliverable: Complete paper submission.

---

## Chapter 19: Troubleshooting Common Problems

### 19.1 Out of Memory Errors

Symptom: Training crashes with "CUDA out of memory" or similar.

Solutions:

Reduce batch size (most common fix).

Reduce maximum sequence length.

Use gradient accumulation (accumulate gradients over multiple smaller batches).

Use mixed precision training (uses less memory per parameter).

Move to a GPU with more memory if available.

### 19.2 Training Loss Not Decreasing

Symptom: Loss stays flat or oscillates randomly.

Solutions:

Learning rate may be too high or too low. Try different values.

Data may be shuffled incorrectly or have bugs. Verify a few examples manually.

Labels may be wrong. Check your weak supervision pipeline.

Model may be too small. Try a larger pre-trained model.

### 19.3 Validation Performance Much Worse Than Training

Symptom: High training accuracy, low validation accuracy.

Diagnosis: Overfitting