"""
GriceBench Response Corpus Creator - Part 1, Step 1
====================================================

Creates a high-quality topical response corpus from free HuggingFace datasets
for use in retrieval-augmented Relation repair.

This addresses the core problem: Relation violations require generating NEW content
about a different topic, not editing existing text. We build a retrieval corpus
that maps contexts to relevant responses.

Datasets used (all free, permissive licenses):
- daily_dialog: Daily conversations
- empathetic_dialogues: Emotionally aware dialogues
- blended_skill_talk: Multi-skill conversations

Author: GriceBench
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CorpusConfig:
    """Configuration for corpus creation."""
    output_path: str = "data_processed/topical_corpus.json"
    min_response_length: int = 10  # Minimum words
    max_response_length: int = 150  # Maximum words
    min_context_length: int = 5
    max_samples_per_topic: int = 5000  # Cap per topic to balance corpus
    deduplicate: bool = True


# ============================================================================
# TOPIC TAXONOMY
# ============================================================================

# Comprehensive topic taxonomy with keywords (hierarchical for better matching)
TOPIC_TAXONOMY: Dict[str, List[str]] = {
    "weather": [
        "weather", "rain", "rainy", "sunny", "cloud", "cloudy", "snow", "snowy",
        "cold", "hot", "warm", "temperature", "storm", "thunder", "lightning",
        "humid", "humidity", "forecast", "climate", "wind", "windy", "fog"
    ],
    "food": [
        "food", "eat", "eating", "restaurant", "cook", "cooking", "meal", "hungry",
        "dinner", "lunch", "breakfast", "recipe", "delicious", "taste", "tasty",
        "pizza", "burger", "salad", "soup", "dessert", "cake", "coffee", "tea",
        "vegetarian", "vegan", "cuisine", "chef", "kitchen", "bake", "grill"
    ],
    "work": [
        "work", "job", "office", "boss", "meeting", "project", "deadline",
        "colleague", "career", "salary", "promotion", "interview", "resume",
        "employee", "employer", "company", "business", "corporate", "manager",
        "profession", "occupation", "workplace", "retire", "retirement"
    ],
    "family": [
        "family", "mother", "father", "mom", "dad", "sister", "brother",
        "parents", "children", "kids", "son", "daughter", "grandma", "grandpa",
        "grandmother", "grandfather", "uncle", "aunt", "cousin", "relative",
        "husband", "wife", "spouse", "married", "wedding", "baby"
    ],
    "travel": [
        "travel", "trip", "vacation", "flight", "hotel", "visit", "tourist",
        "tourism", "airport", "plane", "train", "bus", "car", "road",
        "destination", "beach", "mountain", "city", "country", "abroad",
        "passport", "luggage", "suitcase", "journey", "adventure"
    ],
    "health": [
        "health", "doctor", "sick", "medicine", "hospital", "pain", "illness",
        "disease", "symptom", "treatment", "healthy", "exercise", "workout",
        "gym", "fitness", "diet", "weight", "sleep", "tired", "energy",
        "stress", "mental", "therapy", "vitamin", "immune"
    ],
    "entertainment": [
        "movie", "film", "cinema", "music", "song", "concert", "show",
        "television", "tv", "netflix", "game", "gaming", "video", "youtube",
        "book", "read", "reading", "novel", "story", "podcast", "radio",
        "theater", "play", "comedy", "drama", "actor", "actress"
    ],
    "sports": [
        "sport", "sports", "game", "team", "play", "playing", "win", "winning",
        "match", "championship", "football", "soccer", "basketball", "baseball",
        "tennis", "golf", "swimming", "running", "athlete", "coach", "score",
        "tournament", "olympics", "world cup", "league"
    ],
    "education": [
        "school", "study", "studying", "learn", "learning", "class", "teacher",
        "student", "college", "university", "degree", "graduate", "exam",
        "test", "homework", "assignment", "professor", "lecture", "course",
        "education", "academic", "knowledge", "library"
    ],
    "technology": [
        "computer", "phone", "smartphone", "internet", "app", "application",
        "software", "hardware", "tech", "technology", "digital", "online",
        "website", "social media", "facebook", "instagram", "twitter",
        "programming", "code", "coding", "ai", "artificial intelligence",
        "robot", "gadget", "device"
    ],
    "pets": [
        "pet", "dog", "cat", "puppy", "kitten", "animal", "animals",
        "bird", "fish", "hamster", "rabbit", "veterinarian", "vet",
        "walk", "feed", "feeding", "adopt", "adoption", "shelter"
    ],
    "hobbies": [
        "hobby", "hobbies", "art", "painting", "drawing", "photography",
        "garden", "gardening", "craft", "crafts", "diy", "collection",
        "collect", "collecting", "dance", "dancing", "sing", "singing",
        "write", "writing", "instrument", "piano", "guitar"
    ],
    "shopping": [
        "shop", "shopping", "buy", "buying", "store", "mall", "price",
        "expensive", "cheap", "sale", "discount", "online shopping",
        "amazon", "order", "delivery", "clothes", "fashion", "brand"
    ],
    "relationship": [
        "friend", "friends", "friendship", "relationship", "date", "dating",
        "boyfriend", "girlfriend", "partner", "love", "romantic", "romance",
        "breakup", "together", "couple", "single"
    ],
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def extract_topic(text: str) -> str:
    """
    Extract the most relevant topic from text using weighted keyword matching.
    
    Uses a scoring system that considers:
    - Number of keyword matches
    - Position of keywords (earlier = more weight)
    - Keyword specificity
    """
    if not text:
        return "general"
    
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    
    topic_scores: Dict[str, float] = defaultdict(float)
    
    for topic, keywords in TOPIC_TAXONOMY.items():
        for keyword in keywords:
            # Check for exact word match
            if keyword in words:
                # Base score
                score = 1.0
                
                # Bonus for longer keywords (more specific)
                if len(keyword) > 6:
                    score += 0.5
                
                # Bonus for early occurrence
                pos = text_lower.find(keyword)
                if pos != -1 and pos < len(text_lower) / 2:
                    score += 0.3
                
                topic_scores[topic] += score
            
            # Check for partial match in compound words
            elif keyword in text_lower:
                topic_scores[topic] += 0.5
    
    if not topic_scores:
        return "general"
    
    # Return topic with highest score
    return max(topic_scores.items(), key=lambda x: x[1])[0]


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common noise patterns
    text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content like [laughs]
    text = re.sub(r'<.*?>', '', text)    # Remove HTML-like tags
    
    return text.strip()


def is_quality_response(response: str, context: str, config: CorpusConfig) -> bool:
    """
    Check if a response meets quality criteria for inclusion.
    
    Filters out:
    - Too short/long responses
    - Responses with no substance
    - Responses that are just questions
    - Responses with too much repetition
    """
    words = response.split()
    word_count = len(words)
    
    # Length checks
    if word_count < config.min_response_length:
        return False
    if word_count > config.max_response_length:
        return False
    
    # Content quality checks
    # Reject if mostly questions
    question_count = response.count('?')
    if question_count > 2:
        return False
    
    # Reject high repetition (same word > 30% of response)
    word_freq = defaultdict(int)
    for word in words:
        word_freq[word.lower()] += 1
    max_freq = max(word_freq.values()) if word_freq else 0
    if max_freq > word_count * 0.3 and word_count > 10:
        return False
    
    # Reject very short words only (like "ok ok ok")
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 3:
        return False
    
    return True


def get_response_hash(response: str) -> str:
    """Create hash for deduplication."""
    # Normalize for hashing
    normalized = response.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def load_daily_dialog() -> List[Dict]:
    """Load and process daily_dialog dataset."""
    examples = []
    
    try:
        from datasets import load_dataset
        print("  Loading daily_dialog...")
        dataset = load_dataset("daily_dialog", split="train", trust_remote_code=True)
        
        for item in dataset:
            dialog = item.get("dialog", [])
            if len(dialog) < 2:
                continue
            
            for i in range(1, len(dialog)):
                context = clean_text(dialog[i-1])
                response = clean_text(dialog[i])
                
                if context and response:
                    examples.append({
                        "context": context,
                        "response": response,
                        "source": "daily_dialog"
                    })
        
        print(f"    Extracted {len(examples):,} pairs from daily_dialog")
        
    except Exception as e:
        print(f"    Warning: Could not load daily_dialog: {e}")
    
    return examples


def load_empathetic_dialogues() -> List[Dict]:
    """Load and process empathetic_dialogues dataset."""
    examples = []
    
    try:
        from datasets import load_dataset
        print("  Loading empathetic_dialogues...")
        dataset = load_dataset("empathetic_dialogues", split="train", trust_remote_code=True)
        
        # Group by conversation ID
        conversations = defaultdict(list)
        for item in dataset:
            conv_id = item.get("conv_id", "")
            utterance = clean_text(item.get("utterance", ""))
            if conv_id and utterance:
                conversations[conv_id].append(utterance)
        
        # Extract context-response pairs
        for conv_id, utterances in conversations.items():
            for i in range(1, len(utterances)):
                context = utterances[i-1]
                response = utterances[i]
                
                if context and response:
                    examples.append({
                        "context": context,
                        "response": response,
                        "source": "empathetic_dialogues"
                    })
        
        print(f"    Extracted {len(examples):,} pairs from empathetic_dialogues")
        
    except Exception as e:
        print(f"    Warning: Could not load empathetic_dialogues: {e}")
    
    return examples


def load_blended_skill_talk() -> List[Dict]:
    """Load and process blended_skill_talk dataset."""
    examples = []
    
    try:
        from datasets import load_dataset
        print("  Loading blended_skill_talk...")
        dataset = load_dataset("blended_skill_talk", split="train", trust_remote_code=True)
        
        for item in dataset:
            # Get free_messages (the conversation turns)
            previous = item.get("previous_utterance", [])
            free_messages = item.get("free_messages", [])
            guided_messages = item.get("guided_messages", [])
            
            # Combine all messages
            all_messages = list(previous) + list(free_messages) + list(guided_messages)
            
            for i in range(1, len(all_messages)):
                context = clean_text(all_messages[i-1])
                response = clean_text(all_messages[i])
                
                if context and response:
                    examples.append({
                        "context": context,
                        "response": response,
                        "source": "blended_skill_talk"
                    })
        
        print(f"    Extracted {len(examples):,} pairs from blended_skill_talk")
        
    except Exception as e:
        print(f"    Warning: Could not load blended_skill_talk: {e}")
    
    return examples


def create_topical_corpus(config: CorpusConfig = None) -> Dict[str, List[Dict]]:
    """
    Create a high-quality topical corpus from multiple dialogue datasets.
    
    Returns:
        Dict mapping topic names to lists of {context, response} pairs
    """
    if config is None:
        config = CorpusConfig()
    
    print("="*70)
    print("CREATING TOPICAL RESPONSE CORPUS")
    print("="*70)
    
    # Load all datasets
    print("\n1. Loading datasets...")
    all_examples = []
    all_examples.extend(load_daily_dialog())
    all_examples.extend(load_empathetic_dialogues())
    all_examples.extend(load_blended_skill_talk())
    
    print(f"\n   Total raw examples: {len(all_examples):,}")
    
    # Filter for quality
    print("\n2. Filtering for quality...")
    quality_examples = []
    for ex in all_examples:
        if is_quality_response(ex["response"], ex["context"], config):
            quality_examples.append(ex)
    
    print(f"   After quality filter: {len(quality_examples):,} ({len(quality_examples)/len(all_examples)*100:.1f}%)")
    
    # Deduplicate
    if config.deduplicate:
        print("\n3. Deduplicating...")
        seen_hashes = set()
        unique_examples = []
        for ex in quality_examples:
            h = get_response_hash(ex["response"])
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_examples.append(ex)
        
        print(f"   After dedup: {len(unique_examples):,} ({len(unique_examples)/len(quality_examples)*100:.1f}%)")
        quality_examples = unique_examples
    
    # Organize by topic
    print("\n4. Organizing by topic...")
    corpus: Dict[str, List[Dict]] = defaultdict(list)
    
    for ex in quality_examples:
        # Extract topic from both context and response
        combined_text = f"{ex['context']} {ex['response']}"
        topic = extract_topic(combined_text)
        
        corpus[topic].append({
            "context": ex["context"],
            "response": ex["response"],
            "source": ex["source"]
        })
    
    # Cap samples per topic for balance
    print("\n5. Balancing corpus...")
    for topic in corpus:
        if len(corpus[topic]) > config.max_samples_per_topic:
            # Keep diverse samples by taking every Nth
            step = len(corpus[topic]) // config.max_samples_per_topic
            corpus[topic] = corpus[topic][::step][:config.max_samples_per_topic]
    
    # Print statistics
    print("\n" + "="*70)
    print("CORPUS STATISTICS")
    print("="*70)
    total = sum(len(v) for v in corpus.values())
    print(f"\nTotal examples: {total:,}")
    print(f"Topics: {len(corpus)}")
    print("\nPer-topic breakdown:")
    for topic in sorted(corpus.keys(), key=lambda x: len(corpus[x]), reverse=True):
        count = len(corpus[topic])
        sources = set(ex["source"] for ex in corpus[topic])
        print(f"  {topic:15s}: {count:,} examples from {', '.join(sources)}")
    
    return dict(corpus)


def save_corpus(corpus: Dict[str, List[Dict]], output_path: str):
    """Save corpus to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved corpus to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Create the response corpus."""
    config = CorpusConfig()
    
    # Create corpus
    corpus = create_topical_corpus(config)
    
    # Save
    save_corpus(corpus, config.output_path)
    
    print("\n" + "="*70)
    print("CORPUS CREATION COMPLETE!")
    print("="*70)
    print(f"\nNext step: Run build_retrieval_system.py to create FAISS index")


if __name__ == "__main__":
    main()
