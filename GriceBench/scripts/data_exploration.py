"""
GriceBench Data Exploration Script
Explores Topical-Chat and FaithDial datasets to understand structure and statistics.
As described in Chapter 3 of the Implementation Guide.
"""

import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import statistics

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
TOPICAL_CHAT_DIR = DATA_RAW_DIR / "topical_chat"


def load_json(filepath):
    """Load JSON file safely."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def explore_topical_chat():
    """
    Explore Topical-Chat dataset structure and calculate key statistics.
    
    Key questions from Implementation Guide Section 3.4:
    - How long are typical responses?
    - How often do responses use knowledge snippets?
    - What's the vocabulary like?
    - How many turns per conversation?
    """
    print("=" * 70)
    print("TOPICAL-CHAT DATASET EXPLORATION")
    print("=" * 70)
    
    # Load train split for exploration
    train_path = TOPICAL_CHAT_DIR / "train.json"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found!")
        return None
    
    print(f"\nLoading {train_path.name}...")
    conversations = load_json(train_path)
    
    if not conversations:
        return None
    
    # Dataset structure exploration
    print(f"\n{'='*50}")
    print("DATASET STRUCTURE")
    print(f"{'='*50}")
    
    num_conversations = len(conversations)
    print(f"Number of conversations: {num_conversations}")
    
    # Examine first conversation to understand structure
    first_conv_id = list(conversations.keys())[0]
    first_conv = conversations[first_conv_id]
    print(f"\nSample conversation ID: {first_conv_id}")
    print(f"Conversation keys: {list(first_conv.keys())}")
    
    # Count turns and analyze structure
    all_turns = []
    response_lengths = []
    turns_per_conversation = []
    agent_messages = []
    vocabulary = Counter()
    knowledge_usage_count = 0
    total_messages_with_knowledge_field = 0
    
    for conv_id, conv_data in conversations.items():
        content = conv_data.get('content', [])
        turns_per_conversation.append(len(content))
        
        for turn in content:
            all_turns.append(turn)
            message = turn.get('message', '')
            agent = turn.get('agent', '')
            
            # Track response lengths
            words = message.split()
            response_lengths.append(len(words))
            
            # Update vocabulary
            vocabulary.update(word.lower() for word in words)
            
            # Track knowledge usage
            if 'knowledge_source' in turn:
                total_messages_with_knowledge_field += 1
                knowledge = turn.get('knowledge_source')
                if knowledge and knowledge != 'NULL':
                    knowledge_usage_count += 1
            
            if agent:
                agent_messages.append({'agent': agent, 'message': message})
    
    print(f"\n{'='*50}")
    print("BASIC STATISTICS")
    print(f"{'='*50}")
    
    # Response length statistics (for Quantity violation thresholds)
    print(f"\n** Response Length Statistics (words per message) **")
    print(f"  Total messages: {len(response_lengths)}")
    print(f"  Mean length: {statistics.mean(response_lengths):.1f} words")
    print(f"  Median length: {statistics.median(response_lengths):.1f} words")
    print(f"  Std deviation: {statistics.stdev(response_lengths):.1f} words")
    print(f"  Min length: {min(response_lengths)} words")
    print(f"  Max length: {max(response_lengths)} words")
    
    # Percentiles for Quantity thresholds
    sorted_lengths = sorted(response_lengths)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in percentiles:
        idx = int(len(sorted_lengths) * p / 100)
        print(f"    {p}th percentile: {sorted_lengths[idx]} words")
    
    # Turns per conversation
    print(f"\n** Turns Per Conversation **")
    print(f"  Mean turns: {statistics.mean(turns_per_conversation):.1f}")
    print(f"  Median turns: {statistics.median(turns_per_conversation):.1f}")
    print(f"  Min turns: {min(turns_per_conversation)}")
    print(f"  Max turns: {max(turns_per_conversation)}")
    
    # Vocabulary statistics (for Manner violation thresholds)
    print(f"\n** Vocabulary Statistics **")
    print(f"  Unique words: {len(vocabulary)}")
    print(f"  Total word tokens: {sum(vocabulary.values())}")
    print(f"  Top 20 most common words:")
    for word, count in vocabulary.most_common(20):
        print(f"    '{word}': {count}")
    
    # Knowledge usage (for Quality violation detection)
    print(f"\n** Knowledge/Evidence Usage **")
    print(f"  Messages with knowledge field: {total_messages_with_knowledge_field}")
    print(f"  Messages using knowledge: {knowledge_usage_count}")
    if total_messages_with_knowledge_field > 0:
        usage_rate = knowledge_usage_count / total_messages_with_knowledge_field * 100
        print(f"  Knowledge usage rate: {usage_rate:.1f}%")
    
    # Load and explore reading sets (knowledge/evidence)
    print(f"\n{'='*50}")
    print("READING SETS (KNOWLEDGE/EVIDENCE) STRUCTURE")
    print(f"{'='*50}")
    
    reading_sets_dir = TOPICAL_CHAT_DIR / "reading_sets"
    if reading_sets_dir.exists():
        train_knowledge_path = reading_sets_dir / "train.json"
        if train_knowledge_path.exists():
            reading_sets = load_json(train_knowledge_path)
            if reading_sets:
                print(f"\n  Number of reading set entries: {len(reading_sets)}")
                
                # Sample a reading set to understand structure
                first_key = list(reading_sets.keys())[0]
                first_entry = reading_sets[first_key]
                print(f"  Sample entry key: {first_key}")
                print(f"  Entry structure: {type(first_entry)}")
                
                if isinstance(first_entry, dict):
                    print(f"  Entry keys: {list(first_entry.keys())[:5]}...")
    
    # Extract sample (context, evidence, response) tuple
    print(f"\n{'='*50}")
    print("SAMPLE (CONTEXT, EVIDENCE, RESPONSE) EXTRACTION")
    print(f"{'='*50}")
    
    # Get a conversation with knowledge usage
    for conv_id, conv_data in list(conversations.items())[:20]:
        content = conv_data.get('content', [])
        for i, turn in enumerate(content):
            knowledge = turn.get('knowledge_source')
            if knowledge and knowledge != 'NULL' and i > 0:
                # Found a turn that uses knowledge
                context_turns = content[:i]
                context = " | ".join([t.get('message', '')[:50] for t in context_turns[-3:]])
                response = turn.get('message', '')
                
                print(f"\n  Conversation: {conv_id}")
                print(f"  Turn index: {i}")
                print(f"  Context (last 3 turns): {context[:200]}...")
                print(f"  Evidence: {str(knowledge)[:200]}...")
                print(f"  Response: {response[:200]}...")
                break
        else:
            continue
        break
    
    # Summary statistics for violation detection thresholds
    print(f"\n{'='*50}")
    print("RECOMMENDED THRESHOLDS FOR VIOLATION DETECTION")
    print(f"{'='*50}")
    
    mean_len = statistics.mean(response_lengths)
    std_len = statistics.stdev(response_lengths)
    
    print(f"\n** Quantity Violation Thresholds **")
    print(f"  Too short (< mean - 1.5*std): < {mean_len - 1.5*std_len:.0f} words")
    print(f"  Too long (> mean + 2*std): > {mean_len + 2*std_len:.0f} words")
    print(f"  Or use percentiles:")
    print(f"    Too short: < 10th percentile = {sorted_lengths[int(len(sorted_lengths)*0.1)]} words")
    print(f"    Too long: > 95th percentile = {sorted_lengths[int(len(sorted_lengths)*0.95)]} words")
    
    return {
        'num_conversations': num_conversations,
        'total_turns': len(all_turns),
        'mean_response_length': mean_len,
        'median_response_length': statistics.median(response_lengths),
        'std_response_length': std_len,
        'vocabulary_size': len(vocabulary),
        'mean_turns_per_conversation': statistics.mean(turns_per_conversation),
        'knowledge_usage_rate': knowledge_usage_count / total_messages_with_knowledge_field if total_messages_with_knowledge_field > 0 else 0
    }


def main():
    print("\n" + "=" * 70)
    print("GRICEBENCH DATA EXPLORATION")
    print("=" * 70)
    print("\nThis script explores the datasets to understand structure and")
    print("determine appropriate thresholds for violation detection.")
    print("(As described in Implementation Guide, Chapter 3, Section 3.4)")
    
    # Explore Topical-Chat
    stats = explore_topical_chat()
    
    if stats:
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")
        print(f"\nTopical-Chat Statistics:")
        print(f"  - Conversations: {stats['num_conversations']:,}")
        print(f"  - Total turns: {stats['total_turns']:,}")
        print(f"  - Mean response length: {stats['mean_response_length']:.1f} words")
        print(f"  - Vocabulary size: {stats['vocabulary_size']:,} unique words")
        print(f"  - Mean turns/conversation: {stats['mean_turns_per_conversation']:.1f}")
        print(f"  - Knowledge usage rate: {stats['knowledge_usage_rate']*100:.1f}%")
    
    print(f"\n{'='*70}")
    print("Exploration complete! Ready for violation injection pipeline.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
