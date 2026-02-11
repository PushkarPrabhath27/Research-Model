"""
GriceBench Violation Injectors
==============================

This module implements controlled violation injection for each Gricean maxim.
These injectors create synthetic training data with known ground-truth labels.

The four Gricean maxims:
1. QUANTITY - Say enough, but not too much
2. QUALITY - Only say what is true and supported by evidence
3. RELATION - Stay on topic
4. MANNER - Be clear and organized

Key Design Principles:
- Each injector creates violations for ONE maxim only (for single-violation examples)
- Transformations preserve other maxims (e.g., Quantity changes don't affect Relation)
- Both subtle and obvious violations are generated for robust training
- Anti-shortcut measures prevent spurious correlations (e.g., length != violation)

Based on Chapter 4 of the GriceBench Implementation Guide.
"""

import random
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ViolationExample:
    """
    A training example with a known violation.
    
    Attributes:
        original_response: The clean, cooperative response
        violated_response: The response after violation injection
        violation_type: Type of violation (e.g., "quantity_under", "quality_contradiction")
        maxim: Which Gricean maxim is violated (quantity, quality, relation, manner)
        context: Conversation context
        evidence: Available evidence/knowledge
        metadata: Additional info about the transformation applied
    """
    original_response: str
    violated_response: str
    violation_type: str
    maxim: str
    context: str
    evidence: Optional[Any]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'original_response': self.original_response,
            'violated_response': self.violated_response,
            'violation_type': self.violation_type,
            'maxim': self.maxim,
            'context': self.context,
            'evidence': self.evidence,
            'metadata': self.metadata,
            'labels': self._get_labels()
        }
    
    def _get_labels(self) -> Dict[str, int]:
        """Get binary labels for each maxim (1 = violated)."""
        return {
            'quantity': 1 if self.maxim == 'quantity' else 0,
            'quality': 1 if self.maxim == 'quality' else 0,
            'relation': 1 if self.maxim == 'relation' else 0,
            'manner': 1 if self.maxim == 'manner' else 0
        }


@dataclass 
class CleanExample:
    """A clean example with no violations (negative training example)."""
    response: str
    context: str
    evidence: Optional[Any]
    
    def to_dict(self) -> Dict:
        return {
            'original_response': self.response,
            'violated_response': self.response,  # Same as original
            'violation_type': 'none',
            'maxim': 'none',
            'context': self.context,
            'evidence': self.evidence,
            'metadata': {'is_clean': True},
            'labels': {'quantity': 0, 'quality': 0, 'relation': 0, 'manner': 0}
        }


# ============================================================================
# BASE INJECTOR CLASS
# ============================================================================

class BaseViolationInjector(ABC):
    """
    Abstract base class for all violation injectors.
    
    Each injector must implement inject() to create violations for its maxim.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random = random.Random(random_seed)
        self.maxim_name = "base"
    
    @abstractmethod
    def inject(self, response: str, context: str, evidence: Any = None) -> List[ViolationExample]:
        """
        Create violation(s) from a clean response.
        
        Args:
            response: The original clean response
            context: Conversation context
            evidence: Available knowledge/evidence
            
        Returns:
            List of ViolationExample objects with different violation types
        """
        pass
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text."""
        return re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
    
    def _extract_named_entities(self, text: str) -> List[str]:
        """
        Simple named entity extraction (capitalized words).
        For production, use spaCy NER.
        """
        words = text.split()
        entities = []
        for word in words:
            # Skip first word (often capitalized)
            if word[0].isupper() and len(word) > 1:
                clean = re.sub(r'[^\w]', '', word)
                if clean and clean[0].isupper():
                    entities.append(clean)
        return entities


# ============================================================================
# QUANTITY VIOLATION INJECTOR
# ============================================================================

class QuantityInjector(BaseViolationInjector):
    """
    Injects Quantity violations: responses that say too little or too much.
    
    Under-informative: Replaces specific content with vague acknowledgments
    Over-informative: Adds redundant paraphrases and unnecessary repetition
    
    Key constraint: Must NOT change topic (Relation) or truth (Quality)
    """
    
    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed)
        self.maxim_name = "quantity"
        
        # Vague phrases for under-informative violations
        self.vague_phrases = [
            "That's interesting.",
            "It depends on various factors.",
            "There are many aspects to consider.",
            "It's quite complex.",
            "That's a good question.",
            "There's a lot to say about that.",
            "It varies.",
            "Many people wonder about this.",
            "It's worth thinking about.",
            "There are different perspectives on this.",
        ]
        
        # Filler phrases for over-informative violations
        self.filler_phrases = [
            "In other words,",
            "To put it another way,",
            "As I mentioned,",
            "To elaborate further,",
            "Additionally, it's worth noting that",
            "Furthermore,",
            "In fact,",
            "That is to say,",
            "More specifically,",
            "To be more precise,",
        ]
        
        # Redundancy templates
        self.redundancy_templates = [
            "which means {paraphrase}",
            "that is, {paraphrase}",
            "in other words, {paraphrase}",
            "to put it simply, {paraphrase}",
        ]
    
    def inject(self, response: str, context: str, evidence: Any = None) -> List[ViolationExample]:
        """Create under-informative and over-informative violations."""
        violations = []
        
        # Only inject if response is substantial enough
        word_count = self._count_words(response)
        
        # Under-informative: works best on longer responses
        if word_count >= 10:
            under_resp = self._make_under_informative(response, context)
            if under_resp and self._count_words(under_resp) < word_count * 0.5:
                violations.append(ViolationExample(
                    original_response=response,
                    violated_response=under_resp,
                    violation_type="quantity_under",
                    maxim="quantity",
                    context=context,
                    evidence=evidence,
                    metadata={
                        'original_words': word_count,
                        'violated_words': self._count_words(under_resp),
                        'reduction_ratio': self._count_words(under_resp) / word_count
                    }
                ))
        
        # Over-informative: works best on medium responses
        if 8 <= word_count <= 60:
            over_resp = self._make_over_informative(response, context)
            if over_resp and self._count_words(over_resp) > word_count * 1.8:
                violations.append(ViolationExample(
                    original_response=response,
                    violated_response=over_resp,
                    violation_type="quantity_over",
                    maxim="quantity",
                    context=context,
                    evidence=evidence,
                    metadata={
                        'original_words': word_count,
                        'violated_words': self._count_words(over_resp),
                        'expansion_ratio': self._count_words(over_resp) / word_count
                    }
                ))
        
        return violations
    
    def _make_under_informative(self, response: str, context: str) -> Optional[str]:
        """
        Create under-informative version by removing specific content.
        
        Strategy:
        1. Extract topic keywords from context to preserve on-topic appearance
        2. Replace specific information with vague acknowledgments
        3. Keep it short but not empty
        """
        # Extract potential topic words from context
        context_words = set(w.lower() for w in context.split() if len(w) > 4)
        response_words = response.split()
        
        # Find topic-relevant words to keep
        topic_words = []
        for w in response_words[:5]:  # Check first few words
            if w.lower().rstrip('.,!?') in context_words:
                topic_words.append(w)
        
        # Build vague response
        if topic_words:
            topic_phrase = ' '.join(topic_words[:2])
            vague = self.random.choice([
                f"{topic_phrase}... that's interesting to consider.",
                f"Regarding {topic_words[0].lower().rstrip('.,!?')}, there are various aspects.",
                f"Yes, {topic_words[0].lower().rstrip('.,!?')} is quite something.",
            ])
        else:
            vague = self.random.choice(self.vague_phrases)
        
        return vague
    
    def _make_over_informative(self, response: str, context: str) -> Optional[str]:
        """
        Create over-informative version by adding redundancy.
        
        Strategy:
        1. Repeat key facts in different words
        2. Add filler phrases
        3. Include unnecessary elaborations
        4. Preserve truth (don't add new false information)
        """
        sentences = self._extract_sentences(response)
        if not sentences:
            return None
        
        # Build over-informative version
        result_parts = []
        
        for i, sentence in enumerate(sentences):
            result_parts.append(sentence)
            
            # Add redundancy after first sentence
            if i == 0:
                # Paraphrase the opening
                filler = self.random.choice(self.filler_phrases)
                paraphrase = self._create_simple_paraphrase(sentence)
                if paraphrase:
                    result_parts.append(f"{filler} {paraphrase}")
            
            # Add more redundancy for longer responses
            if i == len(sentences) - 1 and len(sentences) > 1:
                # Summarize at the end
                result_parts.append(f"So, to summarize, {sentences[0].lower()}")
        
        # Also repeat any numbers mentioned
        numbers = self._extract_numbers(response)
        if numbers:
            num = numbers[0]
            result_parts.append(f"That's right - {num}.")
        
        return ' '.join(result_parts)
    
    def _create_simple_paraphrase(self, sentence: str) -> Optional[str]:
        """Create a simple paraphrase by restructuring."""
        words = sentence.split()
        if len(words) < 5:
            return None
        
        # Very simple paraphrase: reorder and rephrase
        if words[0].lower() in ['the', 'a', 'an']:
            # "The X is Y" -> "Y is what the X is"
            return sentence.lower()
        
        return sentence.lower()


# ============================================================================
# QUALITY VIOLATION INJECTOR
# ============================================================================

class QualityInjector(BaseViolationInjector):
    """
    Injects Quality violations: responses with false or unsupported information.
    
    Unsupported claims: Adds plausible-sounding facts not in evidence
    Contradictions: Alters facts to contradict the evidence
    
    Key constraint: Must NOT change topic (Relation) or clarity (Manner)
    """
    
    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed)
        self.maxim_name = "quality"
        
        # Unsupported claim templates (plausible-sounding additions)
        self.unsupported_templates = [
            ", and is considered one of the most {adjective} in history",
            ", which has been studied extensively by researchers worldwide",
            ". Interestingly, this accounts for approximately {percent}% of all cases",
            ", making it a popular subject among {field} experts",
            ". Statistics show this affects millions of people annually",
            ", according to recent studies",
            ". Many experts believe this will continue to grow",
            ", which has significant implications for the future",
        ]
        
        self.adjectives = ["significant", "remarkable", "important", "notable", "impressive"]
        self.fields = ["science", "history", "technology", "economics", "psychology"]
        
    def inject(self, response: str, context: str, evidence: Any = None) -> List[ViolationExample]:
        """Create unsupported and contradiction violations."""
        violations = []
        word_count = self._count_words(response)
        
        # Need reasonable length response
        if word_count < 5:
            return violations
        
        # Unsupported claim injection
        unsupported_resp = self._inject_unsupported_claim(response)
        if unsupported_resp:
            violations.append(ViolationExample(
                original_response=response,
                violated_response=unsupported_resp,
                violation_type="quality_unsupported",
                maxim="quality",
                context=context,
                evidence=evidence,
                metadata={
                    'injection_type': 'unsupported_claim',
                    'original_words': word_count,
                    'violated_words': self._count_words(unsupported_resp)
                }
            ))
        
        # Contradiction injection (if there are numbers or specific facts)
        numbers = self._extract_numbers(response)
        if numbers:
            contradiction_resp = self._inject_contradiction(response, numbers)
            if contradiction_resp:
                violations.append(ViolationExample(
                    original_response=response,
                    violated_response=contradiction_resp,
                    violation_type="quality_contradiction",
                    maxim="quality",
                    context=context,
                    evidence=evidence,
                    metadata={
                        'injection_type': 'contradiction',
                        'original_numbers': numbers,
                        'modified': True
                    }
                ))
        
        return violations
    
    def _inject_unsupported_claim(self, response: str) -> str:
        """Add a plausible but unsupported claim to the response."""
        sentences = self._extract_sentences(response)
        if not sentences:
            return response
        
        # Pick a template and fill it
        template = self.random.choice(self.unsupported_templates)
        
        filled = template.format(
            adjective=self.random.choice(self.adjectives),
            field=self.random.choice(self.fields),
            percent=self.random.randint(15, 85)
        )
        
        # Add to first sentence (most impactful)
        first_sentence = sentences[0].rstrip('.')
        sentences[0] = first_sentence + filled + '.'
        
        return ' '.join(sentences)
    
    def _inject_contradiction(self, response: str, numbers: List[str]) -> Optional[str]:
        """Change a number to create a contradiction."""
        if not numbers:
            return None
        
        target_num = numbers[0]
        
        # Decide: subtle or obvious contradiction
        is_subtle = self.random.random() < 0.5
        
        try:
            # Parse number
            num_clean = target_num.replace(',', '')
            if '.' in num_clean:
                original_val = float(num_clean)
                if is_subtle:
                    # Subtle: change by 5-15%
                    factor = self.random.uniform(0.85, 0.95) if self.random.random() < 0.5 else self.random.uniform(1.05, 1.15)
                    new_val = original_val * factor
                    new_num = f"{new_val:.1f}"
                else:
                    # Obvious: change by 50-200%
                    factor = self.random.uniform(0.3, 0.6) if self.random.random() < 0.5 else self.random.uniform(1.8, 3.0)
                    new_val = original_val * factor
                    new_num = f"{new_val:.1f}"
            else:
                original_val = int(num_clean)
                if is_subtle:
                    delta = max(1, int(original_val * 0.1))
                    new_val = original_val + self.random.choice([-delta, delta])
                else:
                    new_val = original_val * self.random.choice([2, 3]) if original_val < 100 else original_val // 2
                
                # Preserve comma formatting if original had it
                if ',' in target_num:
                    new_num = f"{new_val:,}"
                else:
                    new_num = str(new_val)
            
            return response.replace(target_num, new_num, 1)
            
        except (ValueError, ZeroDivisionError):
            return None


# ============================================================================
# RELATION VIOLATION INJECTOR
# ============================================================================

class RelationInjector(BaseViolationInjector):
    """
    Injects Relation violations: responses that don't address the question.
    
    Off-topic substitution: Replace with response from different domain
    Same-domain drift: Stay in domain but answer different question
    
    Key constraint: Substituted responses must be similar LENGTH (anti-shortcut)
    """
    
    def __init__(self, response_pool: List[Dict] = None, random_seed: int = 42):
        super().__init__(random_seed)
        self.maxim_name = "relation"
        self.response_pool = response_pool or []
        
        # Pre-built off-topic responses for different domains
        self.off_topic_templates = {
            'food': [
                "The Mediterranean diet emphasizes olive oil, fish, and vegetables, and has been associated with numerous health benefits.",
                "When preparing pasta, always salt the water generously and cook until al dente for the best texture.",
                "Coffee originated in Ethiopia and has become one of the world's most popular beverages.",
            ],
            'technology': [
                "Cloud computing has revolutionized how businesses store and process data in recent years.",
                "Artificial intelligence systems are increasingly being used in healthcare diagnostics.",
                "The first smartphone was introduced in the early 1990s, combining phone and PDA features.",
            ],
            'nature': [
                "Coral reefs support approximately 25% of all marine species despite covering less than 1% of the ocean floor.",
                "Migration patterns of birds are influenced by changes in daylight hours and temperature.",
                "Rainforests produce about 20% of the world's oxygen and are often called the lungs of the Earth.",
            ],
            'history': [
                "The printing press, invented by Gutenberg in the 15th century, transformed the spread of knowledge.",
                "Ancient Rome's road system spanned over 50,000 miles and connected the vast empire.",
                "The Industrial Revolution began in Britain in the late 18th century and spread across Europe.",
            ],
            'science': [
                "Water molecules consist of two hydrogen atoms bonded to one oxygen atom.",
                "The human brain contains approximately 86 billion neurons that form complex networks.",
                "Photosynthesis allows plants to convert sunlight into chemical energy for growth.",
            ]
        }
    
    def inject(self, response: str, context: str, evidence: Any = None) -> List[ViolationExample]:
        """Create off-topic and same-domain-drift violations."""
        violations = []
        word_count = self._count_words(response)
        
        if word_count < 5:
            return violations
        
        # Off-topic substitution
        off_topic_resp = self._substitute_off_topic(response, context)
        if off_topic_resp:
            violations.append(ViolationExample(
                original_response=response,
                violated_response=off_topic_resp,
                violation_type="relation_off_topic",
                maxim="relation",
                context=context,
                evidence=evidence,
                metadata={
                    'substitution_type': 'off_topic',
                    'original_words': word_count,
                    'violated_words': self._count_words(off_topic_resp)
                }
            ))
        
        # Same-domain drift (if we can detect the domain)
        drift_resp = self._create_same_domain_drift(response, context)
        if drift_resp:
            violations.append(ViolationExample(
                original_response=response,
                violated_response=drift_resp,
                violation_type="relation_drift",
                maxim="relation", 
                context=context,
                evidence=evidence,
                metadata={
                    'substitution_type': 'same_domain_drift',
                    'original_words': word_count,
                    'violated_words': self._count_words(drift_resp)
                }
            ))
        
        return violations
    
    def _substitute_off_topic(self, response: str, context: str) -> Optional[str]:
        """Replace with a completely off-topic response of similar length."""
        target_length = self._count_words(response)
        
        # Detect domain to avoid (pick different domain)
        context_lower = context.lower()
        domains = list(self.off_topic_templates.keys())
        
        # Simple domain detection
        detected_domain = None
        if any(word in context_lower for word in ['eat', 'food', 'cook', 'recipe', 'meal']):
            detected_domain = 'food'
        elif any(word in context_lower for word in ['computer', 'phone', 'app', 'software', 'internet']):
            detected_domain = 'technology'
        elif any(word in context_lower for word in ['animal', 'plant', 'forest', 'ocean', 'nature']):
            detected_domain = 'nature'
        elif any(word in context_lower for word in ['war', 'king', 'ancient', 'century', 'empire']):
            detected_domain = 'history'
        
        # Pick different domain
        available_domains = [d for d in domains if d != detected_domain]
        if not available_domains:
            available_domains = domains
        
        chosen_domain = self.random.choice(available_domains)
        candidates = self.off_topic_templates[chosen_domain]
        
        # Find best length match
        best_match = min(candidates, key=lambda x: abs(self._count_words(x) - target_length))
        return best_match
    
    def _create_same_domain_drift(self, response: str, context: str) -> Optional[str]:
        """Stay on same topic but answer a different aspect."""
        # Extract likely topic from context
        context_words = context.lower().split()
        
        # Common drift patterns: "what is X" -> answer about history of X
        drift_templates = [
            "This topic has been extensively studied by researchers over the years.",
            "There are many interesting historical aspects to consider here.",
            "The origins of this subject date back many centuries.",
            "Popular opinions on this matter vary widely across different cultures.",
            "From a theoretical perspective, there are multiple frameworks to consider.",
        ]
        
        # Check response length and match
        target_length = self._count_words(response)
        drift_response = self.random.choice(drift_templates)
        
        # Only return if length is somewhat similar (within 50%)
        drift_length = self._count_words(drift_response)
        if 0.5 <= drift_length / target_length <= 2.0:
            return drift_response
        
        return None
    
    def set_response_pool(self, pool: List[Dict]):
        """Set pool of real responses for substitution (from other conversations)."""
        self.response_pool = pool


# ============================================================================
# MANNER VIOLATION INJECTOR
# ============================================================================

class MannerInjector(BaseViolationInjector):
    """
    Injects Manner violations: responses that are unclear or poorly organized.
    
    Ambiguous references: Replace nouns with unclear pronouns
    Sentence shuffling: Randomize sentence order
    Jargon injection: Replace common words with technical terms
    Run-on creation: Remove punctuation between sentences
    
    Key constraint: Must NOT change truth (Quality) or topic (Relation)
    """
    
    def __init__(self, random_seed: int = 42):
        super().__init__(random_seed)
        self.maxim_name = "manner"
        
        # Jargon replacements (common word -> technical term)
        self.jargon_map = {
            'water': 'H2O',
            'salt': 'sodium chloride',
            'sugar': 'sucrose',
            'air': 'atmospheric gases',
            'fire': 'combustion',
            'heat': 'thermal energy',
            'cold': 'low temperature',
            'light': 'electromagnetic radiation',
            'sound': 'acoustic waves',
            'speed': 'velocity',
            'big': 'substantial',
            'small': 'diminutive',
            'fast': 'rapid',
            'slow': 'gradual',
            'old': 'antiquated',
            'new': 'novel',
            'good': 'optimal',
            'bad': 'suboptimal',
            'important': 'of paramount significance',
            'easy': 'straightforward',
            'hard': 'challenging',
            'many': 'numerous',
            'few': 'a limited quantity of',
            'change': 'modification',
            'help': 'assistance',
            'use': 'utilize',
            'make': 'fabricate',
            'get': 'obtain',
            'give': 'provide',
            'take': 'acquire',
        }
        
        # Pronouns for ambiguous reference
        self.pronouns = {
            'subject': ['it', 'this', 'that', 'they', 'these'],
            'object': ['it', 'them', 'this', 'that'],
        }
    
    def inject(self, response: str, context: str, evidence: Any = None) -> List[ViolationExample]:
        """Create various manner violations."""
        violations = []
        sentences = self._extract_sentences(response)
        word_count = self._count_words(response)
        
        # Ambiguous references (needs substantial text)
        if word_count >= 15:
            ambig_resp = self._inject_ambiguous_references(response)
            if ambig_resp and ambig_resp != response:
                violations.append(ViolationExample(
                    original_response=response,
                    violated_response=ambig_resp,
                    violation_type="manner_ambiguous",
                    maxim="manner",
                    context=context,
                    evidence=evidence,
                    metadata={'transformation': 'ambiguous_references'}
                ))
        
        # Sentence shuffling (needs multiple sentences)
        if len(sentences) >= 3:
            shuffled_resp = self._shuffle_sentences(sentences)
            if shuffled_resp != response:
                violations.append(ViolationExample(
                    original_response=response,
                    violated_response=shuffled_resp,
                    violation_type="manner_shuffled",
                    maxim="manner",
                    context=context,
                    evidence=evidence,
                    metadata={'transformation': 'sentence_shuffle', 'num_sentences': len(sentences)}
                ))
        
        # Jargon injection
        jargon_resp = self._inject_jargon(response)
        if jargon_resp and jargon_resp != response:
            violations.append(ViolationExample(
                original_response=response,
                violated_response=jargon_resp,
                violation_type="manner_jargon",
                maxim="manner",
                context=context,
                evidence=evidence,
                metadata={'transformation': 'jargon_injection'}
            ))
        
        # Run-on sentences
        if len(sentences) >= 2:
            runon_resp = self._create_runon(sentences)
            violations.append(ViolationExample(
                original_response=response,
                violated_response=runon_resp,
                violation_type="manner_runon",
                maxim="manner",
                context=context,
                evidence=evidence,
                metadata={'transformation': 'run_on_sentences'}
            ))
        
        return violations
    
    def _inject_ambiguous_references(self, response: str) -> str:
        """Replace specific nouns with ambiguous pronouns."""
        words = response.split()
        result = []
        replacements_made = 0
        max_replacements = 3
        
        for i, word in enumerate(words):
            clean_word = word.lower().rstrip('.,!?;:')
            
            # Replace capitalized nouns (likely named entities) with pronouns
            if word[0].isupper() and i > 0 and len(clean_word) > 3:
                if replacements_made < max_replacements and self.random.random() < 0.5:
                    # Preserve punctuation
                    punct = word[len(clean_word):] if len(word) > len(clean_word) else ''
                    pronoun = self.random.choice(self.pronouns['subject'])
                    result.append(pronoun + punct)
                    replacements_made += 1
                    continue
            
            result.append(word)
        
        return ' '.join(result) if replacements_made > 0 else response
    
    def _shuffle_sentences(self, sentences: List[str]) -> str:
        """Randomly reorder sentences."""
        if len(sentences) < 2:
            return ' '.join(sentences)
        
        # Shuffle but ensure it's actually different
        shuffled = sentences.copy()
        attempts = 0
        while shuffled == sentences and attempts < 10:
            self.random.shuffle(shuffled)
            attempts += 1
        
        return ' '.join(shuffled)
    
    def _inject_jargon(self, response: str) -> str:
        """Replace common words with technical jargon."""
        result = response
        replacements_made = 0
        max_replacements = 3
        
        for common, technical in self.jargon_map.items():
            if replacements_made >= max_replacements:
                break
            
            # Case-insensitive replacement
            pattern = re.compile(r'\b' + common + r'\b', re.IGNORECASE)
            if pattern.search(result):
                result = pattern.sub(technical, result, count=1)
                replacements_made += 1
        
        return result
    
    def _create_runon(self, sentences: List[str]) -> str:
        """Join sentences without proper punctuation."""
        # Remove ending punctuation and join with just spaces
        cleaned = []
        for s in sentences:
            s = s.rstrip('.!?')
            cleaned.append(s)
        
        return ' '.join(cleaned)


# ============================================================================
# MULTI-MAXIM VIOLATION GENERATOR
# ============================================================================

class MultiMaximViolationGenerator:
    """
    Generates violations that affect multiple maxims simultaneously.
    
    In real model outputs, violations often co-occur. Training on
    multi-maxim examples helps the detector learn complex patterns.
    """
    
    def __init__(self, random_seed: int = 42):
        self.random = random.Random(random_seed)
        self.quantity_injector = QuantityInjector(random_seed)
        self.quality_injector = QualityInjector(random_seed)
        self.relation_injector = RelationInjector(random_seed=random_seed)
        self.manner_injector = MannerInjector(random_seed)
    
    def generate_multi_violation(
        self, 
        response: str, 
        context: str, 
        evidence: Any,
        maxims_to_violate: List[str]
    ) -> Optional[Dict]:
        """
        Create a response that violates multiple specified maxims.
        
        Args:
            response: Original clean response
            context: Conversation context
            evidence: Available evidence
            maxims_to_violate: List of maxims to violate (e.g., ['quantity', 'manner'])
            
        Returns:
            Dictionary with violated response and metadata
        """
        current_response = response
        violations_applied = []
        
        for maxim in maxims_to_violate:
            if maxim == 'quantity':
                # Apply over-informative (easier to combine with others)
                violations = self.quantity_injector.inject(current_response, context, evidence)
                over_violations = [v for v in violations if v.violation_type == 'quantity_over']
                if over_violations:
                    current_response = over_violations[0].violated_response
                    violations_applied.append('quantity_over')
                    
            elif maxim == 'quality':
                violations = self.quality_injector.inject(current_response, context, evidence)
                if violations:
                    current_response = violations[0].violated_response
                    violations_applied.append(violations[0].violation_type)
                    
            elif maxim == 'manner':
                violations = self.manner_injector.inject(current_response, context, evidence)
                if violations:
                    current_response = violations[0].violated_response
                    violations_applied.append(violations[0].violation_type)
        
        if len(violations_applied) < len(maxims_to_violate):
            return None  # Couldn't apply all requested violations
            
        return {
            'original_response': response,
            'violated_response': current_response,
            'violation_type': 'multi_' + '_'.join(maxims_to_violate),
            'maxim': 'multiple',
            'context': context,
            'evidence': evidence,
            'metadata': {
                'violations_applied': violations_applied,
                'maxims_violated': maxims_to_violate
            },
            'labels': {
                'quantity': 1 if 'quantity' in maxims_to_violate else 0,
                'quality': 1 if 'quality' in maxims_to_violate else 0,
                'relation': 1 if 'relation' in maxims_to_violate else 0,
                'manner': 1 if 'manner' in maxims_to_violate else 0
            }
        }


# ============================================================================
# MAIN VIOLATION INJECTION PIPELINE
# ============================================================================

class ViolationInjectionPipeline:
    """
    Main pipeline for generating the GriceBench dataset.
    
    Orchestrates all injectors to create a balanced dataset with:
    - Single-maxim violations (majority)
    - Multi-maxim violations (for robustness)
    - Clean examples (negative examples)
    """
    
    def __init__(self, random_seed: int = 42):
        self.random = random.Random(random_seed)
        
        # Initialize all injectors
        self.quantity_injector = QuantityInjector(random_seed)
        self.quality_injector = QualityInjector(random_seed)
        self.relation_injector = RelationInjector(random_seed=random_seed)
        self.manner_injector = MannerInjector(random_seed)
        self.multi_generator = MultiMaximViolationGenerator(random_seed)
        
        # Injector map for easy access
        self.injectors = {
            'quantity': self.quantity_injector,
            'quality': self.quality_injector,
            'relation': self.relation_injector,
            'manner': self.manner_injector
        }
    
    def process_example(self, example: Dict) -> List[Dict]:
        """
        Process a single example and generate all possible violations.
        
        Args:
            example: Training example with 'response', 'context_text', 'evidence'
            
        Returns:
            List of violation dictionaries
        """
        response = example.get('response', '')
        context = example.get('context_text', '')
        evidence = example.get('evidence')
        
        results = []
        
        # Generate single-maxim violations
        for maxim_name, injector in self.injectors.items():
            try:
                violations = injector.inject(response, context, evidence)
                for v in violations:
                    result = v.to_dict()
                    result['source_example_id'] = example.get('conversation_id', 'unknown')
                    result['source_turn'] = example.get('turn_index', -1)
                    results.append(result)
            except Exception as e:
                # Skip failed injections
                continue
        
        # Generate some multi-maxim violations (10% of the time)
        if self.random.random() < 0.1:
            combo = self.random.choice([
                ['quantity', 'manner'],
                ['quality', 'quantity'],
            ])
            multi = self.multi_generator.generate_multi_violation(
                response, context, evidence, combo
            )
            if multi:
                multi['source_example_id'] = example.get('conversation_id', 'unknown')
                multi['source_turn'] = example.get('turn_index', -1)
                results.append(multi)
        
        # Add clean example (no violations)
        clean = CleanExample(response=response, context=context, evidence=evidence)
        clean_dict = clean.to_dict()
        clean_dict['source_example_id'] = example.get('conversation_id', 'unknown')
        clean_dict['source_turn'] = example.get('turn_index', -1)
        results.append(clean_dict)
        
        return results
    
    def generate_dataset(
        self, 
        examples: List[Dict], 
        target_size: int = 50000,
        balance_maxims: bool = True
    ) -> List[Dict]:
        """
        Generate the full GriceBench dataset.
        
        Args:
            examples: List of clean training examples
            target_size: Target number of examples to generate
            balance_maxims: Whether to balance violation types
            
        Returns:
            List of violation examples ready for training
        """
        print(f"Generating GriceBench dataset from {len(examples)} source examples...")
        print(f"Target size: {target_size}")
        
        all_results = []
        
        for i, example in enumerate(examples):
            if i % 5000 == 0:
                print(f"  Processing example {i}/{len(examples)}...")
            
            try:
                violations = self.process_example(example)
                all_results.extend(violations)
            except Exception as e:
                continue
            
            # Stop if we have enough
            if len(all_results) >= target_size * 1.2:  # Generate extra for filtering
                break
        
        print(f"  Generated {len(all_results)} raw examples")
        
        # Balance if requested
        if balance_maxims:
            all_results = self._balance_dataset(all_results, target_size)
        
        # Shuffle
        self.random.shuffle(all_results)
        
        # Trim to target size
        all_results = all_results[:target_size]
        
        print(f"  Final dataset size: {len(all_results)}")
        
        # Print distribution
        self._print_distribution(all_results)
        
        return all_results
    
    def _balance_dataset(self, examples: List[Dict], target_size: int) -> List[Dict]:
        """Balance the dataset across violation types."""
        from collections import defaultdict
        
        # Group by violation type
        by_type = defaultdict(list)
        for ex in examples:
            vtype = ex.get('violation_type', 'unknown')
            by_type[vtype].append(ex)
        
        # Calculate per-type target
        num_types = len(by_type)
        per_type_target = target_size // num_types
        
        # Sample from each type
        balanced = []
        for vtype, type_examples in by_type.items():
            sample_size = min(len(type_examples), per_type_target)
            sampled = self.random.sample(type_examples, sample_size)
            balanced.extend(sampled)
        
        return balanced
    
    def _print_distribution(self, examples: List[Dict]):
        """Print the distribution of violation types."""
        from collections import Counter
        
        type_counts = Counter(ex.get('violation_type', 'unknown') for ex in examples)
        maxim_counts = Counter(ex.get('maxim', 'unknown') for ex in examples)
        
        print("\n  Violation Type Distribution:")
        for vtype, count in sorted(type_counts.items()):
            pct = count / len(examples) * 100
            print(f"    {vtype}: {count} ({pct:.1f}%)")
        
        print("\n  Maxim Distribution:")
        for maxim, count in sorted(maxim_counts.items()):
            pct = count / len(examples) * 100
            print(f"    {maxim}: {count} ({pct:.1f}%)")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Command-line interface for violation injection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate GriceBench violation dataset')
    parser.add_argument('--input', type=str, default='data_processed/train_examples.json',
                       help='Path to clean examples JSON')
    parser.add_argument('--output', type=str, default='data_processed/gricebench_weak_50k.json',
                       help='Path to output dataset')
    parser.add_argument('--target-size', type=int, default=50000,
                       help='Target number of examples')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load examples
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    output_path = project_root / args.output
    
    print(f"Loading examples from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")
    
    # Generate dataset
    pipeline = ViolationInjectionPipeline(random_seed=args.seed)
    dataset = pipeline.generate_dataset(examples, target_size=args.target_size)
    
    # Save
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print("Done!")


if __name__ == "__main__":
    main()
