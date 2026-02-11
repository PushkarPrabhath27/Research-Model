"""
GriceBench Weak Supervision Heuristics
======================================

This module implements rule-based heuristics for automatically labeling
examples for each Gricean maxim. These "weak" labels are noisy but enable
large-scale training without manual annotation.

Heuristics for each maxim:
1. QUANTITY - Length ratio, redundancy (self-BLEU), information density
2. QUALITY - NLI contradiction, evidence coverage, entity consistency
3. RELATION - Semantic similarity, keyword overlap
4. MANNER - Readability scores, ambiguous pronouns, sentence complexity

Based on Chapter 5 of the GriceBench Implementation Guide.
"""

import re
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter
from pathlib import Path

# Third-party imports
try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("Warning: textstat not available. Manner heuristics will be limited.")

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Some heuristics will be limited.")

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: nltk not available. Some heuristics will be limited.")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class HeuristicResult:
    """Result from a single heuristic evaluation."""
    heuristic_name: str
    maxim: str
    score: float  # 0.0 = no violation, 1.0 = strong violation signal
    confidence: float  # How confident the heuristic is (0.0 to 1.0)
    details: Dict[str, Any]  # Additional information about the decision

    def to_dict(self) -> Dict:
        return {
            'heuristic_name': self.heuristic_name,
            'maxim': self.maxim,
            'score': self.score,
            'confidence': self.confidence,
            'details': self.details
        }


@dataclass
class CombinedLabel:
    """Combined weak label from multiple heuristics."""
    maxim: str
    violation_probability: float  # 0.0 to 1.0
    heuristic_results: List[HeuristicResult]
    agreement_score: float  # How much heuristics agree

    def to_dict(self) -> Dict:
        return {
            'maxim': self.maxim,
            'violation_probability': self.violation_probability,
            'agreement_score': self.agreement_score,
            'heuristic_results': [h.to_dict() for h in self.heuristic_results]
        }


# ============================================================================
# BASE HEURISTIC CLASS
# ============================================================================

class BaseHeuristic:
    """Base class for all heuristic implementations."""
    
    def __init__(self):
        self.maxim_name = "base"
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_keywords(self, text: str, min_length: int = 4) -> List[str]:
        """Extract content keywords (non-stopwords)."""
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'and', 'or', 'but', 'if', 'then', 'than', 'when', 'where', 'what',
            'who', 'which', 'that', 'this', 'these', 'those', 'it', 'its',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'from',
            'not', 'no', 'yes', 'so', 'as', 'like', 'just', 'very', 'really',
            'i', 'you', 'we', 'they', 'he', 'she', 'me', 'us', 'them', 'my', 'your'
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) >= min_length]
        return keywords
    
    def _extract_pronouns(self, text: str) -> List[str]:
        """Extract pronouns from text."""
        pronouns = ['it', 'they', 'them', 'this', 'that', 'these', 'those', 
                    'he', 'she', 'him', 'her', 'his', 'hers', 'its', 'their']
        words = text.lower().split()
        return [w for w in words if w.rstrip('.,!?;:') in pronouns]


# ============================================================================
# QUANTITY HEURISTICS
# ============================================================================

class QuantityHeuristics(BaseHeuristic):
    """
    Heuristics for detecting Quantity violations (too little or too much).
    
    Methods:
    - Length ratio: Compare actual length to expected length
    - Redundancy (self-BLEU): Detect repetition within response
    - Information density: Unique content per sentence
    """
    
    def __init__(self, 
                 short_threshold: float = 0.3,
                 long_threshold: float = 3.0,
                 redundancy_threshold: float = 0.5):
        super().__init__()
        self.maxim_name = "quantity"
        self.short_threshold = short_threshold  # Below 30% of expected = too short
        self.long_threshold = long_threshold    # Above 300% of expected = too long
        self.redundancy_threshold = redundancy_threshold
    
    def evaluate(self, response: str, context: str, evidence: Any = None) -> List[HeuristicResult]:
        """Run all Quantity heuristics."""
        results = []
        
        # Length ratio heuristic
        length_result = self._length_ratio_heuristic(response, context)
        results.append(length_result)
        
        # Redundancy heuristic (only for longer responses)
        if self._count_words(response) >= 20:
            redundancy_result = self._redundancy_heuristic(response)
            results.append(redundancy_result)
        
        # Information density heuristic
        density_result = self._info_density_heuristic(response)
        results.append(density_result)
        
        return results
    
    def _length_ratio_heuristic(self, response: str, context: str) -> HeuristicResult:
        """
        Compare actual response length to expected length based on question complexity.
        
        Question complexity factors:
        - Question words (who, what, when, where, why, how)
        - Multi-part questions (contains 'and' or multiple '?')
        - Named entities mentioned
        """
        response_length = self._count_words(response)
        
        # Estimate expected length from question complexity
        question_words = len(re.findall(r'\b(who|what|when|where|why|how)\b', context.lower()))
        has_multiple_parts = '?' in context[:-1] or ' and ' in context.lower()
        
        # Base expected length
        if question_words == 0:
            expected_length = 10  # Simple statement/acknowledgment expected
        elif question_words == 1:
            expected_length = 20  # Single question
        else:
            expected_length = 30  # Complex question
        
        if has_multiple_parts:
            expected_length *= 1.5
        
        # Calculate ratio
        ratio = response_length / expected_length if expected_length > 0 else 1.0
        
        # Determine violation
        if ratio < self.short_threshold:
            score = min(1.0, (self.short_threshold - ratio) / self.short_threshold)
            violation_type = "too_short"
        elif ratio > self.long_threshold:
            score = min(1.0, (ratio - self.long_threshold) / self.long_threshold)
            violation_type = "too_long"
        else:
            score = 0.0
            violation_type = "appropriate"
        
        return HeuristicResult(
            heuristic_name="length_ratio",
            maxim="quantity",
            score=score,
            confidence=0.7,  # Length is a rough proxy
            details={
                'response_length': response_length,
                'expected_length': expected_length,
                'ratio': ratio,
                'violation_type': violation_type
            }
        )
    
    def _redundancy_heuristic(self, response: str) -> HeuristicResult:
        """
        Detect repetition within the response using simple n-gram overlap.
        (Simplified self-BLEU approach)
        """
        sentences = self._extract_sentences(response)
        
        if len(sentences) < 2:
            return HeuristicResult(
                heuristic_name="redundancy",
                maxim="quantity",
                score=0.0,
                confidence=0.3,
                details={'reason': 'too_few_sentences'}
            )
        
        # Calculate n-gram overlap between sentences
        def get_ngrams(text: str, n: int = 3) -> set:
            words = text.lower().split()
            return set(tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1)))
        
        overlaps = []
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences):
                if i < j:
                    ngrams1 = get_ngrams(sent1)
                    ngrams2 = get_ngrams(sent2)
                    if ngrams1 and ngrams2:
                        overlap = len(ngrams1 & ngrams2) / min(len(ngrams1), len(ngrams2))
                        overlaps.append(overlap)
        
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        
        # High overlap -> redundancy violation
        score = min(1.0, avg_overlap / self.redundancy_threshold) if avg_overlap > 0.2 else 0.0
        
        return HeuristicResult(
            heuristic_name="redundancy",
            maxim="quantity",
            score=score,
            confidence=0.6,
            details={
                'avg_ngram_overlap': avg_overlap,
                'num_sentences': len(sentences)
            }
        )
    
    def _info_density_heuristic(self, response: str) -> HeuristicResult:
        """
        Measure information density: unique content keywords per sentence.
        Low density suggests padding without substance.
        """
        sentences = self._extract_sentences(response)
        keywords = self._extract_keywords(response)
        
        if not sentences:
            return HeuristicResult(
                heuristic_name="info_density",
                maxim="quantity",
                score=0.0,
                confidence=0.3,
                details={'reason': 'empty_response'}
            )
        
        unique_keywords = set(keywords)
        density = len(unique_keywords) / len(sentences)
        
        # Very low density indicates potential over-informative padding
        # (Many sentences but few unique concepts)
        if density < 2.0 and len(sentences) > 2:
            score = min(1.0, (2.0 - density) / 2.0)
        else:
            score = 0.0
        
        return HeuristicResult(
            heuristic_name="info_density",
            maxim="quantity",
            score=score,
            confidence=0.5,
            details={
                'unique_keywords': len(unique_keywords),
                'num_sentences': len(sentences),
                'density': density
            }
        )


# ============================================================================
# QUALITY HEURISTICS
# ============================================================================

class QualityHeuristics(BaseHeuristic):
    """
    Heuristics for detecting Quality violations (unsupported/false claims).
    
    Methods:
    - Evidence coverage: Semantic similarity between response and evidence
    - Entity consistency: Check if response entities appear in evidence
    - NLI contradiction: Use sentence embeddings for contradiction detection
    """
    
    def __init__(self, similarity_threshold: float = 0.3):
        super().__init__()
        self.maxim_name = "quality"
        self.similarity_threshold = similarity_threshold
        
        # Load sentence transformer for semantic similarity
        self._model = None
    
    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use a small, fast model
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model
    
    def evaluate(self, response: str, context: str, evidence: Any = None) -> List[HeuristicResult]:
        """Run all Quality heuristics."""
        results = []
        
        # Evidence coverage (if evidence available)
        if evidence:
            coverage_result = self._evidence_coverage_heuristic(response, evidence)
            results.append(coverage_result)
        
        # Entity consistency
        if evidence:
            entity_result = self._entity_consistency_heuristic(response, evidence)
            results.append(entity_result)
        
        # If no evidence, use simpler heuristic
        if not evidence:
            results.append(HeuristicResult(
                heuristic_name="no_evidence",
                maxim="quality",
                score=0.0,
                confidence=0.2,
                details={'reason': 'no_evidence_provided'}
            ))
        
        return results
    
    def _evidence_coverage_heuristic(self, response: str, evidence: Any) -> HeuristicResult:
        """
        Check how well response claims are supported by evidence.
        Low semantic similarity suggests unsupported claims.
        """
        model = self._get_model()
        
        # Convert evidence to string
        if isinstance(evidence, dict):
            evidence_text = ' '.join(str(v) for v in evidence.values())
        elif isinstance(evidence, list):
            evidence_text = ' '.join(str(e) for e in evidence)
        else:
            evidence_text = str(evidence)
        
        if not model:
            # Fallback: simple keyword overlap
            response_keywords = set(self._extract_keywords(response))
            evidence_keywords = set(self._extract_keywords(evidence_text))
            
            if response_keywords and evidence_keywords:
                overlap = len(response_keywords & evidence_keywords) / len(response_keywords)
            else:
                overlap = 0.5
            
            return HeuristicResult(
                heuristic_name="evidence_coverage",
                maxim="quality",
                score=1.0 - overlap if overlap < self.similarity_threshold else 0.0,
                confidence=0.4,
                details={'method': 'keyword_overlap', 'overlap': overlap}
            )
        
        # Use semantic similarity
        try:
            response_embedding = model.encode(response, convert_to_tensor=True)
            evidence_embedding = model.encode(evidence_text, convert_to_tensor=True)
            similarity = float(util.pytorch_cos_sim(response_embedding, evidence_embedding)[0][0])
        except Exception as e:
            similarity = 0.5  # Neutral on error
        
        # Low similarity -> possible Quality violation
        if similarity < self.similarity_threshold:
            score = min(1.0, (self.similarity_threshold - similarity) / self.similarity_threshold)
        else:
            score = 0.0
        
        return HeuristicResult(
            heuristic_name="evidence_coverage",
            maxim="quality",
            score=score,
            confidence=0.7,
            details={
                'method': 'semantic_similarity',
                'similarity': similarity
            }
        )
    
    def _entity_consistency_heuristic(self, response: str, evidence: Any) -> HeuristicResult:
        """
        Check if named entities in response appear in evidence.
        New entities not in evidence may indicate hallucination.
        """
        # Simple entity extraction (capitalized words)
        def extract_entities(text: str) -> set:
            words = text.split()
            entities = set()
            for word in words:
                clean = re.sub(r'[^\w]', '', word)
                if clean and clean[0].isupper() and len(clean) > 2:
                    entities.add(clean.lower())
            return entities
        
        # Convert evidence to string
        if isinstance(evidence, dict):
            evidence_text = ' '.join(str(v) for v in evidence.values())
        elif isinstance(evidence, list):
            evidence_text = ' '.join(str(e) for e in evidence)
        else:
            evidence_text = str(evidence)
        
        response_entities = extract_entities(response)
        evidence_entities = extract_entities(evidence_text)
        
        if not response_entities:
            return HeuristicResult(
                heuristic_name="entity_consistency",
                maxim="quality",
                score=0.0,
                confidence=0.3,
                details={'reason': 'no_entities_in_response'}
            )
        
        # Check how many response entities are NOT in evidence
        novel_entities = response_entities - evidence_entities
        novel_ratio = len(novel_entities) / len(response_entities)
        
        # High ratio of novel entities -> potential hallucination
        score = min(1.0, novel_ratio) if novel_ratio > 0.5 else 0.0
        
        return HeuristicResult(
            heuristic_name="entity_consistency",
            maxim="quality",
            score=score,
            confidence=0.5,
            details={
                'response_entities': list(response_entities),
                'novel_entities': list(novel_entities),
                'novel_ratio': novel_ratio
            }
        )


# ============================================================================
# RELATION HEURISTICS
# ============================================================================

class RelationHeuristics(BaseHeuristic):
    """
    Heuristics for detecting Relation violations (off-topic responses).
    
    Methods:
    - Semantic similarity: Embedding similarity between question and response
    - Keyword overlap: Check if question keywords appear in response
    """
    
    def __init__(self, similarity_threshold: float = 0.2):
        super().__init__()
        self.maxim_name = "relation"
        self.similarity_threshold = similarity_threshold
        self._model = None
    
    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model
    
    def evaluate(self, response: str, context: str, evidence: Any = None) -> List[HeuristicResult]:
        """Run all Relation heuristics."""
        results = []
        
        # Semantic similarity
        sim_result = self._semantic_similarity_heuristic(response, context)
        results.append(sim_result)
        
        # Keyword overlap
        keyword_result = self._keyword_overlap_heuristic(response, context)
        results.append(keyword_result)
        
        return results
    
    def _semantic_similarity_heuristic(self, response: str, context: str) -> HeuristicResult:
        """
        Compute semantic similarity between question and response.
        Very low similarity suggests off-topic response.
        """
        model = self._get_model()
        
        if not model:
            # Fallback: simple overlap
            return self._keyword_overlap_heuristic(response, context)
        
        try:
            response_embedding = model.encode(response, convert_to_tensor=True)
            context_embedding = model.encode(context, convert_to_tensor=True)
            similarity = float(util.pytorch_cos_sim(response_embedding, context_embedding)[0][0])
        except Exception:
            similarity = 0.5
        
        # Low similarity -> Relation violation
        if similarity < self.similarity_threshold:
            score = min(1.0, (self.similarity_threshold - similarity) / self.similarity_threshold)
        else:
            score = 0.0
        
        return HeuristicResult(
            heuristic_name="semantic_similarity",
            maxim="relation",
            score=score,
            confidence=0.8,
            details={'similarity': similarity}
        )
    
    def _keyword_overlap_heuristic(self, response: str, context: str) -> HeuristicResult:
        """
        Check if question keywords appear in response.
        Complete absence suggests off-topic.
        """
        context_keywords = set(self._extract_keywords(context))
        response_keywords = set(self._extract_keywords(response))
        
        if not context_keywords:
            return HeuristicResult(
                heuristic_name="keyword_overlap",
                maxim="relation",
                score=0.0,
                confidence=0.3,
                details={'reason': 'no_context_keywords'}
            )
        
        overlap = len(context_keywords & response_keywords)
        overlap_ratio = overlap / len(context_keywords)
        
        # No overlap -> likely off-topic
        if overlap_ratio == 0:
            score = 0.8
        elif overlap_ratio < 0.2:
            score = 0.4
        else:
            score = 0.0
        
        return HeuristicResult(
            heuristic_name="keyword_overlap",
            maxim="relation",
            score=score,
            confidence=0.6,
            details={
                'context_keywords': list(context_keywords),
                'overlap_count': overlap,
                'overlap_ratio': overlap_ratio
            }
        )


# ============================================================================
# MANNER HEURISTICS
# ============================================================================

class MannerHeuristics(BaseHeuristic):
    """
    Heuristics for detecting Manner violations (unclear/disorganized responses).
    
    Methods:
    - Readability score: Flesch-Kincaid grade level
    - Ambiguous pronouns: Pronouns without clear antecedents
    - Sentence complexity: Average sentence length
    """
    
    def __init__(self, 
                 readability_threshold: float = 16.0,  # Grade 16+ is very complex
                 pronoun_ratio_threshold: float = 0.15):
        super().__init__()
        self.maxim_name = "manner"
        self.readability_threshold = readability_threshold
        self.pronoun_ratio_threshold = pronoun_ratio_threshold
    
    def evaluate(self, response: str, context: str, evidence: Any = None) -> List[HeuristicResult]:
        """Run all Manner heuristics."""
        results = []
        
        # Readability (if textstat available)
        if TEXTSTAT_AVAILABLE and len(response.split()) >= 10:
            readability_result = self._readability_heuristic(response)
            results.append(readability_result)
        
        # Ambiguous pronouns
        pronoun_result = self._ambiguous_pronoun_heuristic(response)
        results.append(pronoun_result)
        
        # Sentence complexity
        complexity_result = self._sentence_complexity_heuristic(response)
        results.append(complexity_result)
        
        return results
    
    def _readability_heuristic(self, response: str) -> HeuristicResult:
        """
        Compute Flesch-Kincaid grade level.
        Very high grade level suggests overly complex writing.
        """
        try:
            grade_level = textstat.flesch_kincaid_grade(response)
        except Exception:
            grade_level = 10.0  # Neutral
        
        # Very high grade level -> Manner violation
        if grade_level > self.readability_threshold:
            score = min(1.0, (grade_level - self.readability_threshold) / 4.0)
        else:
            score = 0.0
        
        return HeuristicResult(
            heuristic_name="readability",
            maxim="manner",
            score=score,
            confidence=0.6,
            details={
                'flesch_kincaid_grade': grade_level,
                'threshold': self.readability_threshold
            }
        )
    
    def _ambiguous_pronoun_heuristic(self, response: str) -> HeuristicResult:
        """
        Count pronouns and estimate how many lack clear antecedents.
        High pronoun density at sentence starts suggests ambiguity.
        """
        sentences = self._extract_sentences(response)
        total_words = self._count_words(response)
        
        if total_words < 5:
            return HeuristicResult(
                heuristic_name="ambiguous_pronouns",
                maxim="manner",
                score=0.0,
                confidence=0.3,
                details={'reason': 'too_short'}
            )
        
        # Count pronouns at sentence starts (often ambiguous)
        sentence_start_pronouns = 0
        ambiguous_pronouns = ['it', 'this', 'that', 'they', 'these', 'those']
        
        for sent in sentences:
            first_word = sent.split()[0].lower().rstrip('.,') if sent.split() else ''
            if first_word in ambiguous_pronouns:
                sentence_start_pronouns += 1
        
        all_pronouns = self._extract_pronouns(response)
        pronoun_ratio = len(all_pronouns) / total_words
        
        # High pronoun ratio + sentence-start pronouns = likely ambiguity
        if pronoun_ratio > self.pronoun_ratio_threshold and sentence_start_pronouns > 1:
            score = min(1.0, pronoun_ratio / 0.3)
        elif sentence_start_pronouns >= len(sentences) * 0.5 and len(sentences) >= 2:
            score = 0.6
        else:
            score = 0.0
        
        return HeuristicResult(
            heuristic_name="ambiguous_pronouns",
            maxim="manner",
            score=score,
            confidence=0.5,
            details={
                'total_pronouns': len(all_pronouns),
                'pronoun_ratio': pronoun_ratio,
                'sentence_start_pronouns': sentence_start_pronouns
            }
        )
    
    def _sentence_complexity_heuristic(self, response: str) -> HeuristicResult:
        """
        Measure sentence length and complexity.
        Very long sentences without punctuation indicate run-ons.
        """
        sentences = self._extract_sentences(response)
        
        if not sentences:
            return HeuristicResult(
                heuristic_name="sentence_complexity",
                maxim="manner",
                score=0.0,
                confidence=0.3,
                details={'reason': 'no_sentences'}
            )
        
        # Calculate average sentence length
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        max_length = max(sentence_lengths)
        
        # Very long sentences or high variance indicates complexity issues
        if max_length > 50:
            score = 0.8  # Very long sentence = likely run-on
        elif avg_length > 35:
            score = 0.5
        elif avg_length > 25 and max_length > 40:
            score = 0.4
        else:
            score = 0.0
        
        return HeuristicResult(
            heuristic_name="sentence_complexity",
            maxim="manner",
            score=score,
            confidence=0.6,
            details={
                'avg_sentence_length': avg_length,
                'max_sentence_length': max_length,
                'num_sentences': len(sentences)
            }
        )


# ============================================================================
# COMBINED HEURISTIC LABELER
# ============================================================================

class WeakSupervisionLabeler:
    """
    Combines all heuristics to produce weak labels for each maxim.
    
    Uses voting/averaging to combine multiple heuristics per maxim.
    """
    
    def __init__(self, 
                 voting_threshold: float = 0.5,
                 confidence_filter: float = 0.3):
        self.quantity = QuantityHeuristics()
        self.quality = QualityHeuristics()
        self.relation = RelationHeuristics()
        self.manner = MannerHeuristics()
        
        self.voting_threshold = voting_threshold
        self.confidence_filter = confidence_filter
        
        self.heuristics_by_maxim = {
            'quantity': self.quantity,
            'quality': self.quality,
            'relation': self.relation,
            'manner': self.manner
        }
    
    def label_example(self, response: str, context: str, evidence: Any = None) -> Dict[str, CombinedLabel]:
        """
        Apply all heuristics and produce combined labels for each maxim.
        
        Returns:
            Dictionary mapping maxim name to CombinedLabel
        """
        combined_labels = {}
        
        for maxim_name, heuristic in self.heuristics_by_maxim.items():
            results = heuristic.evaluate(response, context, evidence)
            
            # Filter by confidence
            confident_results = [r for r in results if r.confidence >= self.confidence_filter]
            
            if not confident_results:
                # No confident heuristics - use all with low confidence
                confident_results = results
            
            # Combine scores using weighted average by confidence
            if confident_results:
                total_weight = sum(r.confidence for r in confident_results)
                weighted_score = sum(r.score * r.confidence for r in confident_results) / total_weight
                
                # Calculate agreement (how similar are the scores)
                scores = [r.score for r in confident_results]
                if len(scores) > 1:
                    score_variance = sum((s - weighted_score)**2 for s in scores) / len(scores)
                    agreement = 1.0 - min(1.0, score_variance * 4)  # Scale variance to agreement
                else:
                    agreement = 1.0
            else:
                weighted_score = 0.0
                agreement = 0.0
            
            combined_labels[maxim_name] = CombinedLabel(
                maxim=maxim_name,
                violation_probability=weighted_score,
                heuristic_results=confident_results,
                agreement_score=agreement
            )
        
        return combined_labels
    
    def label_dataset(self, examples: List[Dict], verbose: bool = True) -> List[Dict]:
        """
        Apply weak supervision to an entire dataset.
        
        Args:
            examples: List of examples with 'response', 'context_text', 'evidence'
            
        Returns:
            List of examples with added 'weak_labels' field
        """
        labeled = []
        
        for i, example in enumerate(examples):
            if verbose and i % 500 == 0:
                print(f"  Labeling example {i}/{len(examples)}...")
            
            try:
                response = example.get('violated_response', example.get('response', ''))
                context = example.get('context', '')
                evidence = example.get('evidence')
                
                labels = self.label_example(response, context, evidence)
                
                # Add weak labels to example
                example_with_labels = example.copy()
                example_with_labels['weak_labels'] = {
                    maxim: label.to_dict() for maxim, label in labels.items()
                }
                
                labeled.append(example_with_labels)
                
            except Exception as e:
                # Skip failed examples
                continue
        
        return labeled


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Apply weak supervision to the generated dataset."""
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply weak supervision heuristics')
    parser.add_argument('--input', type=str, default='data_processed/gricebench_weak_50k.json')
    parser.add_argument('--output', type=str, default='data_processed/gricebench_weak_labeled.json')
    parser.add_argument('--limit', type=int, default=None, help='Limit examples to process')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    output_path = project_root / args.output
    
    print(f"Loading dataset from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    if args.limit:
        examples = examples[:args.limit]
    
    print(f"Loaded {len(examples)} examples")
    
    # Apply weak supervision
    print("\nApplying weak supervision heuristics...")
    labeler = WeakSupervisionLabeler()
    labeled_examples = labeler.label_dataset(examples)
    
    print(f"\nLabeled {len(labeled_examples)} examples")
    
    # Save
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labeled_examples, f, indent=2, ensure_ascii=False)
    
    # Print sample
    if labeled_examples:
        print("\nSample weak labels:")
        sample = labeled_examples[0]
        for maxim, label_data in sample.get('weak_labels', {}).items():
            prob = label_data.get('violation_probability', 0)
            print(f"  {maxim}: {prob:.3f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
