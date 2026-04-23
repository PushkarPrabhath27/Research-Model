"""
Realistic Violation Injectors for GriceBench
=============================================

Implements Fix 6 from morechanges.md (lines 1298-1417):
- Realistic violation patterns that match real-world violations
- NOT random shuffling or obvious contradictions
- Patterns that will help detector generalize to natural violations

Author: GriceBench Team
Date: 2026-01-27
"""

import re
import random
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


# =============================================================================
# QUANTITY INJECTORS - Real verbosity patterns
# =============================================================================

class QuantityInjector:
    """
    Inject realistic quantity violations:
    - Repetition (same point multiple ways)
    - Tangential anecdotes
    - Excessive examples
    - Unnecessary detail
    """
    
    def __init__(self):
        self.tangent_templates = [
            "This reminds me of a time when {anecdote}. Anyway, {original}",
            "Speaking of which, {anecdote}. But getting back to your question, {original}",
            "I remember {anecdote}. That's somewhat related to this. So, {original}",
            "You know, my friend once {anecdote}, which is kind of similar. In any case, {original}",
        ]
        
        self.example_templates = [
            "For example, {example1}. Another example would be {example2}. Yet another case is {example3}. And there's also {example4}.",
            "Consider {example1}. Or think about {example2}. Don't forget {example3}. Plus there's {example4}.",
            "Like {example1}. Or {example2}. Same with {example3}. And {example4} too.",
        ]
        
        self.repetition_bridges = [
            "In other words,",
            "To put it differently,",
            "What I mean is,",
            "Essentially,",
            "To rephrase that,",
            "Said another way,",
        ]
        
        self.anecdotes = [
            "I had a similar experience last year",
            "my neighbor dealt with something like this",
            "I read an article about this exact topic",
            "someone I know went through this",
            "I was just thinking about this yesterday",
            "there was this one time at work when something similar happened",
        ]
        
        self.filler_phrases = [
            "It's worth mentioning that",
            "One thing worth noting is that",
            "It should be pointed out that",
            "Additionally, it's important to understand that",
            "Moreover, we should consider that",
            "Furthermore, let me add that",
        ]
    
    def inject_repetition(self, response: str) -> str:
        """
        Strategy 1: Repeat the same point 3 different ways
        Real verbose people do this - they say the same thing multiple times
        """
        sentences = sent_tokenize(response)
        if len(sentences) < 1:
            return response
        
        key_sentence = sentences[0]
        
        # Create paraphrases (simple word substitutions for now)
        paraphrase1 = f"{random.choice(self.repetition_bridges)} {key_sentence.lower()}"
        paraphrase2 = f"{random.choice(self.repetition_bridges)} {self._simple_rephrase(key_sentence)}"
        
        # Insert repetitions
        verbose = f"{key_sentence} {paraphrase1} {paraphrase2}"
        if len(sentences) > 1:
            verbose += " " + " ".join(sentences[1:])
        
        return verbose
    
    def inject_tangent(self, response: str) -> str:
        """
        Strategy 2: Add tangential personal anecdote
        Real verbose people often go off on loosely-related stories
        """
        anecdote = random.choice(self.anecdotes)
        template = random.choice(self.tangent_templates)
        
        return template.format(anecdote=anecdote, original=response)
    
    def inject_excessive_examples(self, response: str) -> str:
        """
        Strategy 3: Add way too many examples
        Real verbose people pile on examples when one would suffice
        """
        sentences = sent_tokenize(response)
        
        # Generate topic-neutral examples
        examples = [
            "the case of everyday situations",
            "what happens in typical scenarios",
            "the common experience people have",
            "situations that occur regularly",
        ]
        
        example_block = random.choice(self.example_templates).format(
            example1=examples[0],
            example2=examples[1],
            example3=examples[2],
            example4=examples[3]
        )
        
        # Insert examples after first sentence
        if len(sentences) > 1:
            return f"{sentences[0]} {example_block} {' '.join(sentences[1:])}"
        return f"{response} {example_block}"
    
    def inject_unnecessary_detail(self, response: str) -> str:
        """
        Strategy 4: Add filler phrases and unnecessary qualifications
        """
        sentences = sent_tokenize(response)
        result = []
        
        for i, sent in enumerate(sentences):
            if i > 0 and random.random() < 0.5:
                filler = random.choice(self.filler_phrases)
                result.append(f"{filler} {sent.lower()}")
            else:
                result.append(sent)
        
        return " ".join(result)
    
    def inject_vague_nonanswer(self, response: str, context: str = "") -> str:
        """
        Strategy 5: Vague non-answer (from morechanges.md)
        Provides lots of words but no actual information
        """
        vague_intros = [
            "Well, it really depends on a lot of factors. There are many things to consider here.",
            "That's a complex question with no simple answer. There are various perspectives.",
            "It's hard to say definitively. There are pros and cons to consider.",
            "This varies quite a bit depending on circumstances. Generally speaking though,",
        ]
        
        vague_outros = [
            "Of course, this is just one way to look at it, and there are other viewpoints as well.",
            "But then again, it could also be different in other situations.",
            "Though ultimately, it's really up to individual interpretation.",
        ]
        
        intro = random.choice(vague_intros)
        outro = random.choice(vague_outros)
        
        return f"{intro} {response} {outro}"
    
    def _simple_rephrase(self, sentence: str) -> str:
        """Simple rephrasing by substituting common words"""
        synonyms = {
            'is': 'represents',
            'are': 'constitute',
            'can': 'is able to',
            'will': 'is going to',
            'has': 'possesses',
            'have': 'possess',
            'do': 'perform',
            'make': 'create',
            'get': 'obtain',
            'give': 'provide',
        }
        
        words = sentence.split()
        result = []
        for word in words:
            lower = word.lower()
            if lower in synonyms and random.random() < 0.3:
                result.append(synonyms[lower])
            else:
                result.append(word)
        
        return " ".join(result)
    
    def inject(self, response: str, strategy: str = "random") -> Tuple[str, str]:
        """
        Apply a quantity violation injection
        Returns: (violated_response, strategy_used)
        """
        strategies = {
            'repetition': self.inject_repetition,
            'tangent': self.inject_tangent,
            'examples': self.inject_excessive_examples,
            'detail': self.inject_unnecessary_detail,
            'vague': self.inject_vague_nonanswer,
        }
        
        if strategy == "random":
            strategy = random.choice(list(strategies.keys()))
        
        violated = strategies[strategy](response)
        return violated, f"quantity_{strategy}"


# =============================================================================
# QUALITY INJECTORS - Subtle unsupported claims
# =============================================================================

class QualityInjector:
    """
    Inject realistic quality violations:
    - Weasel words ("Some say", "Studies suggest")
    - Unsupported statistics
    - Overgeneralizations
    - Hidden qualifications
    """
    
    def __init__(self):
        self.weasel_phrases = [
            "Some experts believe that",
            "Many people say that",
            "Studies have shown that",
            "Research suggests that",
            "It's widely known that",
            "According to some sources,",
            "There's evidence that",
            "Some would argue that",
            "It's been said that",
            "Reportedly,",
        ]
        
        self.fake_stats = [
            "approximately {pct}% of people",
            "nearly {pct}% of cases",
            "about {pct} out of 10",
            "studies show {pct}% success rate",
            "an estimated {pct}% of the time",
        ]
        
        self.overgeneralization_templates = [
            "Everyone knows that {claim}.",
            "It's always the case that {claim}.",
            "Without exception, {claim}.",
            "In all situations, {claim}.",
            "People always {claim}.",
        ]
        
        self.hedge_words = [
            "probably", "possibly", "might", "could be", "perhaps",
            "seemingly", "apparently", "supposedly", "allegedly",
        ]
    
    def inject_weasel_words(self, response: str) -> str:
        """
        Strategy 1: Add weasel word claims without sources
        Real misinformation often uses appeal to vague authority
        """
        sentences = sent_tokenize(response)
        
        # Insert weasel claim after first sentence
        weasel_claim = f"{random.choice(self.weasel_phrases)} this is particularly important to consider."
        
        if len(sentences) > 0:
            return f"{sentences[0]} {weasel_claim} {' '.join(sentences[1:])}"
        return f"{weasel_claim} {response}"
    
    def inject_unsupported_stat(self, response: str) -> str:
        """
        Strategy 2: Add fake statistics without sources
        Real misinformation often cites made-up numbers
        """
        pct = random.randint(50, 95)
        stat_template = random.choice(self.fake_stats).format(pct=pct)
        
        claim = f"Interestingly, {stat_template} experience this."
        
        sentences = sent_tokenize(response)
        if len(sentences) > 1:
            insert_pos = min(1, len(sentences) - 1)
            sentences.insert(insert_pos, claim)
            return " ".join(sentences)
        
        return f"{response} {claim}"
    
    def inject_overgeneralization(self, response: str) -> str:
        """
        Strategy 3: Turn specific claims into overgeneralizations
        "This sometimes works" → "This always works"
        """
        # Extract a claim from the response
        sentences = sent_tokenize(response)
        if not sentences:
            return response
        
        # Add overgeneralization
        overgeneralization = random.choice([
            "This is universally true in all cases.",
            "Everyone experiences this the same way.",
            "There are no exceptions to this.",
            "This applies to absolutely everyone.",
        ])
        
        return f"{response} {overgeneralization}"
    
    def inject_false_certainty(self, response: str) -> str:
        """
        Strategy 4: Remove appropriate hedging, add false certainty
        Real misinformation is often overconfident
        """
        # Remove hedging words and add certainty
        for hedge in self.hedge_words:
            response = response.replace(hedge, "definitely")
            response = response.replace(hedge.capitalize(), "Definitely")
        
        # Add certainty claim
        certainty_claims = [
            " This is absolutely certain.",
            " There's no doubt about this.",
            " This is a proven fact.",
            " This is definitely true.",
        ]
        
        return response + random.choice(certainty_claims)
    
    def inject_appeal_to_authority(self, response: str) -> str:
        """
        Strategy 5: Appeal to vague/false authority
        """
        fake_authorities = [
            "According to leading scientists,",
            "Top researchers have confirmed that",
            "Experts unanimously agree that",
            "Harvard studies have proven that",
            "Nobel laureates have stated that",
        ]
        
        return f"{random.choice(fake_authorities)} {response.lower()}"
    
    def inject(self, response: str, strategy: str = "random") -> Tuple[str, str]:
        """
        Apply a quality violation injection
        Returns: (violated_response, strategy_used)
        """
        strategies = {
            'weasel': self.inject_weasel_words,
            'stat': self.inject_unsupported_stat,
            'overgeneralize': self.inject_overgeneralization,
            'certainty': self.inject_false_certainty,
            'authority': self.inject_appeal_to_authority,
        }
        
        if strategy == "random":
            strategy = random.choice(list(strategies.keys()))
        
        violated = strategies[strategy](response)
        return violated, f"quality_{strategy}"


# =============================================================================
# MANNER INJECTORS - Real clarity issues
# =============================================================================

class MannerInjector:
    """
    Inject realistic manner violations:
    - Ambiguous pronouns
    - Passive voice (hiding agency)
    - Buried lede
    - Unnecessary jargon
    - Run-on sentences
    - Disorganized structure
    """
    
    def __init__(self):
        self.jargon_map = {
            'use': 'utilize',
            'help': 'facilitate',
            'start': 'initiate',
            'end': 'terminate',
            'show': 'demonstrate',
            'make': 'fabricate',
            'give': 'furnish',
            'get': 'acquire',
            'need': 'necessitate',
            'try': 'endeavor',
            'about': 'regarding',
            'buy': 'procure',
            'build': 'construct',
            'change': 'modify',
            'think': 'cogitate',
            'understand': 'comprehend',
            'important': 'paramount',
            'big': 'substantial',
            'small': 'diminutive',
            'good': 'beneficial',
            'bad': 'detrimental',
        }
        
        self.filler_sentences = [
            "There are various factors to consider in this regard.",
            "It's important to look at this from multiple angles.",
            "The situation is more nuanced than it might appear.",
            "Several elements come into play here.",
            "This requires careful consideration of multiple aspects.",
        ]
        
        self.run_on_connectors = [
            " and also ",
            " but then ",
            " and furthermore ",
            " which means that ",
            " and in addition to that ",
            " plus also ",
            " and on top of that ",
        ]
    
    def inject_ambiguous_pronouns(self, response: str) -> str:
        """
        Strategy 1: Replace specific nouns with ambiguous pronouns
        Makes it unclear what "it" or "they" refers to
        """
        words = response.split()
        nouns_replaced = 0
        
        result = []
        for i, word in enumerate(words):
            # Simple heuristic: capitalize words after periods might be nouns
            clean_word = word.strip('.,!?;:')
            
            # Replace some nouns with ambiguous pronouns
            if (len(clean_word) > 3 and 
                clean_word[0].isupper() and 
                i > 0 and 
                nouns_replaced < 2 and 
                random.random() < 0.4):
                
                if word.endswith('.'):
                    result.append("it.")
                elif word.endswith(','):
                    result.append("it,")
                else:
                    result.append("it")
                nouns_replaced += 1
            else:
                result.append(word)
        
        return " ".join(result)
    
    def inject_passive_voice(self, response: str) -> str:
        """
        Strategy 2: Convert to passive voice (hide agency)
        "John did X" → "X was done"
        """
        sentences = sent_tokenize(response)
        result = []
        
        passive_phrases = [
            "It was determined that",
            "It has been observed that",
            "It should be noted that",
            "It can be seen that",
            "It was found that",
        ]
        
        for sent in sentences:
            if random.random() < 0.5:
                # Wrap in passive construction
                phrase = random.choice(passive_phrases)
                result.append(f"{phrase} {sent.lower()}")
            else:
                result.append(sent)
        
        return " ".join(result)
    
    def inject_bury_lede(self, response: str) -> str:
        """
        Strategy 3: Move important info to end, add filler at start
        Real unclear responses often bury the key point
        """
        sentences = sent_tokenize(response)
        if len(sentences) < 2:
            # Add filler at start
            filler = random.sample(self.filler_sentences, min(2, len(self.filler_sentences)))
            return " ".join(filler) + " " + response
        
        # Assume first sentence is the key point - move it to end
        key_point = sentences[0]
        remaining = sentences[1:]
        
        # Add filler at beginning
        filler = random.sample(self.filler_sentences, min(2, len(self.filler_sentences)))
        
        return " ".join(filler) + " " + " ".join(remaining) + " " + key_point
    
    def inject_jargon(self, response: str) -> str:
        """
        Strategy 4: Replace simple words with unnecessarily complex ones
        "use" → "utilize", "help" → "facilitate"
        """
        words = response.split()
        result = []
        
        for word in words:
            clean = word.lower().strip('.,!?;:')
            punctuation = word[len(clean):] if len(word) > len(clean) else ""
            
            if clean in self.jargon_map and random.random() < 0.7:
                jargon = self.jargon_map[clean]
                # Preserve capitalization
                if word[0].isupper():
                    jargon = jargon.capitalize()
                result.append(jargon + punctuation)
            else:
                result.append(word)
        
        return " ".join(result)
    
    def inject_runon(self, response: str) -> str:
        """
        Strategy 5: Create run-on sentences
        Combine sentences with "and"/"but" without proper breaks
        """
        sentences = sent_tokenize(response)
        if len(sentences) < 2:
            return response
        
        result = []
        i = 0
        while i < len(sentences):
            if i < len(sentences) - 1 and random.random() < 0.6:
                # Combine two sentences
                connector = random.choice(self.run_on_connectors)
                combined = sentences[i].rstrip('.!?') + connector + sentences[i+1].lower()
                result.append(combined)
                i += 2
            else:
                result.append(sentences[i])
                i += 1
        
        return " ".join(result)
    
    def inject_disorganized(self, response: str) -> str:
        """
        Strategy 6: Disorganize structure (but NOT random shuffle)
        Move sentences around in a way that breaks logical flow
        """
        sentences = sent_tokenize(response)
        if len(sentences) < 3:
            return response
        
        # Interleave sentences with filler
        result = []
        for i, sent in enumerate(sentences):
            if i > 0 and random.random() < 0.3:
                result.append(random.choice(self.filler_sentences))
            result.append(sent)
        
        # Swap middle sentences (not random, just poor organization)
        if len(result) >= 4:
            mid = len(result) // 2
            result[mid], result[mid-1] = result[mid-1], result[mid]
        
        return " ".join(result)
    
    def inject_hedging_overload(self, response: str) -> str:
        """
        Strategy 7: Add excessive hedging making it unclear
        """
        hedges = [
            "sort of", "kind of", "maybe", "possibly", "perhaps",
            "in a way", "to some extent", "more or less", "somewhat"
        ]
        
        words = response.split()
        result = []
        
        for i, word in enumerate(words):
            result.append(word)
            # Insert hedge after verbs occasionally
            if i < len(words) - 1 and random.random() < 0.15:
                result.append(random.choice(hedges))
        
        return " ".join(result)
    
    def inject(self, response: str, strategy: str = "random") -> Tuple[str, str]:
        """
        Apply a manner violation injection
        Returns: (violated_response, strategy_used)
        """
        strategies = {
            'pronouns': self.inject_ambiguous_pronouns,
            'passive': self.inject_passive_voice,
            'bury': self.inject_bury_lede,
            'jargon': self.inject_jargon,
            'runon': self.inject_runon,
            'disorganized': self.inject_disorganized,
            'hedging': self.inject_hedging_overload,
        }
        
        if strategy == "random":
            strategy = random.choice(list(strategies.keys()))
        
        violated = strategies[strategy](response)
        return violated, f"manner_{strategy}"


# =============================================================================
# RELATION INJECTORS - Gradual topic drift
# =============================================================================

class RelationInjector:
    """
    Inject realistic relation violations:
    - Gradual topic drift (not sudden swaps)
    - Tangent that never returns
    - Answering adjacent question
    - Focus on wrong aspect
    """
    
    def __init__(self):
        self.drift_templates = [
            "{start} By the way, {tangent} Speaking of which, {drift}",
            "{start} This reminds me of {tangent} Actually, {drift}",
            "{start} You know what's interesting? {tangent} And that leads to {drift}",
        ]
        
        self.tangent_topics = [
            "there's a similar concept in nature where animals do this.",
            "I was reading about how this relates to psychology.",
            "historically, people approached this very differently.",
            "there's an economic angle to consider here as well.",
            "technology has really changed how we think about this.",
        ]
        
        self.drift_endings = [
            "So really, it all comes down to perspective.",
            "Which is why you have to look at the bigger picture.",
            "And that's what makes this topic so fascinating.",
            "It's all connected in interesting ways.",
            "There's so much more to explore here.",
        ]
    
    def inject_gradual_drift(self, response: str, context: str = "") -> str:
        """
        Strategy 1: Gradual topic drift (not sudden)
        Start on-topic, slowly drift away, never fully return
        """
        sentences = sent_tokenize(response)
        if not sentences:
            return response
        
        start = sentences[0]
        tangent = random.choice(self.tangent_topics)
        drift = random.choice(self.drift_endings)
        
        template = random.choice(self.drift_templates)
        return template.format(start=start, tangent=tangent, drift=drift)
    
    def inject_tangent_loop(self, response: str) -> str:
        """
        Strategy 2: Start on-topic, go on tangent, never come back
        """
        sentences = sent_tokenize(response)
        if not sentences:
            return response
        
        # Keep first sentence (on-topic hook)
        start = sentences[0]
        
        tangent_story = (
            "Actually, this makes me think about something completely different. "
            "Have you ever noticed how people always seem to focus on the wrong things? "
            "I find that quite interesting. There's a lot to unpack there. "
            f"{random.choice(self.tangent_topics)} "
            f"{random.choice(self.drift_endings)}"
        )
        
        return f"{start} {tangent_story}"
    
    def inject_adjacent_answer(self, response: str, context: str = "") -> str:
        """
        Strategy 3: Answer a related but different question
        """
        sentences = sent_tokenize(response)
        
        adjacent_intro = random.choice([
            "That's an interesting question, but what's really important is",
            "I think the more relevant point here is",
            "Rather than answering that directly, let me address",
            "The question you should really be asking is about",
        ])
        
        if sentences:
            return f"{adjacent_intro} something related. {response}"
        return f"{adjacent_intro} a different aspect entirely."
    
    def inject_wrong_focus(self, response: str) -> str:
        """
        Strategy 4: Focus on minor/irrelevant aspect of the topic
        """
        wrong_focus_intros = [
            "The most important thing here, really, is the terminology.",
            "What people often overlook is the historical context.",
            "I want to focus specifically on one small aspect:",
            "Let me address a minor but fascinating detail:",
        ]
        
        return f"{random.choice(wrong_focus_intros)} {response}"
    
    def inject_personal_agenda(self, response: str) -> str:
        """
        Strategy 5: Hijack to push unrelated personal agenda
        """
        agendas = [
            "but what really matters in today's world is personal development.",
            "which really shows why self-care is so important.",
            "and this is exactly why people need to be more mindful.",
            "proving once again that education is the key to everything.",
        ]
        
        sentences = sent_tokenize(response)
        if sentences:
            return f"{sentences[0]} But honestly, {random.choice(agendas)}"
        return response
    
    def inject(self, response: str, context: str = "", strategy: str = "random") -> Tuple[str, str]:
        """
        Apply a relation violation injection
        Returns: (violated_response, strategy_used)
        """
        strategies = {
            'drift': lambda r: self.inject_gradual_drift(r, context),
            'tangent': self.inject_tangent_loop,
            'adjacent': lambda r: self.inject_adjacent_answer(r, context),
            'focus': self.inject_wrong_focus,
            'agenda': self.inject_personal_agenda,
        }
        
        if strategy == "random":
            strategy = random.choice(list(strategies.keys()))
        
        violated = strategies[strategy](response)
        return violated, f"relation_{strategy}"


# =============================================================================
# MASTER INJECTOR CLASS
# =============================================================================

class RealisticViolationInjector:
    """
    Master class combining all realistic injectors
    """
    
    def __init__(self):
        self.quantity = QuantityInjector()
        self.quality = QualityInjector()
        self.manner = MannerInjector()
        self.relation = RelationInjector()
    
    def inject(
        self,
        response: str,
        maxim: str,
        context: str = "",
        strategy: str = "random"
    ) -> Dict:
        """
        Inject a violation of the specified maxim
        
        Args:
            response: Original clean response
            maxim: One of 'quantity', 'quality', 'manner', 'relation'
            context: Optional context for relation violations
            strategy: Specific strategy or 'random'
            
        Returns:
            Dict with 'violated_response', 'original', 'maxim', 'strategy'
        """
        if maxim == 'quantity':
            violated, strategy_used = self.quantity.inject(response, strategy)
        elif maxim == 'quality':
            violated, strategy_used = self.quality.inject(response, strategy)
        elif maxim == 'manner':
            violated, strategy_used = self.manner.inject(response, strategy)
        elif maxim == 'relation':
            violated, strategy_used = self.relation.inject(response, context, strategy)
        else:
            raise ValueError(f"Unknown maxim: {maxim}")
        
        return {
            'original_response': response,
            'violated_response': violated,
            'context': context,
            'maxim': maxim,
            'strategy': strategy_used,
            'labels': {
                'quantity': 1 if maxim == 'quantity' else 0,
                'quality': 1 if maxim == 'quality' else 0,
                'relation': 1 if maxim == 'relation' else 0,
                'manner': 1 if maxim == 'manner' else 0,
            }
        }
    
    def inject_batch(
        self,
        examples: List[Dict],
        target_per_maxim: int = 1000,
        include_clean: bool = True
    ) -> List[Dict]:
        """
        Inject violations into a batch of examples
        
        Args:
            examples: List of dicts with 'context' and 'response'
            target_per_maxim: Number of violations per maxim to generate
            include_clean: Whether to include clean examples
            
        Returns:
            List of injected examples
        """
        results = []
        maxims = ['quantity', 'quality', 'manner', 'relation']
        
        for maxim in maxims:
            count = 0
            for ex in examples:
                if count >= target_per_maxim:
                    break
                
                response = ex.get('response', ex.get('violated_response', ''))
                context = ex.get('context', '')
                
                if not response or len(response) < 20:
                    continue
                
                injected = self.inject(response, maxim, context)
                injected['source'] = 'realistic_injector'
                results.append(injected)
                count += 1
        
        # Add clean examples if requested
        if include_clean:
            clean_count = target_per_maxim // 2
            for ex in examples[:clean_count]:
                response = ex.get('response', ex.get('violated_response', ''))
                context = ex.get('context', '')
                
                if response:
                    results.append({
                        'original_response': response,
                        'violated_response': response,  # Same as original = clean
                        'context': context,
                        'maxim': 'clean',
                        'strategy': 'clean',
                        'labels': {
                            'quantity': 0,
                            'quality': 0,
                            'relation': 0,
                            'manner': 0,
                        },
                        'source': 'clean'
                    })
        
        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def demo():
    """Demonstrate the injectors"""
    injector = RealisticViolationInjector()
    
    example_response = (
        "The capital of France is Paris. It's known for the Eiffel Tower "
        "and is one of the most visited cities in the world."
    )
    
    print("=" * 80)
    print("REALISTIC VIOLATION INJECTOR DEMO")
    print("=" * 80)
    print(f"\nOriginal: {example_response}\n")
    print("-" * 80)
    
    for maxim in ['quantity', 'quality', 'manner', 'relation']:
        result = injector.inject(example_response, maxim)
        print(f"\n{maxim.upper()} Violation ({result['strategy']}):")
        print(f"{result['violated_response']}")
        print("-" * 80)


if __name__ == '__main__':
    demo()
