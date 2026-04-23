"""
GriceBench Integrated Repair Model - Part 1, Step 3
====================================================

Unified repair model that intelligently routes violations to the
appropriate repair strategy:

- RELATION violations → Retrieval-based repair (new approach)
- QUALITY/QUANTITY/MANNER violations → T5-based editing (existing approach)

This solves the core problem: Relation violations can't be fixed by editing
because they require completely new content about a different topic.

Architecture:
┌─────────────────┐
│ Violation Label │
└────────┬────────┘
         │
         ▼
    ┌─────────────┐
    │   Router    │──── RELATION ────▶ RelationRepairRetriever
    └─────────────┘                              │
         │                                       ▼
         │                           Semantic Search + Return
    OTHER MAXIMS
         │
         ▼
    T5 Repair Model
         │
         ▼
    Edited Response

Author: GriceBench
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import sys

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IntegratedRepairConfig:
    """Configuration for the integrated repair model."""
    t5_model_path: str = "models/repair/repair_model"
    corpus_path: str = "data_processed/topical_corpus.json"
    index_path: str = "data_processed/faiss_index.pkl"
    
    # T5 generation settings
    max_input_length: int = 512
    max_output_length: int = 256
    num_beams: int = 4
    early_stopping: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# INTEGRATED REPAIR MODEL
# ============================================================================

class IntegratedRepairModel:
    """
    Combined repair model with intelligent routing.
    
    Routes violations to the optimal repair strategy:
    - Relation → Retrieval (can't edit off-topic to on-topic)
    - Quality → T5 edit (fix factual errors)
    - Quantity → T5 edit (add/remove information)
    - Manner → T5 edit (improve clarity)
    
    Usage:
        model = IntegratedRepairModel()
        repaired = model.repair(
            context="What is your favorite food?",
            evidence="",
            response="The stock market is up.",
            violation_type="RELATION"
        )
    """
    
    def __init__(
        self,
        t5_model_path: str = None,
        corpus_path: str = None,
        config: IntegratedRepairConfig = None,
        lazy_load: bool = False
    ):
        """
        Initialize the integrated repair model.
        
        Args:
            t5_model_path: Path to trained T5 repair model
            corpus_path: Path to topical corpus for retrieval
            config: Configuration object
            lazy_load: If True, don't load models until first use
        """
        self.config = config or IntegratedRepairConfig()
        
        if t5_model_path:
            self.config.t5_model_path = t5_model_path
        if corpus_path:
            self.config.corpus_path = corpus_path
        
        self.t5_model = None
        self.t5_tokenizer = None
        self.relation_retriever = None
        self.device = torch.device(self.config.device)
        
        self._t5_loaded = False
        self._retriever_loaded = False
        
        if not lazy_load:
            self._load_all()
    
    def _load_all(self):
        """Load both repair systems."""
        self._load_t5()
        self._load_retriever()
    
    def _load_t5(self):
        """Load the T5 repair model for non-Relation violations."""
        if self._t5_loaded:
            return
        
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            model_path = Path(self.config.t5_model_path)
            
            if not model_path.exists():
                print(f"Warning: T5 model not found at {model_path}")
                print("T5 repair will be disabled. Only retrieval-based repair available.")
                return
            
            print(f"Loading T5 repair model from {model_path}")
            
            self.t5_tokenizer = T5Tokenizer.from_pretrained(model_path)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.t5_model.to(self.device)
            self.t5_model.eval()
            
            self._t5_loaded = True
            print(f"  T5 model loaded on {self.device}")
            
        except Exception as e:
            print(f"Warning: Could not load T5 model: {e}")
            print("T5 repair will be disabled.")
    
    def _load_retriever(self):
        """Load the retrieval system for Relation violations."""
        if self._retriever_loaded:
            return
        
        try:
            from build_retrieval_system import RelationRepairRetriever, RetrieverConfig
            
            retriever_config = RetrieverConfig(
                corpus_path=self.config.corpus_path,
                index_path=self.config.index_path
            )
            
            print(f"Loading retrieval system...")
            self.relation_retriever = RelationRepairRetriever(retriever_config)
            self._retriever_loaded = True
            print("  Retriever loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load retriever: {e}")
            print("Relation repair will fall back to T5.")
    
    def repair(
        self,
        context: str,
        evidence: str,
        response: str,
        violation_type: Union[str, List[str]]
    ) -> str:
        """
        Repair a violated response.
        
        Intelligently routes to the appropriate repair strategy based on
        violation type.
        
        Args:
            context: The conversation context
            evidence: Any available evidence/knowledge
            response: The violated response to repair
            violation_type: One of "QUANTITY", "QUALITY", "RELATION", "MANNER"
                           or a list for multi-violation repair
        
        Returns:
            The repaired response
        """
        # Normalize violation type(s)
        if isinstance(violation_type, str):
            violations = [violation_type.upper()]
        else:
            violations = [v.upper() for v in violation_type]
        
        # Route based on violation type
        if "RELATION" in violations:
            # Relation violations need retrieval, not editing
            return self._repair_relation(context, response)
        else:
            # Other violations can be fixed by editing
            return self._repair_with_t5(context, evidence, response, violations)
    
    def _repair_relation(self, context: str, response: str) -> str:
        """
        Repair a Relation violation using retrieval.
        
        When a response is off-topic, we can't "edit" it to be on-topic.
        Instead, we retrieve a relevant response from our corpus.
        """
        if self.relation_retriever is None:
            self._load_retriever()
        
        if self.relation_retriever is None:
            # Fallback to T5 if retriever not available
            print("Warning: Retriever not available, falling back to T5")
            return self._repair_with_t5(context, "", response, ["RELATION"])
        
        return self.relation_retriever.repair_relation_violation(context, response)
    
    def _repair_with_t5(
        self,
        context: str,
        evidence: str,
        response: str,
        violations: List[str]
    ) -> str:
        """
        Repair using the T5 editing model.
        
        Used for Quality, Quantity, and Manner violations which can be
        fixed by editing the existing response.
        """
        if self.t5_model is None:
            self._load_t5()
        
        if self.t5_model is None:
            # No T5 available, return original
            print("Warning: T5 model not available, returning original response")
            return response
        
        # Format input with control tokens
        input_text = self._format_repair_input(context, evidence, response, violations)
        
        # Tokenize
        inputs = self.t5_tokenizer(
            input_text,
            max_length=self.config.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.t5_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=self.config.max_output_length,
                num_beams=self.config.num_beams,
                early_stopping=self.config.early_stopping
            )
        
        # Decode
        repaired = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return repaired.strip()
    
    def _format_repair_input(
        self,
        context: str,
        evidence: str,
        response: str,
        violations: List[str]
    ) -> str:
        """
        Format input for T5 with control tokens.
        
        Format: [REPAIR] [VIOLATION=X] [CONTEXT] ... [EVIDENCE] ... [RESPONSE] ...
        """
        # Build violation tokens
        violation_tokens = " ".join(f"[VIOLATION={v}]" for v in violations)
        
        # Build input
        parts = ["[REPAIR]", violation_tokens]
        
        parts.append(f"[CONTEXT] {context.strip()}")
        
        if evidence and evidence.strip():
            parts.append(f"[EVIDENCE] {evidence.strip()}")
        
        parts.append(f"[RESPONSE] {response.strip()}")
        
        return " ".join(parts)
    
    def repair_batch(
        self,
        examples: List[Dict],
        show_progress: bool = True
    ) -> List[str]:
        """
        Repair a batch of violated responses.
        
        Args:
            examples: List of dicts with keys: context, evidence, response, violation_type
            show_progress: Whether to show progress bar
        
        Returns:
            List of repaired responses
        """
        results = []
        
        iterator = examples
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(examples, desc="Repairing")
            except ImportError:
                pass
        
        for ex in iterator:
            repaired = self.repair(
                context=ex.get("context", ""),
                evidence=ex.get("evidence", ""),
                response=ex.get("response", ""),
                violation_type=ex.get("violation_type", "")
            )
            results.append(repaired)
        
        return results
    
    def get_repair_strategy(self, violation_type: str) -> str:
        """
        Get the repair strategy for a violation type.
        
        Useful for logging/debugging.
        """
        if violation_type.upper() == "RELATION":
            return "retrieval"
        else:
            return "t5_edit"


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Demo the integrated repair model."""
    print("="*70)
    print("INTEGRATED REPAIR MODEL DEMO")
    print("="*70)
    
    # Initialize model
    print("\nInitializing integrated repair model...")
    model = IntegratedRepairModel()
    
    # Test cases for each violation type
    test_cases = [
        {
            "context": "What is your favorite food?",
            "evidence": "",
            "response": "The stock market closed up 2% yesterday.",
            "violation_type": "RELATION",
            "expected_strategy": "retrieval"
        },
        {
            "context": "When was the Eiffel Tower built?",
            "evidence": "The Eiffel Tower was completed in 1889 for the World's Fair.",
            "response": "The Eiffel Tower was built in 1920 for tourism.",
            "violation_type": "QUALITY",
            "expected_strategy": "t5_edit"
        },
        {
            "context": "Can you explain quantum computing?",
            "evidence": "",
            "response": "Yes.",
            "violation_type": "QUANTITY",
            "expected_strategy": "t5_edit"
        },
        {
            "context": "How do I make bread?",
            "evidence": "",
            "response": "Combine triticum with H2O and saccharomyces cerevisiae then apply thermal energy.",
            "violation_type": "MANNER",
            "expected_strategy": "t5_edit"
        }
    ]
    
    print("\n" + "-"*70)
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case['violation_type']} violation")
        print(f"Strategy: {model.get_repair_strategy(case['violation_type'])}")
        print(f"Context: {case['context']}")
        print(f"Original: {case['response']}")
        
        repaired = model.repair(
            context=case["context"],
            evidence=case["evidence"],
            response=case["response"],
            violation_type=case["violation_type"]
        )
        
        print(f"Repaired: {repaired}")
        print("-"*70)
    
    print("\n" + "="*70)
    print("INTEGRATED REPAIR MODEL READY!")
    print("="*70)
    print("\nNext step: Run evaluate_relation_repair.py to evaluate improvements")


if __name__ == "__main__":
    main()
