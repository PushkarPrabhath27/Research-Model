"""
GriceBench Retrieval System - Part 1, Step 2
=============================================

High-performance retrieval system for fixing Relation violations using
semantic search with FAISS and sentence-transformers.

Architecture:
- Encoder: sentence-transformers/all-MiniLM-L6-v2 (384-dim, 80MB, fast)
- Index: FAISS IndexFlatIP (inner product = cosine after L2 normalization)
- Retrieval: Top-k similar responses to the context

This replaces the broken "edit" approach for Relation violations with
a "retrieve relevant response" approach.

Author: GriceBench
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RetrieverConfig:
    """Configuration for the retrieval system."""
    corpus_path: str = "data_processed/topical_corpus.json"
    index_path: str = "data_processed/faiss_index.pkl"
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    min_similarity: float = 0.3  # Minimum cosine similarity to return
    batch_size: int = 64  # For encoding


# ============================================================================
# RETRIEVER CLASS
# ============================================================================

class RelationRepairRetriever:
    """
    Retrieval system for fixing Relation violations.
    
    Uses semantic search to find contextually relevant responses
    instead of trying to edit off-topic text.
    
    Usage:
        retriever = RelationRepairRetriever()
        repaired = retriever.repair_relation_violation(
            context="What is your favorite food?",
            original_response="The stock market is up 2%."
        )
    """
    
    def __init__(self, config: RetrieverConfig = None, lazy_load: bool = False):
        """
        Initialize the retriever.
        
        Args:
            config: Configuration options
            lazy_load: If True, don't load models until first use
        """
        self.config = config or RetrieverConfig()
        self.encoder = None
        self.index = None
        self.response_metadata = None
        self.all_responses = None
        
        if not lazy_load:
            self._initialize()
    
    def _initialize(self):
        """Load models and build/load index."""
        self._load_encoder()
        
        # Try to load existing index, otherwise build new one
        if Path(self.config.index_path).exists():
            self._load_index()
        else:
            self._load_corpus()
            self._build_index()
            self._save_index()
    
    def _load_encoder(self):
        """Load the sentence encoder model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading encoder: {self.config.encoder_model}")
            self.encoder = SentenceTransformer(self.config.encoder_model)
            print(f"  Embedding dimension: {self.encoder.get_sentence_embedding_dimension()}")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            )
    
    def _load_corpus(self):
        """Load the topical corpus."""
        corpus_path = Path(self.config.corpus_path)
        
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus not found at {corpus_path}. "
                "Run create_response_corpus.py first."
            )
        
        print(f"Loading corpus from {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        
        # Flatten corpus into lists
        self.all_responses = []
        self.response_metadata = []
        
        for topic, responses in corpus.items():
            for resp in responses:
                self.all_responses.append(resp["response"])
                self.response_metadata.append({
                    "topic": topic,
                    "context": resp["context"],
                    "response": resp["response"],
                    "source": resp.get("source", "unknown")
                })
        
        print(f"  Loaded {len(self.all_responses):,} responses across {len(corpus)} topics")
    
    def _build_index(self):
        """Build FAISS index from corpus."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required. Install with: "
                "pip install faiss-cpu  (or faiss-gpu for GPU)"
            )
        
        print(f"Building FAISS index for {len(self.all_responses):,} responses...")
        
        # Encode all responses in batches
        embeddings = self.encoder.encode(
            self.all_responses,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )
        
        # Create FAISS index (Inner Product = cosine after normalization)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        print(f"  Index built: {self.index.ntotal} vectors, {dimension}D")
    
    def _save_index(self):
        """Save index and metadata for fast loading."""
        index_path = Path(self.config.index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle (includes FAISS index + metadata)
        save_data = {
            "index_bytes": self._serialize_faiss_index(),
            "response_metadata": self.response_metadata,
            "all_responses": self.all_responses
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Saved index to {index_path}")
    
    def _serialize_faiss_index(self) -> bytes:
        """Serialize FAISS index to bytes."""
        import faiss
        import io
        
        # Write to bytes
        writer = faiss.VectorIOWriter()
        faiss.write_index(self.index, writer)
        return writer.get_bytes()
    
    def _load_index(self):
        """Load pre-built index."""
        import faiss
        
        print(f"Loading pre-built index from {self.config.index_path}")
        
        with open(self.config.index_path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Deserialize FAISS index
        reader = faiss.VectorIOReader()
        reader.set_bytes(save_data["index_bytes"])
        self.index = faiss.read_index(reader)
        
        self.response_metadata = save_data["response_metadata"]
        self.all_responses = save_data["all_responses"]
        
        print(f"  Loaded {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        """
        Retrieve top-k similar responses to a query.
        
        Args:
            query: The query text (usually the conversation context)
            k: Number of results to return (default: config.top_k)
        
        Returns:
            List of dicts with keys: response, context, topic, score
        """
        if self.encoder is None:
            self._initialize()
        
        k = k or self.config.top_k
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Build results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
            if dist < self.config.min_similarity:
                continue
            
            metadata = self.response_metadata[idx]
            results.append({
                "response": metadata["response"],
                "context": metadata["context"],
                "topic": metadata["topic"],
                "source": metadata["source"],
                "score": float(dist),
                "rank": i + 1
            })
        
        return results
    
    def repair_relation_violation(
        self,
        context: str,
        original_response: str,
        return_candidates: bool = False
    ) -> str:
        """
        Repair a Relation violation by finding a contextually relevant response.
        
        This is the main entry point for the Relation repair pipeline.
        
        Args:
            context: The conversation context (what the user said/asked)
            original_response: The off-topic response that violated Relation
            return_candidates: If True, return all candidates instead of just best
        
        Returns:
            A relevant response from the corpus that matches the context
        """
        # Retrieve candidates based on context
        candidates = self.retrieve(context, k=self.config.top_k)
        
        if not candidates:
            # Fallback: return original if no good matches found
            print(f"Warning: No relevant responses found for context: {context[:50]}...")
            return original_response
        
        if return_candidates:
            return candidates
        
        # Return the best match
        best = candidates[0]
        return best["response"]
    
    def get_relevance_score(self, context: str, response: str) -> float:
        """
        Calculate semantic similarity between context and response.
        
        Useful for evaluation: comparing original vs repaired relevance.
        """
        if self.encoder is None:
            self._initialize()
        
        embeddings = self.encoder.encode(
            [context, response],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Cosine similarity (dot product since normalized)
        return float(np.dot(embeddings[0], embeddings[1]))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def build_index_standalone(corpus_path: str = None, index_path: str = None):
    """
    Build FAISS index from corpus (standalone utility function).
    
    Can be run independently to pre-build the index.
    """
    config = RetrieverConfig()
    if corpus_path:
        config.corpus_path = corpus_path
    if index_path:
        config.index_path = index_path
    
    retriever = RelationRepairRetriever(config)
    print("\nIndex built and saved successfully!")
    return retriever


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Build retrieval system and run demo."""
    print("="*70)
    print("BUILDING RELATION REPAIR RETRIEVAL SYSTEM")
    print("="*70)
    
    # Create retriever (will build index if not exists)
    retriever = RelationRepairRetriever()
    
    # Demo
    print("\n" + "="*70)
    print("DEMO: Relation Violation Repair")
    print("="*70)
    
    test_cases = [
        {
            "context": "What is your favorite food?",
            "violated": "The stock market closed up 2% yesterday."
        },
        {
            "context": "Do you have any pets?",
            "violated": "I think the weather will be nice tomorrow."
        },
        {
            "context": "How was your weekend?",
            "violated": "The capital of France is Paris."
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Case {i} ---")
        print(f"Context: {case['context']}")
        print(f"Violated Response: {case['violated']}")
        
        # Get original relevance
        orig_score = retriever.get_relevance_score(case['context'], case['violated'])
        
        # Repair
        repaired = retriever.repair_relation_violation(case['context'], case['violated'])
        
        # Get repaired relevance
        new_score = retriever.get_relevance_score(case['context'], repaired)
        
        print(f"Repaired Response: {repaired}")
        print(f"Relevance: {orig_score:.3f} → {new_score:.3f} (Δ{new_score-orig_score:+.3f})")
    
    print("\n" + "="*70)
    print("RETRIEVAL SYSTEM READY!")
    print("="*70)
    print("\nNext step: Run integrated_repair_model.py to combine with T5 repair")


if __name__ == "__main__":
    main()
