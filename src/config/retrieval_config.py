"""
Configuration for the RAG retrieval pipeline.
Defines limits, thresholds, and algorithmic constants for hybrid search.
"""

from dataclasses import dataclass

from src.config.settings import settings


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Immutable configuration for the retrieval pipeline.
    Aggregates settings and defines algorithmic constants.
    """

    # Retrieval Limits (Top-K)

    # Number of candidates to retrieve from Vector Search (Dense)
    DENSE_TOP_K: int = settings.RETRIEVAL_TOP_K

    # Number of candidates to retrieve from BM25 (Sparse)
    SPARSE_TOP_K: int = settings.RETRIEVAL_TOP_K

    # Final number of chunks to send to the LLM (after reranking)
    FINAL_TOP_K: int = settings.RERANKER_TOP_K

    
    # Hybrid Search (Reciprocal Rank Fusion)
    
    # RRF constant 'k'. Standard value is 60.
    # Formula: score = sum(1 / (k + rank))
    RRF_K: int = settings.RRF_K

    
    # Score Thresholds (Filtering)
    
    # Minimum cosine similarity (0.0 to 1.0) to consider a vector match relevant.
    # Matches below this are discarded to save compute during reranking.
    MIN_DENSE_SCORE: float = 0.35
    
    # Minimum score from the Reranker (Cross-Encoder) to keep a chunk.
    # This is model-specific, but 0.1 is a safe conservative start.
    MIN_RERANK_SCORE: float = 0.1

    
    # Query Constraints
    
    # Maximum characters allowed in a user query (prevents abuse/DoS)
    MAX_QUERY_LENGTH: int = 2000


# Global singleton instance for easy import
retrieval_config = RetrievalConfig()