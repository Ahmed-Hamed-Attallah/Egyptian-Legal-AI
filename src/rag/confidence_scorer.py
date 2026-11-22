# src/rag/confidence_scorer.py
"""
Confidence scoring for RAG responses.
"""
import logging
from typing import List

from src.domain.queries import RetrievedChunk

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Calculates response confidence based on retrieval quality.
    """

    def calculate(self, chunks: List[RetrievedChunk]) -> float:
        """
        Confidence is based on:
        1. Top chunk's rerank score (higher = more confident)
        2. Number of supporting chunks above threshold (more = more confident)
        """
        if not chunks:
            return 0.0

        top_score = chunks[0].score
        # Count chunks with strong relevance (e.g., > 0.5 sigmoid probability)
        num_supporting = sum(1 for c in chunks if c.score > 0.5)

        # Weighted combination: 60% top score, 40% supporting evidence density
        # Cap supporting density at 3 chunks to prevent overconfidence from noise
        supporting_factor = min(num_supporting / 3.0, 1.0)

        confidence = (top_score * 0.6) + (supporting_factor * 0.4)
        return round(max(0.0, min(1.0, confidence)), 3)

    def get_label(self, confidence: float) -> str:
        if confidence >= 0.75:
            return "high"
        elif confidence >= 0.45:
            return "medium"
        else:
            return "low"