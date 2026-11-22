# src/reranking/base.py
"""
Abstract base class for rerankers.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.queries import RetrievedChunk

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Rerank a list of candidates based on the query."""
        pass