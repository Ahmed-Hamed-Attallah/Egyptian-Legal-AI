# src/vector_store/base.py
"""
Abstract base class for vector storage backends.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from src.domain.chunks import Chunk
from src.domain.queries import RetrievedChunk
from src.retrieval.metadata_filters import MetadataFilter

class BaseVectorStore(ABC):
    """
    Database-independent interface for vector storage operations.
    """
    @abstractmethod
    def insert(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Insert chunks and their corresponding embeddings."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int, filters: Optional[MetadataFilter] = None) -> List[RetrievedChunk]:
        """Search for similar vectors and return ranked RetrievedChunks."""
        pass

    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> None:
        """Delete vectors by their unique IDs."""
        pass

    @abstractmethod
    def update(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Update existing vectors (typically handled via upsert)."""
        pass