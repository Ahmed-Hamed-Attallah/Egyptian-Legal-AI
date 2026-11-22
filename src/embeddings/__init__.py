# src/embeddings/__init__.py
"""
Embeddings Module.
Provides interfaces and implementations for generating vector representations
of legal text and queries.
"""
from src.embeddings.base import BaseEmbeddingModel
from src.embeddings.embedding_model import SentenceTransformerEmbedding
from src.embeddings.model_registry import EmbeddingModelRegistry, get_default_embedding_model

__all__ = [
    "BaseEmbeddingModel",
    "SentenceTransformerEmbedding",
    "EmbeddingModelRegistry",
    "get_default_embedding_model",
]