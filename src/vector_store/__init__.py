# src/vector_store/__init__.py
"""
Vector Storage Module.
Provides database-agnostic interfaces and concrete implementations for 
storing and retrieving legal chunk embeddings.
"""
from src.vector_store.base import BaseVectorStore
from src.vector_store.qdrant_store import QdrantVectorStore
from src.vector_store.collection_manager import CollectionManager

__all__ = [
    "BaseVectorStore",
    "QdrantVectorStore",
    "CollectionManager",
]