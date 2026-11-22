# src/chunking/__init__.py
"""
Legal Chunking Module.
Provides strategies for splitting preprocessed legal documents into 
retrieval-ready chunks while preserving legal boundaries and metadata.
"""
from src.chunking.base_chunker import BaseChunker
from src.chunking.recursive_chunker import RecursiveChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.legal_chunker import LegalChunker
from src.chunking.pipeline import ChunkingPipeline, ChunkingConfig

__all__ = [
    "BaseChunker",
    "RecursiveChunker",
    "SemanticChunker",
    "LegalChunker",
    "ChunkingPipeline",
    "ChunkingConfig",
]