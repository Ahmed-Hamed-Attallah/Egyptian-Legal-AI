# src/rag/__init__.py
"""
RAG Module.
Provides end-to-end RAG pipeline for legal question answering.
"""
from src.rag.pipeline import RAGPipeline
from src.rag.orchestrator import RAGOrchestrator
from src.rag.citations import CitationResolver

__all__ = [
    "RAGPipeline",
    "RAGOrchestrator",
    "CitationResolver",
]