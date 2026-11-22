# src/generation/__init__.py
"""
Generation Module.
Provides prompt construction, context building, and answer generation for legal RAG.
"""
from src.generation.prompt_templates import LegalPromptTemplates
from src.generation.context_builder import ContextBuilder
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_formatter import CitationFormatter
from src.generation.response_validator import ResponseValidator

__all__ = [
    "LegalPromptTemplates",
    "ContextBuilder",
    "AnswerGenerator",
    "CitationFormatter",
    "ResponseValidator",
]