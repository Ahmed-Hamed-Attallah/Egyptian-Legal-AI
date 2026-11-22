# src/llm/providers/__init__.py
"""
Concrete LLM provider implementations.
"""
from src.llm.providers.groq_provider import GroqProvider
from src.llm.providers.openai_provider import OpenAIProvider
from src.llm.providers.transformers_provider import TransformersProvider

# Gemini is optional — only import if available
try:
    from src.llm.providers.gemini_provider import GeminiProvider
except ImportError:
    GeminiProvider = None

__all__ = [
    "GroqProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "TransformersProvider",
]