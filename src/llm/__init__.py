# src/llm/__init__.py
"""
LLM Providers Module.
Provides provider-agnostic interfaces and concrete implementations for 
generating legal answers with automatic fallback capabilities.
"""
from src.llm.base_provider import BaseLLMProvider
from src.llm.provider_factory import ProviderFactory
from src.llm.provider_router import ProviderRouter
from src.llm.fallback_manager import FallbackManager
from src.llm.model_registry import LLMModelRegistry

__all__ = [
    "BaseLLMProvider",
    "ProviderFactory",
    "ProviderRouter",
    "FallbackManager",
    "LLMModelRegistry",
]