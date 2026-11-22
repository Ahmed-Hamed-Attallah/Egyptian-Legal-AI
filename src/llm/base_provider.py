# src/llm/base_provider.py
"""
Abstract base class for LLM providers.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from src.domain.responses import GenerationRequest, GenerationResponse
from src.domain.exceptions import LLMError

class BaseLLMProvider(ABC):
    """
    Provider-independent interface for LLM generation.
    """
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate a response based on the provided request.
        """
        pass

    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """
        Returns metadata about the provider (name, model, capabilities).
        """
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the provider is available and responsive.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the provider name.
        """
        pass