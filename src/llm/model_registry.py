# src/llm/model_registry.py
"""
LLM Model Registry.
Manages model configurations and capabilities.
"""
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from src.config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    model_id: str
    provider: str
    max_tokens: int
    supports_streaming: bool
    context_window: int
    description: str

class LLMModelRegistry:
    """
    Registry for supported LLM models and their configurations.
    """
    _models: Dict[str, ModelConfig] = {
        "llama-3.3-70b-versatile": ModelConfig(
            model_id="llama-3.3-70b-versatile",
            provider="groq",
            max_tokens=8192,
            supports_streaming=True,
            context_window=128000,
            description="Meta Llama 3.3 70B via Groq"
        ),
        "gpt-4o-mini": ModelConfig(
            model_id="gpt-4o-mini",
            provider="openai",
            max_tokens=16384,
            supports_streaming=True,
            context_window=128000,
            description="OpenAI GPT-4o Mini"
        ),
        "gemini-2.0-flash-exp": ModelConfig(
            model_id="gemini-2.0-flash-exp",
            provider="gemini",
            max_tokens=8192,
            supports_streaming=True,
            context_window=1000000,
            description="Google Gemini 2.0 Flash"
        ),
        "local-model": ModelConfig(
            model_id="local-model",
            provider="transformers",
            max_tokens=2048,
            supports_streaming=False,
            context_window=4096,
            description="Local Hugging Face Model"
        ),
    }

    @classmethod
    def get_model(cls, model_id: str) -> Optional[ModelConfig]:
        """Retrieve model configuration by ID."""
        return cls._models.get(model_id)

    @classmethod
    def get_default_model(cls) -> ModelConfig:
        """Get the default configured model."""
        model_id = settings.LLM_MODEL
        config = cls._models.get(model_id)
        if not config:
            logger.warning(f"Model {model_id} not in registry, using fallback")
            return cls._models.get("llama-3.3-70b-versatile")
        return config

    @classmethod
    def register_model(cls, config: ModelConfig):
        """Register a new model configuration."""
        cls._models[config.model_id] = config
        logger.info(f"Registered model: {config.model_id}")

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model IDs."""
        return list(cls._models.keys())