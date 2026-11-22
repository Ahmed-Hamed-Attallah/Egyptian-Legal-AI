# src/embeddings/model_registry.py
"""
Embedding Model Registry.
Manages instantiation and caching of embedding models.
"""
import logging
from typing import Dict, Optional
from src.embeddings.base import BaseEmbeddingModel
from src.embeddings.embedding_model import SentenceTransformerEmbedding
from src.config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingModelRegistry:
    """
    Singleton registry for embedding models.
    Prevents reloading the same model multiple times.
    """
    _instances: Dict[str, BaseEmbeddingModel] = {}

    @classmethod
    def get_model(cls, model_name: Optional[str] = None, **kwargs) -> BaseEmbeddingModel:
        name = model_name or settings.EMBEDDING_MODEL_NAME
        
        if name not in cls._instances:
            logger.info(f"Model '{name}' not in registry. Initializing...")
            # Currently only supporting sentence_transformers
            if settings.EMBEDDING_PROVIDER == "sentence_transformers":
                cls._instances[name] = SentenceTransformerEmbedding(model_name=name, **kwargs)
            else:
                raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")
                
        return cls._instances[name]

    @classmethod
    def clear_cache(cls):
        """Clears the model cache to free memory."""
        cls._instances.clear()
        logger.info("Embedding model registry cache cleared.")

def get_default_embedding_model() -> BaseEmbeddingModel:
    """Convenience function to get the default configured model."""
    return EmbeddingModelRegistry.get_model()