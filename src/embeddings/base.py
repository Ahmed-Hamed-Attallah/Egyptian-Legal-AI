# src/embeddings/base.py
"""
Base interface for embedding models.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseEmbeddingModel(ABC):
    """
    Abstract base class for embedding providers.
    Distinguishes between document and query encoding to support 
    models that require asymmetric prefixes (e.g., E5).
    """
    @abstractmethod
    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of document texts into embeddings.
        """
        pass

    @abstractmethod
    def encode_queries(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of query texts into embeddings.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns metadata about the model (name, dimension, etc.).
        """
        pass