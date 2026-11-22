# src/embeddings/embedding_model.py
"""
Concrete implementation of the embedding model using Sentence Transformers.
"""
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.embeddings.base import BaseEmbeddingModel
from src.domain.exceptions import EmbeddingError
from src.config.model_config import get_embedding_config
from src.config.settings import settings

logger = logging.getLogger(__name__)

class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    SentenceTransformer-based embedding model.
    Hardware-aware and optimized for Arabic legal text.
    """
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None, 
                 batch_size: Optional[int] = None, normalize: Optional[bool] = None):
        config = get_embedding_config()
        
        self.model_name = model_name or config["model_name"]
        self.device = device or config["device"]
        self.batch_size = batch_size or config["batch_size"]
        self.normalize = normalize if normalize is not None else config["normalize_embeddings"]
        self.dimension = config["dimension"]
        
        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Verify dimension matches config
            actual_dim = self.model.get_embedding_dimension()
            if actual_dim != self.dimension:
                logger.warning(f"Configured dimension ({self.dimension}) differs from model actual dimension ({actual_dim}). Updating config dimension.")
                self.dimension = actual_dim
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            raise EmbeddingError(message=f"Model loading failed: {str(e)}", model_name=self.model_name)

    def encode_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        # E5 models require "passage: " prefix for documents
        if "e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]
            
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            raise EmbeddingError(message=f"Encoding failed: {str(e)}", model_name=self.model_name)

    def encode_queries(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
            
        # E5 models require "query: " prefix for queries
        if "e5" in self.model_name.lower():
            texts = [f"query: {t}" for t in texts]
            
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise EmbeddingError(message=f"Encoding failed: {str(e)}", model_name=self.model_name)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length,
            "provider": "sentence_transformers"
        }