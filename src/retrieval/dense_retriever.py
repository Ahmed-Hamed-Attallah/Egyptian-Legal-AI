# src/retrieval/dense_retriever.py
"""
Semantic (Dense) Retriever.
Uses Phase 6 embeddings and Phase 7 vector store.
"""
import logging
from typing import List, Optional

from src.embeddings.base import BaseEmbeddingModel
from src.vector_store.base import BaseVectorStore
from src.domain.queries import RetrievedChunk
from src.domain.exceptions import RetrievalError
from src.retrieval.metadata_filters import MetadataFilter
from src.config.retrieval_config import retrieval_config
from src.config.settings import settings

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Retrieves chunks based on semantic similarity using dense embeddings.

    Low-confidence matches (cosine similarity below MIN_DENSE_SCORE) are
    discarded before hybrid fusion, so noise never reaches the reranker
    or the LLM.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        top_k: Optional[int] = None,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k or settings.RETRIEVAL_TOP_K

    def retrieve(
        self,
        query: str,
        filters: Optional[MetadataFilter] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve semantically similar chunks using dense embeddings.
        """
        if not query or not query.strip():
            return []

        effective_top_k = top_k or self.top_k or settings.RETRIEVAL_TOP_K

        try:
            query_embeddings = self.embedding_model.encode_queries(query)
            if not query_embeddings or not query_embeddings[0]:
                raise RetrievalError(
                    "Query embedding generation returned an empty result.",
                    query=query,
                )

            query_embedding = query_embeddings[0]
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=effective_top_k,
                filters=filters,
            )
            return self._apply_score_threshold(results, query)

        except RetrievalError:
            raise
        except Exception as exc:
            logger.error("Dense retrieval failed: %s", exc)
            raise RetrievalError(
                f"Dense retrieval failed: {exc}",
                query=query,
            ) from exc

    def _apply_score_threshold(
        self,
        results: List[RetrievedChunk],
        query: str,
    ) -> List[RetrievedChunk]:
        """Drop matches whose cosine similarity is below MIN_DENSE_SCORE."""
        if not results:
            return []

        min_score = retrieval_config.MIN_DENSE_SCORE
        kept = [r for r in results if r.score >= min_score]

        dropped = len(results) - len(kept)
        if dropped:
            logger.info(
                "DENSE THRESHOLD | query=%s | kept=%d | dropped=%d | min_score=%.2f",
                query[:80],
                len(kept),
                dropped,
                min_score,
            )

        return kept