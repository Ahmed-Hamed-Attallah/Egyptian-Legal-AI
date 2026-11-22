# src/retrieval/hybrid_retriever.py
"""
Hybrid Retriever combining dense and sparse strategies.
"""
import logging
from typing import List, Optional

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.score_fusion import ReciprocalRankFusion
from src.domain.queries import RetrievedChunk
from src.domain.exceptions import RetrievalError
from src.retrieval.metadata_filters import MetadataFilter
from src.monitoring.pipeline_monitor import stage_timer
from src.config.settings import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Executes both dense and sparse retrieval, then fuses the results.
    Handles partial failures gracefully (if one retriever fails, the other's
    results are used).

    ``top_k`` semantics: this is the CANDIDATE POOL size for retrieval.
    Final answer-chunk selection happens later in the reranking stage.
    When omitted, ``settings.RETRIEVAL_TOP_K`` is used so a wide candidate
    pool reaches the reranker.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        fusion: ReciprocalRankFusion,
        top_k: Optional[int] = None,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        filters: Optional[MetadataFilter] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve using dense and sparse strategies, then fuse results.
        """
        if not query or not query.strip():
            return []

        effective_top_k = top_k or self.top_k or settings.RETRIEVAL_TOP_K

        dense_results: List[RetrievedChunk] = []
        sparse_results: List[RetrievedChunk] = []

        dense_enabled = getattr(settings, "DENSE_RETRIEVAL_ENABLED", True)
        sparse_enabled = getattr(settings, "SPARSE_RETRIEVAL_ENABLED", True)

        if dense_enabled:
            try:
                with stage_timer("DENSE RETRIEVAL"):
                    dense_results = self.dense_retriever.retrieve(
                        query=query,
                        filters=filters,
                        top_k=effective_top_k,
                    )
            except Exception as exc:
                logger.error(
                    "Dense retrieval failed in hybrid pipeline: %s",
                    exc,
                )
                dense_results = []
        else:
            logger.debug("Dense retrieval disabled by configuration.")

        if sparse_enabled:
            try:
                with stage_timer("SPARSE RETRIEVAL"):
                    sparse_results = self.sparse_retriever.retrieve(
                        query=query,
                        filters=filters,
                        top_k=effective_top_k,
                    )
            except Exception as exc:
                logger.error(
                    "Sparse retrieval failed in hybrid pipeline: %s",
                    exc,
                )
                sparse_results = []
        else:
            logger.debug("Sparse retrieval disabled by configuration.")

        if not dense_results and not sparse_results:
            logger.warning(
                "Both dense and sparse retrieval failed or returned empty results."
            )
            return []

        result_sets = []
        if dense_results:
            result_sets.append(dense_results)
        if sparse_results:
            result_sets.append(sparse_results)

        return self.fusion.fuse(
            result_sets,
            top_k=effective_top_k,
        )