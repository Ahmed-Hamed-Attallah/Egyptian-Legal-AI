# src/reranking/reranking_pipeline.py
"""
Reranking Pipeline Orchestrator.
"""
import logging
from typing import List, Optional

from src.reranking.base import BaseReranker
from src.domain.queries import RetrievedChunk
from src.domain.exceptions import RerankingError
from src.config.retrieval_config import retrieval_config
from src.config.settings import settings

logger = logging.getLogger(__name__)


class RerankingPipeline:
    """
    Orchestrates the reranking process.

    Responsibilities:
      - Delegate scoring to the wrapped reranker.
      - Enforce MIN_RERANK_SCORE so irrelevant chunks never reach the LLM.
      - Limit final output to ``top_k`` (per-call) or RERANKER_TOP_K (default).
      - Fall back to original candidate order if the reranker fails.
    """

    def __init__(
        self,
        reranker: BaseReranker,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ):
        self.reranker = reranker
        self.top_k = top_k or settings.RERANKER_TOP_K
        self.min_score = (
            min_score if min_score is not None else retrieval_config.MIN_RERANK_SCORE
        )

    def process(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """
        Rerank candidates and return the final evidence set.

        Args:
            query: The user's question.
            candidates: Wide candidate pool from hybrid retrieval.
            top_k: Optional per-call override for the final chunk count.
                   Falls back to ``self.top_k`` (settings.RERANKER_TOP_K).
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to RerankingPipeline")
            return []

        if not candidates:
            return []

        limit = top_k or self.top_k

        try:
            reranked = self.reranker.rerank(query, candidates, top_k=None)
        except RerankingError as e:
            logger.error(f"Reranking pipeline failed: {e}. Falling back to original scores.")
            return candidates[:limit]
        except Exception as e:
            logger.error(f"Unexpected error in reranking pipeline: {e}")
            raise RerankingError(f"Pipeline failed: {e}")

        kept = [c for c in reranked if c.score >= self.min_score]
        dropped = len(reranked) - len(kept)

        if dropped:
            logger.info(
                "RERANK THRESHOLD | kept=%d | dropped=%d | min_score=%.3f",
                len(kept),
                dropped,
                self.min_score,
            )

        if not kept:
            logger.warning(
                "All %d reranked candidates fell below min_score=%.3f. "
                "Returning empty result so the pipeline reports insufficient evidence.",
                len(reranked),
                self.min_score,
            )
            return []

        return kept[:limit]