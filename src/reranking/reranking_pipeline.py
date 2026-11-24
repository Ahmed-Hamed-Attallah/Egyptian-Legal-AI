"""Reranking pipeline orchestrator."""

from __future__ import annotations

import logging
from typing import List, Optional

from src.config.retrieval_config import retrieval_config
from src.config.settings import settings
from src.domain.exceptions import RerankingError
from src.domain.queries import RetrievedChunk
from src.reranking.base import BaseReranker

logger = logging.getLogger(__name__)

# A spread this small means the cross-encoder produced no usable ranking signal.
DEGENERATE_SPREAD = 1e-3


class RerankingPipeline:
    """Delegate scoring, enforce MIN_RERANK_SCORE, limit output.

    Failure policy:
      * Reranker raises      → fall back to hybrid (RRF) order.
      * Reranker is degenerate (all scores identical) → fall back to RRF order,
        because an arbitrary shuffle is worse than the fusion ranking.
      * All candidates below threshold → return [] so the pipeline honestly
        reports insufficient evidence rather than citing irrelevant text.
    """

    def __init__(
        self,
        reranker: BaseReranker,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> None:
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
        if not query or not query.strip():
            logger.warning("Empty query provided to RerankingPipeline")
            return []
        if not candidates:
            return []

        limit = top_k or self.top_k

        try:
            reranked = self.reranker.rerank(query, candidates, top_k=None)
        except RerankingError as exc:
            logger.error("Reranking failed: %s. Falling back to hybrid order.", exc)
            return candidates[:limit]
        except Exception as exc:
            logger.error("Unexpected reranking error: %s", exc)
            raise RerankingError(f"Pipeline failed: {exc}") from exc

        if not reranked:
            logger.warning("Reranker returned no candidates; falling back to hybrid order.")
            return candidates[:limit]

        scores = [c.score for c in reranked]
        spread = max(scores) - min(scores)
        if len(reranked) > 1 and spread < DEGENERATE_SPREAD:
            logger.error(
                "RERANK DEGENERATE | candidates=%d | spread=%.6f | score=%.4f. "
                "Cross-encoder produced no ranking signal — using hybrid order. "
                "Check the activation contract in CrossEncoderReranker.",
                len(reranked), spread, scores[0],
            )
            return candidates[:limit]

        kept = [c for c in reranked if c.score >= self.min_score]
        dropped = len(reranked) - len(kept)
        if dropped:
            logger.info(
                "RERANK THRESHOLD | kept=%d | dropped=%d | min_score=%.3f | top=%.4f",
                len(kept), dropped, self.min_score, max(scores),
            )

        if not kept:
            logger.warning(
                "All %d candidates below min_score=%.3f (best=%.4f, spread=%.4f). "
                "Returning empty so the pipeline reports insufficient evidence.",
                len(reranked), self.min_score, max(scores), spread,
            )
            return []

        return kept[:limit]