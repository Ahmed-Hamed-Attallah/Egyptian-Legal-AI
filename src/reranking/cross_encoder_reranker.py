# src/reranking/cross_encoder_reranker.py
"""
Cross-Encoder Reranker implementation.

Raw cross-encoder output is an unbounded logit (often negative for irrelevant
pairs). This module normalizes logits into 0.0-1.0 probabilities with a
sigmoid so that scores:

1. Satisfy the ``RetrievedChunk.score`` contract (Pydantic ``ge=0.0``).
   Without this, any negative logit raised a ValidationError and the whole
   reranking stage silently fell back to unranked RRF scores (~0.01), which
   surfaced in the UI as "0.0% relevance".
2. Become interpretable for thresholding (MIN_RERANK_SCORE) and for
   relevance percentages in the frontend.
"""
import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from src.reranking.base import BaseReranker
from src.domain.queries import RetrievedChunk
from src.domain.exceptions import RerankingError
from src.config.settings import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """
    Reranks candidates using a multilingual Cross-Encoder.

    Note: While multilingual models (e.g., ms-marco-multilingual-MiniLM) provide
    a strong baseline for Arabic, they may not capture highly specialized Egyptian
    legal terminology perfectly. Model selection is fully configurable to allow
    future fine-tuning or replacement with a domain-specific model.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        self.model_name = model_name or settings.RERANKING_MODEL_NAME
        self.device = device or settings.RERANKING_DEVICE
        self.batch_size = batch_size or settings.RERANKING_BATCH_SIZE

        logger.info(f"Loading cross-encoder model: {self.model_name} on {self.device}")
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise RerankingError(f"Model loading failed: {e}", model_name=self.model_name)

    @staticmethod
    def _sigmoid(logits) -> np.ndarray:
        """Numerically stable sigmoid: maps unbounded logits to (0, 1)."""
        x = np.asarray(logits, dtype=np.float64)
        out = np.empty_like(x)
        positive = x >= 0
        out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
        exp_x = np.exp(x[~positive])
        out[~positive] = exp_x / (1.0 + exp_x)
        return out

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        try:
            pairs = [[query, c.chunk.content] for c in candidates]
            raw_logits = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            probabilities = self._sigmoid(raw_logits)

            scored_candidates: List[RetrievedChunk] = []
            for candidate, logit, probability in zip(candidates, raw_logits, probabilities):
                reranked = RetrievedChunk(
                    chunk=candidate.chunk,
                    score=float(probability),
                    rerank_score=float(probability),
                    dense_score=candidate.dense_score,
                    sparse_score=candidate.sparse_score,
                    retriever_type=f"{candidate.retriever_type}_reranked",
                )
                scored_candidates.append(reranked)
                logger.debug(
                    "RERANK SCORE | logit=%.4f | probability=%.4f | article=%s",
                    float(logit),
                    float(probability),
                    candidate.chunk.article_number,
                )

            scored_candidates.sort(key=lambda c: c.score, reverse=True)

            logger.info(
                "RERANK COMPLETE | candidates=%d | top_score=%.4f | bottom_score=%.4f",
                len(scored_candidates),
                scored_candidates[0].score,
                scored_candidates[-1].score,
            )

            if top_k:
                scored_candidates = scored_candidates[:top_k]
            return scored_candidates

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise RerankingError(
                f"Reranking failed: {e}",
                model_name=self.model_name,
                chunk_count=len(candidates),
            )