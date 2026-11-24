"""Cross-Encoder Reranker.

Activation contract
-------------------
`CrossEncoder.predict()` applies `default_activation_function`, which
sentence-transformers sets to `nn.Sigmoid()` for single-label models such as
BAAI/bge-reranker-base. Its output is therefore ALREADY a probability.
We request the identity activation so `predict()` returns raw logits, then apply
exactly one numerically-stable sigmoid here.

A sigmoid on top of an already-activated output compresses every score into
[0.5, 0.622], defeating MIN_RERANK_SCORE and making irrelevant chunks look ~50%
relevant. Raw logits are logged so the score range stays observable.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from src.config.settings import settings
from src.domain.exceptions import RerankingError
from src.domain.queries import RetrievedChunk
from src.reranking.base import BaseReranker

logger = logging.getLogger(__name__)

DEGENERATE_LOGIT_SPREAD = 0.1


def _identity(logits):
    """Pass-through activation: keep output as raw logits."""
    return logits


class CrossEncoderReranker(BaseReranker):
    """Rerank candidates with a multilingual cross-encoder."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or settings.RERANKING_MODEL_NAME
        self.device = device or settings.RERANKING_DEVICE
        self.batch_size = batch_size or settings.RERANKING_BATCH_SIZE

        logger.info("Loading cross-encoder model: %s on %s", self.model_name, self.device)
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
        except Exception as exc:
            logger.error("Failed to load cross-encoder model: %s", exc)
            raise RerankingError(
                f"Model loading failed: {exc}", model_name=self.model_name
            ) from exc

        logger.info(
            "Cross-encoder ready | default_activation=%s (bypassed; one sigmoid here)",
            getattr(self.model, "default_activation_function", None),
        )

    @staticmethod
    def _sigmoid(logits) -> np.ndarray:
        x = np.asarray(logits, dtype=np.float64).reshape(-1)
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        exp_x = np.exp(x[~pos])
        out[~pos] = exp_x / (1.0 + exp_x)
        return out

    def _predict_logits(self, pairs: List[List[str]]) -> np.ndarray:
        try:
            raw = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                activation_fct=_identity,
            )
            return np.asarray(raw, dtype=np.float64).reshape(-1)
        except TypeError:
            logger.warning(
                "CrossEncoder.predict() does not accept activation_fct; "
                "falling back to default activation and detecting scale."
            )

        raw = np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False),
            dtype=np.float64,
        ).reshape(-1)

        if raw.size and float(raw.min()) >= 0.0 and float(raw.max()) <= 1.0:
            logger.warning(
                "Reranker output is probability-scaled; inverting to logits so "
                "exactly one sigmoid is applied."
            )
            clipped = np.clip(raw, 1e-9, 1 - 1e-9)
            return np.log(clipped / (1 - clipped))
        return raw

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
            raw_logits = self._predict_logits(pairs)

            if raw_logits.size != len(candidates):
                raise RerankingError(
                    f"Reranker returned {raw_logits.size} scores for "
                    f"{len(candidates)} candidates.",
                    model_name=self.model_name,
                    chunk_count=len(candidates),
                )

            probabilities = self._sigmoid(raw_logits)
            spread = float(raw_logits.max() - raw_logits.min())

            logger.info(
                "RERANK RAW | logit_min=%.4f | logit_max=%.4f | spread=%.4f "
                "| prob_min=%.4f | prob_max=%.4f",
                float(raw_logits.min()), float(raw_logits.max()), spread,
                float(probabilities.min()), float(probabilities.max()),
            )
            if len(candidates) > 1 and spread < DEGENERATE_LOGIT_SPREAD:
                logger.error(
                    "RERANK NO SIGNAL | logit_spread=%.4f across %d candidates.",
                    spread, len(candidates),
                )

            scored: List[RetrievedChunk] = []
            for candidate, probability in zip(candidates, probabilities):
                scored.append(
                    RetrievedChunk(
                        chunk=candidate.chunk,
                        score=float(probability),
                        rerank_score=float(probability),
                        dense_score=candidate.dense_score,
                        sparse_score=candidate.sparse_score,
                        retriever_type=f"{candidate.retriever_type}_reranked",
                        metadata=dict(candidate.metadata),
                    )
                )

            scored.sort(key=lambda c: c.score, reverse=True)
            # rank is exposed to the LLM via to_dict_for_llm(); it must be set.
            for position, item in enumerate(scored, start=1):
                item.rank = position

            logger.info(
                "RERANK COMPLETE | candidates=%d | top_score=%.4f | bottom_score=%.4f",
                len(scored), scored[0].score, scored[-1].score,
            )
            return scored[:top_k] if top_k else scored

        except RerankingError:
            raise
        except Exception as exc:
            logger.error("Reranking failed: %s", exc)
            raise RerankingError(
                f"Reranking failed: {exc}",
                model_name=self.model_name,
                chunk_count=len(candidates),
            ) from exc