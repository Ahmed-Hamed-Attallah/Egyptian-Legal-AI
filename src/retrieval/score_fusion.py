"""Score fusion for combining retrieval results."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from src.config.settings import settings
from src.domain.queries import RetrievedChunk

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion with preserved provenance and rank.

    `rank` is 1-based and must be set: `RetrievedChunk.to_dict_for_llm()` exposes
    it, and leaving it at 0 tells the model every chunk is equally ranked.
    `dense_score` / `sparse_score` are retained so retrieval quality is
    diagnosable (e.g. "dense_only" reveals a stale BM25 index).
    """

    def __init__(self, k: Optional[int] = None) -> None:
        self.k = k or settings.RRF_K

    @staticmethod
    def _key(chunk) -> str:
        """Must match SparseRetriever's key so the same chunk fuses correctly."""
        return f"{chunk.document_id}_{chunk.chunk_index}"

    def fuse(
        self,
        result_sets: List[List[RetrievedChunk]],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not result_sets:
            return []

        scores: Dict[str, float] = {}
        representative: Dict[str, RetrievedChunk] = {}
        dense: Dict[str, float] = {}
        sparse: Dict[str, float] = {}
        sources: Dict[str, Set[str]] = {}

        for results in result_sets:
            for rank, res in enumerate(results):
                key = self._key(res.chunk)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.k + rank + 1)
                representative.setdefault(key, res)

                kind = (res.retriever_type or "").lower()
                bucket = sources.setdefault(key, set())
                if "dense" in kind:
                    dense[key] = max(0.0, float(res.score))
                    bucket.add("dense")
                elif "sparse" in kind:
                    sparse[key] = max(0.0, float(res.score))
                    bucket.add("sparse")
                else:
                    bucket.add(kind or "unknown")

        ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
        limit = top_k or settings.RETRIEVAL_TOP_K

        fused: List[RetrievedChunk] = []
        for position, key in enumerate(ordered[:limit], start=1):
            res = representative[key]
            fused.append(
                RetrievedChunk(
                    chunk=res.chunk,
                    score=scores[key],
                    dense_score=dense.get(key),
                    sparse_score=sparse.get(key),
                    rank=position,
                    retriever_type="hybrid",
                    metadata={"fusion": "rrf", "sources": sorted(sources.get(key, ()))},
                )
            )

        window = ordered[:limit]
        both = sum(1 for k in window if len(sources.get(k, ())) > 1)
        dense_only = sum(1 for k in window if sources.get(k) == {"dense"})
        sparse_only = sum(1 for k in window if sources.get(k) == {"sparse"})
        logger.info(
            "RRF FUSION | candidates=%d | returned=%d | both=%d | dense_only=%d "
            "| sparse_only=%d | k=%d",
            len(scores), len(fused), both, dense_only, sparse_only, self.k,
        )
        return fused