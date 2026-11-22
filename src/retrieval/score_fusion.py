# src/retrieval/score_fusion.py
"""
Score fusion strategies for combining retrieval results.
"""
import logging
from typing import List
from src.domain.queries import RetrievedChunk
from src.config.settings import settings

logger = logging.getLogger(__name__)

class ReciprocalRankFusion:
    """
    Combines multiple ranked lists using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, k: int = None):
        self.k = k or settings.RRF_K

    
    def fuse(
        self,
        result_sets: List[List[RetrievedChunk]],
        top_k: int = None,
    ) -> List[RetrievedChunk]:

        if not result_sets:
            return []

        scores = {}
        chunk_map = {}

        for results in result_sets:
            for rank, res in enumerate(results):
                chunk_id = f"{res.chunk.document_id}_{res.chunk.chunk_index}"

                if chunk_id not in scores:
                    scores[chunk_id] = 0.0
                    chunk_map[chunk_id] = res

                scores[chunk_id] += 1.0 / (self.k + rank + 1)

        sorted_ids = sorted(
            scores,
            key=lambda x: scores[x],
            reverse=True,
        )

        limit = top_k or settings.RETRIEVAL_TOP_K

        fused_results = []

        for chunk_id in sorted_ids[:limit]:
            res = chunk_map[chunk_id]

            fused_results.append(
                RetrievedChunk(
                    chunk=res.chunk,
                    score=scores[chunk_id],
                    retriever_type="hybrid",
                )
            )

        return fused_results
