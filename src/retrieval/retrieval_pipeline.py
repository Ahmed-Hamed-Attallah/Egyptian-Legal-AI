# src/retrieval/retrieval_pipeline.py
"""
Retrieval pipeline: index lifecycle + single retrieval entry point.

Why this module exists
----------------------
`HybridRetriever` already fuses dense + sparse, and `RerankingPipeline` already
scores candidates. What was missing was ownership of the SPARSE INDEX LIFECYCLE:
`factory.py` inlined `scroll_all()` + `build_index()` once at startup, so chunks
ingested afterwards were invisible to BM25 until a restart.

This module owns:
  * building / rebuilding the BM25 index from Qdrant
  * detecting corpus growth and refreshing automatically before retrieval
  * a single `retrieve()` facade (hybrid → optional rerank)
  * index statistics for health/admin surfaces
  * a CLI so retrieval can be exercised without the API

It does NOT reimplement normalization, fusion, or scoring.

CLI
---
    python -m src.retrieval.retrieval_pipeline "ما حكم المادة ١٤٧؟"
    python -m src.retrieval.retrieval_pipeline --stats
    python -m src.retrieval.retrieval_pipeline --rebuild
    python -m src.retrieval.retrieval_pipeline "عقوبة السرقة" --no-rerank --top-k 8
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config.settings import settings
from src.domain.chunks import Chunk
from src.domain.queries import RetrievedChunk
from src.monitoring.pipeline_monitor import stage_timer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.metadata_filters import FilterCondition, MetadataFilter
from src.retrieval.sparse_retriever import SparseRetriever

logger = logging.getLogger(__name__)

# Minimum wall-clock gap between Qdrant corpus-size checks.
_FRESHNESS_CHECK_INTERVAL_SECONDS = 5.0


@dataclass
class IndexStats:
    """Snapshot of sparse-index state versus the vector store."""

    indexed_chunks: int
    qdrant_points: Optional[int]
    vocabulary_size: int
    stale: bool
    last_build_seconds: Optional[float]
    last_build_at: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "indexed_chunks": self.indexed_chunks,
            "qdrant_points": self.qdrant_points,
            "vocabulary_size": self.vocabulary_size,
            "stale": self.stale,
            "last_build_seconds": self.last_build_seconds,
            "last_build_at": self.last_build_at,
        }


class RetrievalPipeline:
    """Owns the sparse index and exposes one retrieval entry point."""

    def __init__(
        self,
        vector_store: Any,
        sparse_retriever: SparseRetriever,
        hybrid_retriever: HybridRetriever,
        reranking_pipeline: Optional[Any] = None,
        freshness_check_interval: float = _FRESHNESS_CHECK_INTERVAL_SECONDS,
    ) -> None:
        self.vector_store = vector_store
        self.sparse_retriever = sparse_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranking_pipeline = reranking_pipeline
        self._freshness_interval = freshness_check_interval

        self._lock = threading.RLock()
        self._last_check_at = 0.0
        self._last_build_seconds: Optional[float] = None
        self._last_build_at: Optional[float] = None
        self._known_qdrant_points: Optional[int] = None

    # ------------------------------------------------------------------
    # Corpus introspection
    # ------------------------------------------------------------------
    def qdrant_point_count(self) -> Optional[int]:
        """Exact point count, or None when the store cannot report it."""
        try:
            client = getattr(self.vector_store, "client", None)
            collection = getattr(self.vector_store, "collection_name", None)
            if client is None or collection is None:
                return None
            result = client.count(collection_name=collection, exact=True)
            return int(getattr(result, "count", result))
        except Exception as exc:
            logger.warning("Could not read Qdrant point count: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------
    def build_index(self) -> IndexStats:
        """Full rebuild of the BM25 index from every chunk in Qdrant."""
        with self._lock:
            with stage_timer("BM25 INDEX BUILD"):
                started = time.perf_counter()
                chunks: List[Chunk] = self.vector_store.scroll_all(
                    page_size=256,
                    max_points=100000,
                )
                self.sparse_retriever.build_index(chunks)
                self._last_build_seconds = time.perf_counter() - started
                self._last_build_at = time.time()
                self._known_qdrant_points = self.qdrant_point_count()
                self._last_check_at = time.time()

            logger.info(
                "BM25 index built from %d Qdrant chunks in %.2f s",
                self.sparse_retriever.indexed_count,
                self._last_build_seconds,
            )
            return self.stats()

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Incrementally index freshly ingested chunks (no full re-normalization)."""
        if not chunks:
            return 0
        with self._lock:
            written = self.sparse_retriever.add_chunks(chunks)
            self._known_qdrant_points = self.qdrant_point_count()
            self._last_check_at = time.time()
            return written

    def remove_document(self, document_id: str) -> int:
        """Drop a document's chunks from the sparse index."""
        with self._lock:
            removed = self.sparse_retriever.remove_document(document_id)
            self._known_qdrant_points = self.qdrant_point_count()
            self._last_check_at = time.time()
            return removed

    def ensure_fresh(self, force: bool = False) -> bool:
        """Rebuild the sparse index when the Qdrant corpus has changed.

        Cheap by design: at most one `count()` round trip per freshness window.
        Returns True when a rebuild happened.
        """
        if not getattr(settings, "SPARSE_RETRIEVAL_ENABLED", True):
            return False

        with self._lock:
            now = time.time()
            if not force and (now - self._last_check_at) < self._freshness_interval:
                return False
            self._last_check_at = now

            current = self.qdrant_point_count()
            indexed = self.sparse_retriever.indexed_count

            if current is None:
                # Cannot verify; build once if we have nothing at all.
                if indexed == 0:
                    self.build_index()
                    return True
                return False

            if current == indexed and indexed > 0:
                self._known_qdrant_points = current
                return False

            logger.info(
                "Sparse index stale | qdrant=%d | indexed=%d -> rebuilding",
                current, indexed,
            )
            self.build_index()
            return True

    def stats(self) -> IndexStats:
        indexed = self.sparse_retriever.indexed_count
        known = self._known_qdrant_points
        bm25 = getattr(self.sparse_retriever, "bm25", None)
        vocabulary = len(bm25.idf) if bm25 is not None else 0
        return IndexStats(
            indexed_chunks=indexed,
            qdrant_points=known,
            vocabulary_size=vocabulary,
            stale=(known is not None and known != indexed),
            last_build_seconds=self._last_build_seconds,
            last_build_at=self._last_build_at,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    @staticmethod
    def build_filter(
        tenant_id: Optional[str] = None,
        document_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[MetadataFilter]:
        """Compose an AND filter from common scoping fields."""
        conditions: List[FilterCondition] = []
        if tenant_id:
            conditions.append(FilterCondition(field="tenant_id", value=tenant_id))
        if document_id:
            conditions.append(FilterCondition(field="document_id", value=str(document_id)))
        for key, value in (extra or {}).items():
            conditions.append(FilterCondition(field=key, value=value))
        return MetadataFilter(conditions=conditions) if conditions else None

    def retrieve(
        self,
        query: str,
        filters: Optional[MetadataFilter] = None,
        top_k: Optional[int] = None,
        rerank: bool = True,
    ) -> List[RetrievedChunk]:
        """Hybrid retrieval, optionally reranked, with a fresh sparse index."""
        if not query or not query.strip():
            return []

        self.ensure_fresh()

        candidate_k = top_k or settings.RETRIEVAL_TOP_K
        with stage_timer("HYBRID RETRIEVAL"):
            candidates = self.hybrid_retriever.retrieve(
                query=query, filters=filters, top_k=candidate_k
            )

        if not candidates:
            logger.warning("Retrieval returned no candidates.")
            return []

        use_rerank = (
            rerank
            and self.reranking_pipeline is not None
            and getattr(settings, "RERANKER_ENABLED", True)
        )
        if not use_rerank:
            return candidates[:candidate_k]

        with stage_timer("RERANKING"):
            return self.reranking_pipeline.process(query, candidates, top_k=top_k)


# ---------------------------------------------------------------------------
# Terminal entry point
# ---------------------------------------------------------------------------
def _build_standalone_pipeline(with_reranker: bool) -> RetrievalPipeline:
    """Construct the pipeline without importing the service factory (no cycle)."""
    from src.embeddings.model_registry import get_default_embedding_model
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.score_fusion import ReciprocalRankFusion
    from src.vector_store.qdrant_store import QdrantVectorStore

    vector_store = QdrantVectorStore()
    sparse = SparseRetriever()
    dense = DenseRetriever(
        embedding_model=get_default_embedding_model(),
        vector_store=vector_store,
    )
    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        fusion=ReciprocalRankFusion(),
    )

    reranking = None
    if with_reranker and getattr(settings, "RERANKER_ENABLED", True):
        from src.reranking.cross_encoder_reranker import CrossEncoderReranker
        from src.reranking.reranking_pipeline import RerankingPipeline

        reranking = RerankingPipeline(reranker=CrossEncoderReranker())

    return RetrievalPipeline(
        vector_store=vector_store,
        sparse_retriever=sparse,
        hybrid_retriever=hybrid,
        reranking_pipeline=reranking,
    )


def _main() -> int:
    import argparse
    import json

    from src.config.logging_config import setup_logging

    parser = argparse.ArgumentParser(
        description="Inspect Egyptian Legal AI retrieval from the terminal."
    )
    parser.add_argument("query", nargs="?", help="Question or keywords to retrieve for.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Force a full BM25 rebuild.")
    parser.add_argument("--stats", action="store_true", help="Print index stats and exit.")
    parser.add_argument("--tenant", default=None)
    parser.add_argument("--document", default=None)
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = parser.parse_args()

    setup_logging()

    pipeline = _build_standalone_pipeline(with_reranker=not args.no_rerank)
    pipeline.build_index()

    if args.stats or not args.query:
        print(json.dumps(pipeline.stats().as_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.rebuild:
        pipeline.build_index()

    results = pipeline.retrieve(
        query=args.query,
        filters=pipeline.build_filter(tenant_id=args.tenant, document_id=args.document),
        top_k=args.top_k,
        rerank=not args.no_rerank,
    )

    if args.json:
        print(json.dumps(
            [
                {
                    "score": r.score,
                    "document_id": str(r.chunk.document_id),
                    "chunk_index": r.chunk.chunk_index,
                    "article_number": r.chunk.article_number,
                    "law_number": r.chunk.metadata.get("law_number"),
                    "law_year": r.chunk.metadata.get("law_year"),
                    "content": r.chunk.content,
                }
                for r in results
            ],
            indent=2, ensure_ascii=False,
        ))
        return 0

    print(f"\n{len(results)} result(s) for: {args.query}\n" + "=" * 72)
    for i, r in enumerate(results, 1):
        meta = r.chunk.metadata
        label = (
            f"law {meta.get('law_number')}/{meta.get('law_year')} "
            f"article {r.chunk.article_number}"
        )
        print(f"\n[{i}] score={r.score:.4f}  {label}")
        print(f"    doc={r.chunk.document_id} chunk={r.chunk.chunk_index}")
        print(f"    {r.chunk.content[:240]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())