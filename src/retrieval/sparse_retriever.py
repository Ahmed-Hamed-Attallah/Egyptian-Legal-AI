# src/retrieval/sparse_retriever.py
"""
Keyword-based (sparse) retriever using BM25.

Index lifecycle
---------------
The in-memory index can be mutated after startup: `add_chunks` and
`remove_document` let ingestion refresh it without a process restart. The
tokenized corpus is retained, so a rebuild costs only BM25 statistics — not
re-normalization of every chunk.

Arabic handling
---------------
Normalization (hamza folding, digit folding, diacritics removal) is applied
identically to index and queries. Ta-marbuta folding is ENABLED here (unlike the
global default) because symmetric folding improves recall for "السرقه" vs
"السرقة". Stored content and dense embeddings are untouched.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from src.config.settings import settings
from src.domain.chunks import Chunk
from src.domain.exceptions import RetrievalError
from src.domain.queries import RetrievedChunk
from src.preprocessing.arabic_normalizer import ArabicNormalizer, NormalizationConfig
from src.preprocessing.patterns import TATWEEL_RE
from src.retrieval.legal_synonyms import expand_query_with_synonyms
from src.retrieval.metadata_filters import MetadataFilter

logger = logging.getLogger(__name__)

# Function words that add noise to BM25 matching. Written in natural Arabic and
# passed through the SAME normalizer used for documents and queries, so folding
# rules stay symmetric. "المادة" is intentionally NOT a stopword — article-number
# queries matter.
_RAW_ARABIC_STOPWORDS = {
    "هل", "ما", "ماذا", "متى", "أين", "كيف", "لماذا", "كم", "أي",
    "في", "من", "على", "إلى", "عن", "مع", "بين", "عند", "حتى", "منذ",
    "ثم", "أو", "لكن", "بل", "كي", "سوف", "قد", "لقد",
    "هو", "هي", "هم", "هن", "أنا", "نحن", "أنت", "أنتم",
    "هذا", "هذه", "ذلك", "تلك", "الذي", "التي", "الذين",
    "كان", "كانت", "يكون", "تكون", "ليس", "ليست",
    "لا", "لم", "لن", "إن", "أن", "إذا", "لو",
    "كل", "بعض", "غير", "ذات", "نفس", "أيضا",
}


class SparseRetriever:
    """BM25 keyword retrieval with symmetric Arabic normalization."""

    def __init__(
        self,
        top_k: Optional[int] = None,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ) -> None:
        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.k1 = k1 or settings.BM25_K1
        self.b = b or settings.BM25_B

        self.bm25: Optional[BM25Okapi] = None
        self.chunks_map: Dict[str, Chunk] = {}
        self.chunk_ids: List[str] = []
        self._corpus: List[List[str]] = []

        # Guards against a rebuild mutating the corpus mid-query.
        self._lock = threading.RLock()

        self.normalizer = ArabicNormalizer(NormalizationConfig(fold_ta_marbuta=True))
        self.stopwords = frozenset(
            self.normalizer.normalize_query(word) for word in _RAW_ARABIC_STOPWORDS
        )

    # ------------------------------------------------------------------ state
    @property
    def indexed_count(self) -> int:
        """Number of chunks currently in the index. Read by the service factory."""
        return len(self.chunk_ids)

    @property
    def is_ready(self) -> bool:
        return self.bm25 is not None

    @property
    def vocabulary_size(self) -> int:
        return len(self.bm25.idf) if self.bm25 is not None else 0

    def __len__(self) -> int:
        return self.indexed_count

    @staticmethod
    def _key(chunk: Chunk) -> str:
        return f"{chunk.document_id}_{chunk.chunk_index}"

    def _tokenize(self, text: str) -> List[str]:
        text = TATWEEL_RE.sub("", text)
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 1 and t not in self.stopwords]

    def _encode(self, chunk: Chunk) -> List[str]:
        normalized = self.normalizer.normalize(chunk.content)
        return self._tokenize(normalized.text_normalized)

    def _rebuild_statistics(self) -> None:
        """Recompute BM25 statistics from the retained tokenized corpus."""
        self.bm25 = (
            BM25Okapi(self._corpus, k1=self.k1, b=self.b) if self._corpus else None
        )

    # ------------------------------------------------------------- index life
    def build_index(self, chunks: List[Chunk]) -> None:
        """Replace the whole index."""
        with self._lock:
            self.chunks_map.clear()
            self.chunk_ids.clear()
            self._corpus.clear()
            self.bm25 = None

            if not chunks:
                logger.warning("No chunks provided to build BM25 index")
                return

            for chunk in chunks:
                key = self._key(chunk)
                if key in self.chunks_map:
                    continue
                self.chunks_map[key] = chunk
                self.chunk_ids.append(key)
                self._corpus.append(self._encode(chunk))

            self._rebuild_statistics()
            logger.info(
                "BM25 READY | chunks=%d | corpus=%d | vocabulary=%d",
                self.indexed_count, len(self._corpus), self.vocabulary_size,
            )

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Add newly ingested chunks and refresh statistics. Returns count added.

        Without this, documents uploaded after startup are invisible to sparse
        retrieval until the process restarts.
        """
        if not chunks:
            return 0
        with self._lock:
            added = 0
            for chunk in chunks:
                key = self._key(chunk)
                if key in self.chunks_map:
                    continue
                self.chunks_map[key] = chunk
                self.chunk_ids.append(key)
                self._corpus.append(self._encode(chunk))
                added += 1
            if added:
                self._rebuild_statistics()
                logger.info(
                    "BM25 INCREMENTAL | added=%d | total=%d", added, self.indexed_count
                )
            return added

    def remove_document(self, document_id: str) -> int:
        """Drop every chunk of a document (re-ingest or deletion). Returns count."""
        target = str(document_id)
        with self._lock:
            keep = [
                i for i, key in enumerate(self.chunk_ids)
                if str(self.chunks_map[key].document_id) != target
            ]
            removed = len(self.chunk_ids) - len(keep)
            if not removed:
                return 0
            self.chunk_ids = [self.chunk_ids[i] for i in keep]
            self._corpus = [self._corpus[i] for i in keep]
            self.chunks_map = {key: self.chunks_map[key] for key in self.chunk_ids}
            self._rebuild_statistics()
            logger.info(
                "BM25 REMOVED | document=%s | removed=%d | total=%d",
                target, removed, self.indexed_count,
            )
            return removed

    # ---------------------------------------------------------------- retrieval
    def retrieve(
        self,
        query: str,
        filters: Optional[MetadataFilter] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not query or not query.strip():
            return []

        with self._lock:
            if not self.bm25:
                raise RetrievalError(
                    "BM25 index not built. Call build_index first.", query=query
                )
            try:
                expanded = expand_query_with_synonyms(query)
                normalized = self.normalizer.normalize(expanded)
                query_tokens = self._tokenize(normalized.text_normalized)
                logger.info("BM25 QUERY | query=%s | tokens=%s", query, query_tokens)

                if not query_tokens:
                    logger.warning(
                        "BM25 query produced no tokens after normalization/stopword removal."
                    )
                    return []

                scores = self.bm25.get_scores(query_tokens)
                logger.info(
                    "BM25 SCORES | max=%.5f | nonzero=%d",
                    float(max(scores)) if len(scores) else 0.0,
                    sum(1 for s in scores if s > 0),
                )

                effective_top_k = top_k or self.top_k or settings.RETRIEVAL_TOP_K
                ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

                results: List[RetrievedChunk] = []
                filtered_out = 0
                for idx in ranked:
                    if scores[idx] <= 0:
                        break  # ranked descending: everything after is zero too
                    chunk = self.chunks_map[self.chunk_ids[idx]]
                    # Single shared evaluator — honours and/or and all operators.
                    if filters is not None and not filters.matches(chunk.metadata):
                        filtered_out += 1
                        continue
                    results.append(
                        RetrievedChunk(
                            chunk=chunk,
                            score=float(scores[idx]),
                            retriever_type="sparse",
                        )
                    )
                    if len(results) >= effective_top_k:
                        break

                if filtered_out:
                    logger.info(
                        "BM25 FILTER | dropped=%d | filter=%s",
                        filtered_out, filters.describe() if filters else "*",
                    )
                return results

            except RetrievalError:
                raise
            except Exception as exc:
                logger.error("Sparse retrieval failed: %s", exc)
                raise RetrievalError(f"Sparse retrieval failed: {exc}", query=query) from exc