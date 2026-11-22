# src/retrieval/sparse_retriever.py
"""
Keyword-based (Sparse) Retriever using BM25.
"""
import logging
import re
from typing import List, Optional, Dict

from rank_bm25 import BM25Okapi

from src.domain.chunks import Chunk
from src.domain.queries import RetrievedChunk
from src.domain.exceptions import RetrievalError
from src.retrieval.metadata_filters import MetadataFilter
from src.preprocessing.arabic_normalizer import ArabicNormalizer, NormalizationConfig
from src.preprocessing.patterns import TATWEEL_RE
from src.config.settings import settings

logger = logging.getLogger(__name__)


# Function words that add noise to BM25 matching. Written in natural Arabic;
# they are passed through the SAME normalizer used for documents and queries,
# so folding rules (hamza, ta-marbuta, diacritics) always stay symmetric.
# NOTE: "المادة" is intentionally NOT a stopword — article-number queries matter.
_RAW_ARABIC_STOPWORDS = {
    # Question words
    "هل", "ما", "ماذا", "متى", "أين", "كيف", "لماذا", "كم", "أي",
    # Prepositions and connectors
    "في", "من", "على", "إلى", "عن", "مع", "بين", "عند", "حتى", "منذ",
    "ثم", "أو", "لكن", "بل", "كي", "سوف", "قد", "لقد",
    # Pronouns
    "هو", "هي", "هم", "هن", "أنا", "نحن", "أنت", "أنتم",
    # Demonstratives and relatives
    "هذا", "هذه", "ذلك", "تلك", "الذي", "التي", "الذين",
    # Auxiliaries, negation, quantifiers
    "كان", "كانت", "يكون", "تكون", "ليس", "ليست",
    "لا", "لم", "لن", "إن", "أن", "إذا", "لو",
    "كل", "بعض", "غير", "ذات", "نفس", "أيضا",
}


class SparseRetriever:
    """
    Retrieves chunks using BM25 keyword matching.

    Trade-offs: In-memory BM25 is extremely fast for development and moderate
    datasets, and preserves exact legal terminology without aggressive
    stemming. However, it does not scale to millions of documents without a
    dedicated engine (e.g., Elasticsearch).

    Arabic handling:
      - Normalization (hamza folding, digit folding, diacritics removal) is
        applied identically to the index and to queries.
      - Ta-marbuta folding is ENABLED here (unlike the global default) because
        symmetric folding strictly improves recall for queries such as
        "السرقه" vs "السرقة". Stored content and dense embeddings are untouched.
      - Tatweel and punctuation are stripped; single-character tokens and
        Arabic function words are dropped.
    """

    def __init__(self, top_k: Optional[int] = None, k1: Optional[float] = None, b: Optional[float] = None):
        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.k1 = k1 or settings.BM25_K1
        self.b = b or settings.BM25_B

        self.bm25: Optional[BM25Okapi] = None
        self.chunks_map: Dict[str, Chunk] = {}
        self.chunk_ids: List[str] = []

        # BM25-only normalization: fold ta-marbuta for better recall.
        self.normalizer = ArabicNormalizer(
            NormalizationConfig(fold_ta_marbuta=True)
        )

        # Normalize stopwords once with the exact same rules used at
        # index/query time to guarantee symmetric matching.
        self.stopwords = frozenset(
            self.normalizer.normalize_query(word) for word in _RAW_ARABIC_STOPWORDS
        )

    def build_index(self, chunks: List[Chunk]):
        """Builds the BM25 index from a list of chunks."""
        self.chunks_map.clear()
        self.chunk_ids.clear()
        self.bm25 = None

        if not chunks:
            logger.warning("No chunks provided to build BM25 index")
            return

        corpus = []
        for chunk in chunks:
            chunk_id = f"{chunk.document_id}_{chunk.chunk_index}"
            self.chunks_map[chunk_id] = chunk
            self.chunk_ids.append(chunk_id)

            # Normalize Arabic text and tokenize
            normalized = self.normalizer.normalize(chunk.content)
            tokens = self._tokenize(normalized.text_normalized)
            corpus.append(tokens)

        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        logger.info(
            "BM25 READY | chunks=%d | corpus=%d | vocabulary=%d",
            len(self.chunks_map),
            len(corpus),
            len(self.bm25.idf),
        )

    def _tokenize(self, text: str) -> List[str]:
        """
        Arabic-aware tokenization: strips tatweel and punctuation, drops
        single-character tokens and normalized stopwords. Legal terminology
        is preserved (no stemming).
        """
        text = TATWEEL_RE.sub("", text)
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [
            token for token in tokens
            if len(token) > 1 and token not in self.stopwords
        ]

    def retrieve(
        self,
        query: str,
        filters: Optional[MetadataFilter] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        if not self.bm25:
            raise RetrievalError("BM25 index not built. Call build_index first.", query=query)

        if not query or not query.strip():
            return []

        try:
            from src.retrieval.legal_synonyms import expand_query_with_synonyms
            expanded_query = expand_query_with_synonyms(query)
            normalized = self.normalizer.normalize(expanded_query)
            query_tokens = self._tokenize(normalized.text_normalized)

            logger.info(
                "BM25 QUERY | query=%s | tokens=%s",
                query,
                query_tokens,
            )

            if not query_tokens:
                logger.warning(
                    "BM25 query produced no tokens after normalization/stopword removal."
                )
                return []

            scores = self.bm25.get_scores(query_tokens)
            logger.info(
                "BM25 SCORES | max=%.5f | nonzero=%d",
                float(max(scores)) if len(scores) else 0.0,
                sum(1 for score in scores if score > 0),
            )

            # Get top_k indices
            effective_top_k = top_k or self.top_k or settings.RETRIEVAL_TOP_K
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:effective_top_k]

            results = []
            for idx in top_indices:
                if scores[idx] == 0:
                    continue

                chunk_id = self.chunk_ids[idx]
                chunk = self.chunks_map[chunk_id]

                # Apply in-memory filters
                if filters and not self._matches_filter(chunk, filters):
                    continue

                results.append(
                    RetrievedChunk(
                        chunk=chunk,
                        score=float(scores[idx]),
                        retriever_type="sparse",
                    )
                )
            return results

        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            raise RetrievalError(f"Sparse retrieval failed: {e}", query=query)

    def _matches_filter(self, chunk: Chunk, filters: MetadataFilter) -> bool:
        """Evaluates abstract filters against chunk metadata in-memory."""
        for cond in filters.conditions:
            val = chunk.metadata.get(cond.field)
            if cond.operator == "eq" and val != cond.value:
                return False
        return True