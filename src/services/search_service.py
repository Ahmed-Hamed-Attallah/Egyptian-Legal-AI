# src/services/search_service.py
"""
Search service.

Reuses HybridRetriever and the optional RerankingPipeline. Visibility is pushed
down as a MetadataFilter so dense AND sparse retrieval are scoped identically.

The previous implementation retrieved unfiltered and post-filtered with
`doc_tenant in (None, "", "default", tenant_id)`, which let missing tenant
metadata and the "default" tenant pass for ANY caller. That is removed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.domain.queries import QueryScope
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.metadata_filters import MetadataFilter
from src.security.authentication import AuthContext

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Optional[Any] = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_reranking: bool = True,
        tenant_id: Optional[str] = None,
        auth: Optional[AuthContext] = None,
        scope: QueryScope = QueryScope.ALL_SOURCES,
    ) -> List[Dict[str, Any]]:
        """Search the official corpus plus the caller's own uploads.

        Pass `auth` whenever available. Falling back to `tenant_id` alone yields
        official-only results (fail closed) rather than exposing private uploads.
        """
        if not query or not query.strip():
            return []

        effective_tenant = (auth.tenant_id if auth is not None else tenant_id)
        if not effective_tenant:
            logger.warning("Search called without tenant context; returning nothing.")
            return []

        visibility = MetadataFilter.for_documents(
            tenant_id=effective_tenant,
            user_id=auth.user_id if auth is not None else None,
            scope=scope,
            is_admin=auth.is_admin() if auth is not None else False,
        )

        candidate_k = min(max(top_k * 3, top_k), 100) if use_reranking else top_k
        retrieved = self.hybrid_retriever.retrieve(
            query, filters=visibility, top_k=candidate_k
        )

        if use_reranking and self.reranker is not None:
            retrieved = self._maybe_rerank(query, retrieved, top_k)

        return [self._serialize_retrieved_chunk(rc) for rc in retrieved[:top_k]]

    def _maybe_rerank(self, query: str, chunks: List[Any], top_k: int) -> List[Any]:
        if not chunks:
            return []
        for method_name in ("process", "rerank", "rank"):
            method = getattr(self.reranker, method_name, None)
            if not callable(method):
                continue
            try:
                return method(query, chunks, top_k)
            except TypeError:
                try:
                    return method(query, chunks)
                except Exception as exc:
                    logger.warning("Reranking failed via %s: %s", method_name, exc)
                    return chunks
            except Exception as exc:
                logger.warning("Reranking failed via %s: %s", method_name, exc)
                return chunks
        return chunks

    @staticmethod
    def _serialize_retrieved_chunk(retrieved_chunk: Any) -> Dict[str, Any]:
        chunk = getattr(retrieved_chunk, "chunk", retrieved_chunk)
        if isinstance(chunk, dict):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {}) or {}
            document_id = chunk.get("document_id") or metadata.get("document_id")
            chunk_index = chunk.get("chunk_index") or metadata.get("chunk_index")
            article_number = chunk.get("article_number") or metadata.get("article_number")
            section_title = chunk.get("section_title") or metadata.get("section_title")
            chunk_id = chunk.get("id") or chunk.get("chunk_id") or f"{document_id}:{chunk_index}"
        else:
            content = getattr(chunk, "content", "")
            metadata = getattr(chunk, "metadata", {}) or {}
            document_id = getattr(chunk, "document_id", metadata.get("document_id"))
            chunk_index = getattr(chunk, "chunk_index", metadata.get("chunk_index"))
            article_number = getattr(chunk, "article_number", metadata.get("article_number"))
            section_title = getattr(chunk, "section_title", metadata.get("section_title"))
            chunk_id = (
                getattr(chunk, "id", None) or getattr(chunk, "chunk_id", None)
                or f"{document_id}:{chunk_index}"
            )

        score = getattr(retrieved_chunk, "score", None)
        if score is None:
            score = getattr(chunk, "score", None)

        scope = str(metadata.get("scope") or "private").lower()
        return {
            "chunk_id": str(chunk_id) if chunk_id is not None else None,
            "document_id": str(document_id) if document_id is not None else None,
            "chunk_index": chunk_index,
            "content": str(content or ""),
            "score": float(score) if score is not None else None,
            "article_number": str(article_number) if article_number is not None else None,
            "section_title": str(section_title) if section_title is not None else None,
            "law_name": metadata.get("law_name") or metadata.get("title"),
            "law_number": metadata.get("law_number"),
            "law_year": metadata.get("law_year"),
            "scope": scope,
            "is_official": scope == "official",
            "owner_id": metadata.get("owner_id"),
            "metadata": metadata,
        }