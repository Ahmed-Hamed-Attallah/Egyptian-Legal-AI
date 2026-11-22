# src/services/search_service.py
"""
Search service.

Reuses existing HybridRetriever and optional RerankingPipeline.
Does not implement retrieval logic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.retrieval.hybrid_retriever import HybridRetriever

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
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        candidate_k = top_k
        if use_reranking:
            candidate_k = min(max(top_k * 3, top_k), 100)

        retrieved = self.hybrid_retriever.retrieve(query, top_k=candidate_k)

        if use_reranking and self.reranker is not None:
            retrieved = self._maybe_rerank(query, retrieved, top_k)

        results = [self._serialize_retrieved_chunk(rc) for rc in retrieved[:top_k]]

        if tenant_id:
            results = [
                item
                for item in results
                if self._tenant_allowed(item.get("metadata", {}), tenant_id)
            ]

        return results

    def _maybe_rerank(self, query: str, chunks: List[Any], top_k: int) -> List[Any]:
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
                getattr(chunk, "id", None)
                or getattr(chunk, "chunk_id", None)
                or f"{document_id}:{chunk_index}"
            )

        score = getattr(retrieved_chunk, "score", None)
        if score is None:
            score = getattr(chunk, "score", None)

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
            "metadata": metadata,
        }

    @staticmethod
    def _tenant_allowed(metadata: Dict[str, Any], tenant_id: str) -> bool:
        doc_tenant = metadata.get("tenant_id")
        return doc_tenant in (None, "", "default", tenant_id)