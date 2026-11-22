# src/generation/citation_formatter.py
"""
Citation formatting and enrichment — single source of truth.

Every value emitted here is copied or derived from an actually retrieved
chunk (its content, its metadata, or its original CSV row). Nothing is
invented. Missing pinpoints are derived from the retrieved text itself
(e.g. "مادة 1", "القانون رقم 25 لسنة 1920"); if they cannot be determined
they are left empty rather than guessed.

This module is used by:
  - CitationFormatter (RAG pipeline)
  - src/services/chat_service.py  (before PostgreSQL persistence)
  - api/serializers.py            (before the HTTP response)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.domain.citations import Citation, CitationConfidence, CitationType
from src.domain.queries import RetrievedChunk

logger = logging.getLogger(__name__)

_ARTICLE_RE = re.compile(r"مادة\s*[\(\[\"]?\s*([0-9٠-٩]+)\s*[\)\]\"]?\s*(مكرر)?", re.UNICODE)
_LAW_RE = re.compile(
    r"قانون\s*(?:رقم)?\s*[\(\[\"]?\s*([0-9٠-٩]+)\s*[\)\]\"]?\s*لسنة\s*([0-9٠-٩]{4})",
    re.UNICODE,
)
_QUOTED_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")
_AR_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_WS_RE = re.compile(r"\s+")

MAX_CITATIONS = 10

_FIELD_SOURCES: Dict[str, tuple] = {
    "article_number": ("article_number",),
    "section_title": ("section_title", "category"),
    "law_name": ("law_name", "title", "law_title"),
    "law_number": ("law_number",),
    "law_year": ("law_year", "year"),
    "legal_version": ("legal_version", "version_label"),
    "effective_date": ("effective_date",),
    "amendment_info": ("amendment_info", "amendments"),
    "source_document": (
        "source_document", "source_path", "file_path", "storage_key", "file_name",
    ),
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _first(*values: Any) -> Optional[Any]:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _digits(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value).translate(_AR_TO_LATIN).strip() or None


def _clean_label(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        parts = [a or b for a, b in _QUOTED_RE.findall(text)]
        if parts:
            text = "، ".join(parts)
    return text.replace("_", " ").strip() or None


def _norm_text(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "")).strip()


def _read(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _unwrap(item: Any) -> Any:
    """RetrievedChunk -> Chunk. Plain Chunk/dict passes through."""
    inner = _read(item, "chunk")
    return inner if inner is not None else item


def _metadata(item: Any) -> Dict[str, Any]:
    value = _read(_unwrap(item), "metadata")
    return value if isinstance(value, dict) else {}


def _row(item: Any) -> Dict[str, Any]:
    value = _metadata(item).get("raw_row_data")
    return value if isinstance(value, dict) else {}


def chunk_get(item: Any, *names: str) -> Optional[Any]:
    inner, meta, row = _unwrap(item), _metadata(item), _row(item)
    for name in names:
        for source in (inner, meta, row):
            value = _read(source, name)
            if value is not None and value != "":
                return value
    return None


def chunk_text(item: Any) -> Optional[str]:
    value = chunk_get(item, "content", "text", "quoted_text", "citation_text", "page_content")
    return str(value) if value is not None else None


def chunk_score(item: Any) -> Optional[float]:
    for name in ("rerank_score", "score", "relevance_score", "rrf_score"):
        value = _read(item, name)
        if value is None or value == "":
            value = _metadata(item).get(name)
        if value is not None and value != "":
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def chunk_identity(item: Any) -> Optional[str]:
    meta, row = _metadata(item), _row(item)
    identity = _first(meta.get("chunk_id"), row.get("id"), _read(_unwrap(item), "id"))
    if identity is None:
        document_id = _first(_read(_unwrap(item), "document_id"), meta.get("document_id"))
        index = _read(_unwrap(item), "chunk_index")
        if document_id is not None and index is not None:
            identity = f"{document_id}:{index}"
    return str(identity) if identity is not None else None


def derive_pinpoints(text: str) -> Dict[str, Any]:
    """Read article/law references that literally appear in the retrieved text."""
    articles: List[str] = []
    for match in _ARTICLE_RE.finditer(text or ""):
        number = _digits(match.group(1))
        if not number:
            continue
        label = f"{number} {match.group(2)}".strip() if match.group(2) else number
        if label not in articles:
            articles.append(label)
    law_number = law_year = None
    law_match = _LAW_RE.search(text or "")
    if law_match:
        law_number, law_year = _digits(law_match.group(1)), _digits(law_match.group(2))
    return {"articles": articles, "law_number": law_number, "law_year": law_year}


def _confidence_for(score: Optional[float]) -> CitationConfidence:
    if score is None:
        return CitationConfidence.UNVERIFIED
    if score >= 0.75:
        return CitationConfidence.HIGH
    if score >= 0.5:
        return CitationConfidence.MEDIUM
    return CitationConfidence.LOW


# -----------------------------------------------------------------------------
# Payload-level enrichment (shared by services and API)
# -----------------------------------------------------------------------------
def as_payload(citation: Any) -> Dict[str, Any]:
    if isinstance(citation, dict):
        return dict(citation)
    if hasattr(citation, "model_dump"):
        try:
            return citation.model_dump(mode="json")
        except TypeError:
            return citation.model_dump()
    if hasattr(citation, "dict"):
        return citation.dict()
    return {"raw": str(citation)}


def fill_citation_payload(payload: Dict[str, Any], item: Any) -> Dict[str, Any]:
    """Copy real chunk data into empty citation fields. Never overwrites content."""
    text = chunk_text(item)
    if text and not payload.get("citation_text") and not payload.get("quoted_text"):
        payload["citation_text"] = text
        payload["quoted_text"] = text
    elif payload.get("quoted_text") and not payload.get("citation_text"):
        payload["citation_text"] = payload["quoted_text"]
    elif payload.get("citation_text") and not payload.get("quoted_text"):
        payload["quoted_text"] = payload["citation_text"]

    for field, candidates in _FIELD_SOURCES.items():
        if payload.get(field) in (None, ""):
            value = chunk_get(item, *candidates)
            if field in ("law_name", "section_title"):
                value = _clean_label(value)
            elif field in ("law_number", "law_year", "article_number"):
                value = _digits(value)
            if value not in (None, ""):
                payload[field] = str(value)

    body = payload.get("citation_text") or text or ""
    if body:
        derived = derive_pinpoints(body)
        if payload.get("article_number") in (None, "") and derived["articles"]:
            payload["article_number"] = derived["articles"][0]
        if payload.get("law_number") in (None, "") and derived["law_number"]:
            payload["law_number"] = derived["law_number"]
        if payload.get("law_year") in (None, "") and derived["law_year"]:
            payload["law_year"] = derived["law_year"]

    if payload.get("source_id") in (None, ""):
        identity = chunk_identity(item)
        if identity:
            payload["source_id"] = identity

    if payload.get("relevance_score") in (None, ""):
        score = chunk_score(item)
        if score is not None:
            payload["relevance_score"] = score

    return payload


def enrich_citation_payloads(
    citations: Optional[List[Any]],
    retrieved_items: Optional[List[Any]],
    max_citations: int = MAX_CITATIONS,
) -> List[Dict[str, Any]]:
    """
    Join citations to their retrieved chunk, backfill missing fields, and add
    any retrieved chunk that was dropped by upstream deduplication.
    """
    payloads = [as_payload(c) for c in (citations or [])]
    items = list(retrieved_items or [])
    if not items:
        for payload in payloads:
            if payload.get("quoted_text") and not payload.get("citation_text"):
                payload["citation_text"] = payload["quoted_text"]
        return payloads

    by_id: Dict[str, Any] = {}
    for item in items:
        identity = chunk_identity(item)
        if identity:
            by_id.setdefault(identity, item)

    covered_ids, covered_texts = set(), set()
    for index, payload in enumerate(payloads):
        key = payload.get("source_id") or payload.get("chunk_id") or payload.get("id")
        item = by_id.get(str(key)) if key else None
        if item is None and index < len(items):
            item = items[index]
        if item is not None:
            fill_citation_payload(payload, item)
            identity = chunk_identity(item)
            if identity:
                covered_ids.add(identity)
        body = payload.get("citation_text") or payload.get("quoted_text")
        if body:
            covered_texts.add(_norm_text(body))

    for item in items:
        if len(payloads) >= max_citations:
            break
        identity = chunk_identity(item)
        if identity and identity in covered_ids:
            continue
        text = chunk_text(item)
        if text and _norm_text(text) in covered_texts:
            continue
        payload = fill_citation_payload({"citation_type": "reference"}, item)
        if payload.get("citation_text"):
            payloads.append(payload)
            if identity:
                covered_ids.add(identity)
            covered_texts.add(_norm_text(text))

    return payloads


# -----------------------------------------------------------------------------
# Formatter
# -----------------------------------------------------------------------------
class CitationFormatter:
    """Formats citations from retrieved chunks."""

    def format_citations(self, chunks: List[RetrievedChunk]) -> List[Citation]:
        citations: List[Citation] = []
        seen: set[str] = set()

        for retrieved_chunk in chunks:
            chunk = retrieved_chunk.chunk
            content = (chunk.content or "").strip()
            identity = chunk_identity(retrieved_chunk) or str(chunk.id)
            key = f"{identity}|{hash(content)}"
            if key in seen:
                continue
            seen.add(key)

            payload = fill_citation_payload({"citation_type": "reference"}, retrieved_chunk)
            derived = derive_pinpoints(content)
            score = payload.get("relevance_score")

            page_number = chunk_get(retrieved_chunk, "page_number")
            try:
                page_number = int(page_number) if page_number is not None else None
                if page_number is not None and page_number < 1:
                    page_number = None
            except (TypeError, ValueError):
                page_number = None

            citations.append(
                Citation(
                    chunk_id=chunk.id,
                    source_id=payload.get("source_id") or identity,
                    law_name=payload.get("law_name"),
                    law_number=payload.get("law_number"),
                    law_year=payload.get("law_year"),
                    article_number=payload.get("article_number"),
                    section_title=payload.get("section_title"),
                    page_number=page_number,
                    quoted_text=content or None,
                    citation_type=CitationType.DIRECT_QUOTE if content else CitationType.REFERENCE,
                    confidence=_confidence_for(score),
                    legal_version=payload.get("legal_version"),
                    effective_date=payload.get("effective_date"),
                    amendment_info=payload.get("amendment_info"),
                    source_document=payload.get("source_document"),
                    relevance_score=score,
                    metadata={
                        "citation_text": content or None,
                        "content": content or None,
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index,
                        "chunk_type": str(chunk.chunk_type) if chunk.chunk_type else None,
                        "articles_in_chunk": derived["articles"],
                    },
                )
            )

        if not citations and chunks:
            logger.warning("Citation formatting produced no citations from %d chunks.", len(chunks))
        return citations

    def format_citation_text(self, citation: Citation, language: str = "ar") -> str:
        english = language.lower() == "en"
        parts: List[str] = []
        if citation.law_name:
            parts.append(citation.law_name)
        if citation.law_number and citation.law_year:
            parts.append(
                f"Law No. {citation.law_number} of {citation.law_year}"
                if english else f"القانون رقم {citation.law_number} لسنة {citation.law_year}"
            )
        if citation.article_number:
            parts.append(f"Article {citation.article_number}" if english else f"المادة {citation.article_number}")
        if not parts and citation.section_title:
            parts.append(citation.section_title)
        if not parts and citation.page_number:
            parts.append(f"Page {citation.page_number}" if english else f"صفحة {citation.page_number}")
        return " - ".join(parts) if parts else ("Unidentified source" if english else "مصدر غير محدد")