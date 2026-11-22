# api/serializers.py
"""
Citation → API schema normalization.

Bridges the domain model (`quoted_text`) to the schema (`citation_text`) and,
when the retrieved chunks are supplied, backfills any field the upstream
formatter left empty using the shared enrichment helper in
src/generation/citation_formatter.py. Nothing is invented.
"""
from __future__ import annotations

from typing import Any, List, Optional

from api.schemas.chat import CitationSchema
from src.generation.citation_formatter import (
    as_payload,
    enrich_citation_payloads,
    fill_citation_payload,
)

_TEXT_KEYS = ("citation_text", "quoted_text", "content", "text", "page_content")

_BACKFILL = {
    "source_id": ("source_id", "chunk_id", "id"),
    "law_name": ("law_name", "title", "law_title"),
    "law_number": ("law_number",),
    "law_year": ("law_year", "year"),
    "article_number": ("article_number",),
    "section_title": ("section_title", "category"),
    "legal_version": ("legal_version", "version_label"),
    "effective_date": ("effective_date",),
    "amendment_info": ("amendment_info", "amendments"),
    "source_document": ("source_document", "source_path", "file_path", "storage_key", "file_name"),
    "relevance_score": ("relevance_score", "score", "rerank_score"),
}

_STRING_FIELDS = {
    "source_id", "law_name", "law_number", "law_year", "article_number",
    "section_title", "legal_version", "source_document", "citation_text",
}


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def normalize_citation(citation: Any, retrieved_chunk: Any = None) -> CitationSchema:
    if isinstance(citation, CitationSchema) and retrieved_chunk is None:
        return citation

    data = as_payload(citation) if not isinstance(citation, CitationSchema) else citation.model_dump()
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    row = metadata.get("raw_row_data") if isinstance(metadata.get("raw_row_data"), dict) else {}

    if retrieved_chunk is not None:
        fill_citation_payload(data, retrieved_chunk)

    if not data.get("citation_text"):
        data["citation_text"] = _first(
            *(data.get(key) for key in _TEXT_KEYS),
            *(metadata.get(key) for key in _TEXT_KEYS),
            *(row.get(key) for key in _TEXT_KEYS),
        )

    for field, candidates in _BACKFILL.items():
        if data.get(field) in (None, ""):
            fallback = _first(
                *(data.get(key) for key in candidates),
                *(metadata.get(key) for key in candidates),
                *(row.get(key) for key in candidates),
            )
            if fallback is not None:
                data[field] = fallback

    for field in _STRING_FIELDS:
        value = data.get(field)
        if value is not None and not isinstance(value, str):
            data[field] = str(value)

    return CitationSchema.model_validate(data)


def build_citation_schemas(
    citations: Optional[List[Any]],
    retrieved_chunks: Optional[List[Any]] = None,
) -> List[CitationSchema]:
    """Enrich against ground-truth chunks, then validate into response schemas."""
    payloads = enrich_citation_payloads(citations, retrieved_chunks)
    return [normalize_citation(payload) for payload in payloads]