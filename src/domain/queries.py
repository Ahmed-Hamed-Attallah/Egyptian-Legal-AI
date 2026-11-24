# src/domain/queries.py
"""Domain models for user queries and retrieval results.

`Query` carries the verified caller context so a single visibility policy can be
derived from it and shared by dense retrieval, sparse retrieval and SQL.

`RetrievedChunk` exposes source authority (`scope`, `is_official`) because
ARCHITECTURE §7 requires user uploads to be distinguishable from curated
official sources in both the LLM context and the UI.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.domain.chunks import Chunk

if TYPE_CHECKING:  # pragma: no cover
    from src.retrieval.metadata_filters import MetadataFilter

SCOPE_OFFICIAL = "official"
SCOPE_PRIVATE = "private"


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class QueryIntent(str, Enum):
    LEGAL_RESEARCH = "legal_research"
    CASE_ANALYSIS = "case_analysis"
    DRAFTING = "drafting"
    GENERAL_QA = "general_qa"
    CONVERSATIONAL = "conversational"   # memory-only; must skip retrieval


class QueryScope(str, Enum):
    OFFICIAL_ONLY = "official_only"
    USER_UPLOADS_ONLY = "user_uploads_only"
    ALL_SOURCES = "all_sources"


# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------
class Query(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    conversation_id: Optional[uuid.UUID] = Field(default=None)
    raw_text: str = Field(..., min_length=1)
    processed_text: Optional[str] = Field(default=None)
    intent: QueryIntent = Field(default=QueryIntent.LEGAL_RESEARCH)
    scope: QueryScope = Field(default=QueryScope.ALL_SOURCES)
    language: str = Field(default="ar")

    # Verified caller context. Never populated from a request body — only from
    # AuthContext in the API layer (tenant isolation, ARCHITECTURE §7).
    tenant_id: Optional[str] = Field(default=None)
    user_id: Optional[str] = Field(default=None)
    is_admin: bool = Field(default=False)

    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def get_effective_text(self) -> str:
        return self.processed_text if self.processed_text else self.raw_text

    def is_restricted_scope(self) -> bool:
        return self.scope != QueryScope.ALL_SOURCES

    def is_official_search(self) -> bool:
        return self.scope == QueryScope.OFFICIAL_ONLY

    def requires_retrieval(self) -> bool:
        """Conversational turns are answered from history, not from sources."""
        return self.intent != QueryIntent.CONVERSATIONAL

    def visibility_filter(self) -> "MetadataFilter":
        """Build the access filter for this query.

        Imported locally: `metadata_filters` imports `QueryScope` from this
        module, so a top-level import would be circular.
        """
        from src.retrieval.metadata_filters import MetadataFilter

        if not self.tenant_id:
            raise ValueError(
                "Query.tenant_id is required to build a visibility filter. "
                "Populate it from AuthContext before retrieval."
            )
        scope = self.scope
        if isinstance(scope, str):
            scope = QueryScope(scope)
        return MetadataFilter.for_documents(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            scope=scope,
            is_admin=self.is_admin,
        )


# -----------------------------------------------------------------------------
# Retrieval Models
# -----------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store with relevance scores.

    Wraps the core Chunk domain model to preserve legal boundaries and metadata.
    """

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    chunk: Chunk = Field(..., description="The underlying legal chunk.")

    score: float = Field(..., ge=0.0, description="Primary relevance score.")
    dense_score: Optional[float] = Field(default=None, ge=0.0)
    sparse_score: Optional[float] = Field(default=None, ge=0.0)
    rerank_score: Optional[float] = None

    rank: int = Field(default=0, ge=0)

    retriever_type: str = Field(default="unknown")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Source authority (ARCHITECTURE §7)
    # -------------------------------------------------------------------------
    @property
    def chunk_metadata(self) -> Dict[str, Any]:
        return getattr(self.chunk, "metadata", None) or {}

    @property
    def scope(self) -> str:
        """'official' or 'private'. Unknown payloads fail closed to private."""
        raw = self.chunk_metadata.get("scope")
        value = str(getattr(raw, "value", raw) or "").strip().lower()
        return SCOPE_OFFICIAL if value == SCOPE_OFFICIAL else SCOPE_PRIVATE

    @property
    def is_official(self) -> bool:
        return self.scope == SCOPE_OFFICIAL

    @property
    def owner_id(self) -> Optional[str]:
        raw = self.chunk_metadata.get("owner_id")
        return str(raw) if raw is not None else None

    @property
    def document_id(self) -> str:
        return str(getattr(self.chunk, "document_id", ""))

    @property
    def source_label(self) -> str:
        """Human-readable authority marker for prompts and UI."""
        return "مصدر رسمي" if self.is_official else "مستند用户上传".replace("用户上传", "مرفوع من المستخدم")

    # -------------------------------------------------------------------------
    # Business logic
    # -------------------------------------------------------------------------
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        return self.score >= threshold

    def has_legal_reference(self) -> bool:
        return self.chunk.has_legal_reference()

    def get_citation_context(self) -> str:
        return self.chunk.get_citation_label()

    def to_dict_for_llm(self) -> Dict[str, Any]:
        """Context payload for the generator.

        `scope` is included so the model can distinguish curated law from a
        user-supplied document instead of presenting both as authoritative.
        """
        return {
            "chunk_id": str(self.chunk.id),
            "content": self.chunk.content,
            "citation": self.get_citation_context(),
            "score": self.score,
            "rank": self.rank,
            "scope": self.scope,
            "is_official": self.is_official,
        }