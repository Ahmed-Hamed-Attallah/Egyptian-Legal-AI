# src/domain/queries.py
"""
Domain models for user queries and retrieval results.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field
from src.domain.chunks import Chunk

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class QueryIntent(str, Enum):
    LEGAL_RESEARCH = "legal_research"
    CASE_ANALYSIS = "case_analysis"
    DRAFTING = "drafting"
    GENERAL_QA = "general_qa"

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
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def get_effective_text(self) -> str:
        return self.processed_text if self.processed_text else self.raw_text

    def is_restricted_scope(self) -> bool:
        return self.scope != QueryScope.ALL_SOURCES

    def is_official_search(self) -> bool:
        return self.scope == QueryScope.OFFICIAL_ONLY

# -----------------------------------------------------------------------------
# Retrieval Models
# -----------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    """
    Represents a chunk retrieved from the vector database with relevance scores.
    Wraps the core Chunk domain model to preserve legal boundaries and metadata.
    """
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    # Core Chunk Reference
    chunk: Chunk = Field(..., description="The underlying legal chunk.")
    
    # Retrieval Scores
    score: float = Field(..., ge=0.0, description="Primary relevance score.")
    dense_score: Optional[float] = Field(default=None, ge=0.0)
    sparse_score: Optional[float] = Field(default=None, ge=0.0)
    rerank_score: Optional[float] = None
    
    # Ranking Information
    rank: int = Field(default=0, ge=0)
    
    # Retrieval Metadata
    retriever_type: str = Field(default="unknown", description="Which retriever found this chunk.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Business Logic Methods
    # -------------------------------------------------------------------------
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        return self.score >= threshold

    def has_legal_reference(self) -> bool:
        return self.chunk.has_legal_reference()

    def get_citation_context(self) -> str:
        return self.chunk.get_citation_label()

    def to_dict_for_llm(self) -> Dict[str, Any]:
        return {
            "chunk_id": str(self.chunk.id),
            "content": self.chunk.content,
            "citation": self.get_citation_context(),
            "score": self.score,
            "rank": self.rank,
        }