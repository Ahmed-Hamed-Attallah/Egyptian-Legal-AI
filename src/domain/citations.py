# src/domain/citations.py
"""
Domain models for legal citations.
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field

class CitationType(str, Enum):
    DIRECT_QUOTE = "direct_quote"
    PARAPHRASE = "paraphrase"
    REFERENCE = "reference"

class CitationConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"

class Citation(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    # Core Identifiers & Linkage (DB Persistence)
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    response_id: Optional[uuid.UUID] = Field(default=None)
    chunk_id: Optional[uuid.UUID] = Field(default=None)
    legal_source_id: Optional[uuid.UUID] = Field(default=None)
    legal_version_id: Optional[uuid.UUID] = Field(default=None)
    article_number: Optional[str] = Field(default=None)
    page_number: Optional[int] = Field(default=None, ge=1)
    
    # Content & Classification
    quoted_text: Optional[str] = Field(default=None)
    citation_type: CitationType = Field(default=CitationType.PARAPHRASE)
    confidence: CitationConfidence = Field(default=CitationConfidence.MEDIUM)
    
    # RAG DTO Fields (Used by Citation Formatter)
    source_id: Optional[str] = Field(default=None)
    law_name: Optional[str] = Field(default=None)
    law_number: Optional[str] = Field(default=None)
    law_year: Optional[str] = Field(default=None)
    section_title: Optional[str] = Field(default=None)
    legal_version: Optional[str] = Field(default=None)
    effective_date: Optional[str] = Field(default=None)
    amendment_info: Optional[Any] = Field(default=None)
    source_document: Optional[str] = Field(default=None)
    relevance_score: Optional[float] = Field(default=None)

    # Extensibility & Timestamps
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def is_direct_quote(self) -> bool:
        return self.citation_type == CitationType.DIRECT_QUOTE

    def has_pinpoint(self) -> bool:
        return bool(self.article_number) or bool(self.page_number)

    def get_display_text(self, source_name: Optional[str] = None, year: Optional[int] = None, law_number: Optional[str] = None) -> str:
        parts = []
        if self.article_number: parts.append(f"المادة {self.article_number}")
        if source_name: parts.append(f"من {source_name}")
        if law_number and year: parts.append(f"رقم {law_number} لسنة {year}")
        elif year: parts.append(f"لسنة {year}")
        if self.page_number and not self.article_number: parts.append(f"(صفحة {self.page_number})")
        return " ".join(parts) if parts else "مرجع قانوني"