# api/schemas/chat.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CitationSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_id: Optional[str] = None
    law_name: Optional[str] = None
    law_number: Optional[str] = None
    law_year: Optional[str] = None
    article_number: Optional[str] = None
    section_title: Optional[str] = None
    legal_version: Optional[str] = None
    effective_date: Optional[Any] = None
    amendment_info: Optional[Any] = None
    source_document: Optional[str] = None
    relevance_score: Optional[float] = None
    citation_text: Optional[str] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    conversation_id: Optional[str] = None
    top_k: Optional[int] = Field(default=None, gt=0, le=50)
    language: str = Field(default="ar", pattern="^(ar|en)$")


class ChatResponse(BaseModel):
    answer: str
    citations: List[CitationSchema] = Field(default_factory=list)
    confidence: Optional[float] = None
    confidence_label: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    validation: Optional[Dict[str, Any]] = None