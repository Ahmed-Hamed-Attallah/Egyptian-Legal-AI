# api/schemas/search.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, gt=0, le=50)
    rerank: bool = True
    filters: Optional[Dict[str, Any]] = None


class SearchResultSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    content: str = ""
    score: Optional[float] = None
    article_number: Optional[str] = None
    section_title: Optional[str] = None
    law_name: Optional[str] = None
    law_number: Optional[str] = None
    law_year: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultSchema] = Field(default_factory=list)
    total: int