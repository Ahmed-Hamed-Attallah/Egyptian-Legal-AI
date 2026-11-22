# api/schemas/documents.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_id: str
    status: str
    filename: Optional[str] = None
    chunk_count: int = 0
    message: Optional[str] = None
    indexed: bool = False
    created_at: Optional[str] = None


class DocumentCapabilitiesResponse(BaseModel):
    allowed_file_types: List[str] = Field(default_factory=list)
    max_upload_size_bytes: int


class DocumentChunkSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    document_id: str
    chunk_index: int
    content: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    law_name: Optional[str] = None
    law_number: Optional[str] = None
    law_year: Optional[str] = None
    article_number: Optional[str] = None
    chunk_type: Optional[str] = None
    embedding_status: Optional[str] = None
    vector_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class DocumentChunksResponse(BaseModel):
    document_id: str
    total: int
    items: List[DocumentChunkSchema] = Field(default_factory=list)