from __future__ import annotations
from typing import Dict, List
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, str] = Field(default_factory=dict)

class LLMStatusSchema(BaseModel):
    primary_provider: str
    fallback_order: str
    available: List[str] = Field(default_factory=list)
    unavailable: List[str] = Field(default_factory=list)
    missing_credentials: List[str] = Field(default_factory=list)

class RAGStatusSchema(BaseModel):
    initialized: bool
    chat_service: str
    search_service: str
    document_service: str

class SettingsSchema(BaseModel):
    embedding_model: str
    embedding_device: str
    qdrant_collection: str
    vector_dimension: int
    reranking_model: str
    auth_enabled: bool
    cache_enabled: bool
    document_storage_dir: str
    max_upload_size_bytes: int
    allowed_file_types: List[str] = Field(default_factory=list)
    worker_enabled: bool
    worker_embedded: bool
    metrics_enabled: bool

class AdminStatusResponse(BaseModel):
    llm: LLMStatusSchema
    rag: RAGStatusSchema
    qdrant: str
    postgres: str
    settings: SettingsSchema