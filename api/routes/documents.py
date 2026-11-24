# api/routes/documents.py
"""Document endpoints.

Every read is scoped by the caller's identity: official corpus + own uploads.
`capabilities` stays public so the upload widget renders before sign-in.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, status

from api.dependencies import (
    AuthContext, get_auth_context, get_document_service, require_admin,
)
from api.schemas.documents import (
    DocumentCapabilitiesResponse, DocumentChunksResponse,
    DocumentScopeUpdateRequest, DocumentUploadResponse,
)
from src.config.settings import settings
from src.domain.queries import QueryScope
from src.services.document_service import DocumentService
from src.storage.database.models.document import DocumentScope

router = APIRouter(prefix="/documents", tags=["documents"])


def _resolve_scope(value: str) -> QueryScope:
    try:
        return QueryScope(value)
    except ValueError:
        return QueryScope.ALL_SOURCES


@router.get("/capabilities", response_model=DocumentCapabilitiesResponse)
def capabilities() -> DocumentCapabilitiesResponse:
    return DocumentCapabilitiesResponse(
        allowed_file_types=settings.ALLOWED_FILE_TYPES,
        max_upload_size_bytes=settings.MAX_UPLOAD_SIZE_BYTES,
    )


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile,
    auth_context: AuthContext = Depends(get_auth_context),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    contents = await file.read()
    result = document_service.ingest_upload(
        file_bytes=contents,
        filename=file.filename or "upload",
        auth=auth_context,
    )
    return DocumentUploadResponse.model_validate(result)


@router.get("", response_model=List[DocumentUploadResponse])
def list_documents(
    scope: str = Query(default=QueryScope.ALL_SOURCES.value),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    auth_context: AuthContext = Depends(get_auth_context),
    document_service: DocumentService = Depends(get_document_service),
) -> List[DocumentUploadResponse]:
    records = document_service.list_documents(
        tenant_id=auth_context.tenant_id,
        user_id=auth_context.user_id,
        is_admin=auth_context.is_admin(),
        scope=_resolve_scope(scope),
        offset=offset,
        limit=limit,
    )
    return [DocumentUploadResponse.model_validate(r) for r in records]


@router.get("/{document_id}", response_model=DocumentUploadResponse)
def get_document(
    document_id: str,
    auth_context: AuthContext = Depends(get_auth_context),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    record = document_service.get_document(
        document_id=document_id,
        tenant_id=auth_context.tenant_id,
        user_id=auth_context.user_id,
        is_admin=auth_context.is_admin(),
    )
    return DocumentUploadResponse.model_validate(record)


@router.get("/{document_id}/chunks", response_model=DocumentChunksResponse)
def get_document_chunks(
    document_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    auth_context: AuthContext = Depends(get_auth_context),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentChunksResponse:
    items, total = document_service.list_document_chunks(
        document_id=document_id,
        tenant_id=auth_context.tenant_id,
        user_id=auth_context.user_id,
        is_admin=auth_context.is_admin(),
        offset=offset,
        limit=limit,
    )
    return DocumentChunksResponse(document_id=document_id, total=total, items=items)


@router.patch("/{document_id}/scope", response_model=DocumentUploadResponse)
def update_document_scope(
    document_id: str,
    payload: DocumentScopeUpdateRequest,
    auth_context: AuthContext = Depends(require_admin),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    """Promote a document into the official corpus, or return it to private."""
    target = (
        DocumentScope.OFFICIAL if payload.scope == "official" else DocumentScope.PRIVATE
    )
    record = document_service.set_document_scope(
        document_id=document_id, tenant_id=auth_context.tenant_id, scope=target
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentUploadResponse.model_validate(record)