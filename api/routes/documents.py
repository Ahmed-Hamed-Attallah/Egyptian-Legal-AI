# api/routes/documents.py
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, Query, UploadFile

from api.dependencies import AuthContext, get_auth_context, get_document_service
from api.schemas.documents import (
    DocumentCapabilitiesResponse,
    DocumentChunksResponse,
    DocumentUploadResponse,
)
from src.config.settings import settings
from src.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["documents"])


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
    auth_context: AuthContext = Depends(get_auth_context),
    document_service: DocumentService = Depends(get_document_service),
) -> List[DocumentUploadResponse]:
    records = document_service.list_documents(tenant_id=auth_context.tenant_id)
    return [DocumentUploadResponse.model_validate(record) for record in records]


@router.get("/{document_id}", response_model=DocumentUploadResponse)
def get_document(
    document_id: str,
    auth_context: AuthContext = Depends(get_auth_context),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentUploadResponse:
    record = document_service.get_document(
        document_id=document_id,
        tenant_id=auth_context.tenant_id,
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
        offset=offset,
        limit=limit,
    )

    return DocumentChunksResponse(
        document_id=document_id,
        total=total,
        items=items,
    )