"""Centralised HTTP error mapping.

`error_payload` is the single error envelope, shared with
api/middleware/request_context.py so every failure — handled or not — has the
same shape and carries CORS headers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.domain.exceptions import (
    AuthenticationError,
    DocumentChunkingError,
    EmbeddingError,
    IngestionError,
    LegalAIError,
    ProviderError,
    ProviderUnavailableError,
    QueryError,
    RerankingError,
    RetrievalError,
    UnsupportedFormatError,
    VectorDatabaseError,
)

logger = logging.getLogger(__name__)

ERROR_STATUS_MAP: list[tuple[type[Exception], int]] = [
    (AuthenticationError, status.HTTP_401_UNAUTHORIZED),
    (UnsupportedFormatError, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
    (QueryError, status.HTTP_400_BAD_REQUEST),
    (IngestionError, 422),
    (DocumentChunkingError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    (EmbeddingError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    (VectorDatabaseError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (RetrievalError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (RerankingError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (ProviderUnavailableError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (ProviderError, status.HTTP_502_BAD_GATEWAY),
    (LegalAIError, status.HTTP_400_BAD_REQUEST),
]

# Server-side bugs that must NOT be reported as client errors.
_NOT_CLIENT_ERRORS: tuple[type[Exception], ...] = (KeyError, IndexError, TypeError)


def status_for(exc: Exception) -> int:
    for exc_type, status_code in ERROR_STATUS_MAP:
        if isinstance(exc, exc_type):
            return status_code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def error_payload(
    message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Canonical error envelope used by handlers and middleware."""
    return {
        "error": {
            "message": message,
            "error_code": error_code,
            "details": details or {},
            "request_id": request_id,
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LegalAIError)
    async def legal_ai_error_handler(request: Request, exc: LegalAIError) -> JSONResponse:
        status_code = status_for(exc)
        request_id = getattr(request.state, "request_id", None)
        logger.warning(
            "LegalAIError request_id=%s path=%s status=%s message=%s",
            request_id, request.url.path, status_code, exc,
        )
        return JSONResponse(
            status_code=status_code,
            content=error_payload(
                message=str(exc),
                error_code=getattr(exc, "error_code", None),
                details=getattr(exc, "details", None),
                request_id=request_id,
            ),
        )

    @app.exception_handler(LookupError)
    async def lookup_error_handler(request: Request, exc: LookupError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        # KeyError / IndexError are bugs, not "resource not found".
        if isinstance(exc, _NOT_CLIENT_ERRORS):
            logger.exception(
                "Server-side lookup bug request_id=%s path=%s", request_id, request.url.path
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_payload("Internal server error.", "INTERNAL_ERROR", None, request_id),
            )
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=error_payload(str(exc), "NOT_FOUND", None, request_id),
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_payload(
                "Request validation failed.",
                "REQUEST_VALIDATION_ERROR",
                {"errors": exc.errors()},
                getattr(request.state, "request_id", None),
            ),
        )

    # NOTE: a handler registered for `Exception` is installed on
    # ServerErrorMiddleware, which sits OUTSIDE CORSMiddleware. It is a last
    # resort only. RequestContextMiddleware converts unhandled exceptions to
    # responses *inside* CORS so the browser can actually read them.
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled exception request_id=%s path=%s", request_id, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_payload("Internal server error.", "INTERNAL_ERROR", None, request_id),
        )