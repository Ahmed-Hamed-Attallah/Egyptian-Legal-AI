"""FastAPI dependency providers.

Services are provisioned in api/lifespan.py and read from app.state. Token
verification is delegated to src.security.authentication — no crypto here.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from fastapi import Depends, Header, HTTPException, Request, status

from src.config.settings import settings
from src.domain.exceptions import AuthenticationError
from src.security.authentication import AuthContext, decode_access_token
from src.services.auth_service import AuthService

logger = logging.getLogger(__name__)

__all__ = ["AuthContext", "get_auth_context", "get_auth_service", "require_admin"]


def get_settings_dependency():
    """Expose centralized settings to routes that need configuration."""
    return settings


def _normalise_roles(roles: object) -> tuple:
    """Accept a list/tuple, or a comma/whitespace-separated string."""
    if isinstance(roles, str):
        return tuple(r for r in re.split(r"[,\s]+", roles.strip()) if r)
    if isinstance(roles, (list, tuple, set)):
        return tuple(str(r).strip() for r in roles if str(r).strip())
    return ()


def get_auth_context(
    authorization: Optional[str] = Header(default=None),
) -> AuthContext:
    """Resolve the caller's tenant, identity and roles from the bearer token."""
    if not settings.AUTH_ENABLED:
        # Development bypass: full access, explicitly NOT authenticated.
        return AuthContext(
            user_id="anonymous",
            tenant_id=settings.DEFAULT_TENANT_ID,
            roles=("admin",),
            is_authenticated=False,
        )

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(token.strip())
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = payload.get("sub") or payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is missing a subject claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return AuthContext(
        user_id=str(user_id),
        tenant_id=payload.get("tenant_id") or settings.DEFAULT_TENANT_ID,
        email=payload.get("email"),
        roles=_normalise_roles(payload.get("roles")) or ("user",),
        is_authenticated=True,
    )


def require_admin(
    auth_context: AuthContext = Depends(get_auth_context),
) -> AuthContext:
    """Guard admin-only endpoints."""
    if not settings.AUTH_ENABLED:
        return auth_context
    if "admin" not in (getattr(auth_context, "roles", ()) or ()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required.",
        )
    return auth_context


def _get_app_service(request: Request, attribute: str, unavailable_message: str):
    service = getattr(request.app.state, attribute, None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=unavailable_message,
        )
    return service


def get_auth_service(request: Request) -> AuthService:
    return _get_app_service(
        request, "auth_service", "Authentication service is unavailable."
    )


def get_chat_service(request: Request):
    return _get_app_service(request, "chat_service", "Chat service is unavailable.")


def get_search_service(request: Request):
    return _get_app_service(request, "search_service", "Search service is unavailable.")


def get_document_service(request: Request):
    return _get_app_service(
        request, "document_service", "Document service is unavailable."
    )


def get_conversation_service(request: Request):
    return _get_app_service(
        request, "conversation_service", "Conversation service is unavailable."
    )


def get_admin_service(request: Request):
    return _get_app_service(request, "admin_service", "Admin service is unavailable.")


def get_job_manager(request: Request):
    return _get_app_service(
        request, "job_manager", "Background job manager is unavailable."
    )