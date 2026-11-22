"""Authentication endpoints (Phase 12/13).

Mounted under settings.API_PREFIX by api/main.py:
    POST {prefix}/auth/register
    POST {prefix}/auth/login
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from api.dependencies import get_auth_service
from src.config.settings import settings
from src.domain.exceptions import AuthenticationError
from src.security.authentication import MAX_PASSWORD_BYTES
from src.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    """Public self-registration payload.

    `tenant_id` is intentionally not accepted: letting an unauthenticated
    caller choose its tenant would break tenant isolation. Extra keys sent by
    older clients are ignored rather than rejected.
    """

    model_config = ConfigDict(extra="ignore")

    email: EmailStr
    password: str = Field(min_length=8, max_length=MAX_PASSWORD_BYTES)

    @field_validator("password")
    @classmethod
    def _check_password_bytes(cls, value: str) -> str:
        # bcrypt counts bytes, not characters — critical for Arabic passwords.
        if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError(
                f"Password must not exceed {MAX_PASSWORD_BYTES} bytes."
            )
        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    email: EmailStr
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_BYTES)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
def register(
    payload: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    try:
        return auth_service.register(
            payload.email,
            payload.password,
            settings.DEFAULT_TENANT_ID,
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Exchange credentials for an access token",
)
def login(
    payload: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    try:
        token = auth_service.login(payload.email, payload.password)
        return TokenResponse(access_token=token)
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)
        ) from exc