"""Authentication primitives and JWT token lifecycle.

Owns password hashing, the verified-identity contract (AuthContext), and both
issuance and verification of access tokens. Keeping both sides here guarantees
the API, services and workers agree on one signing scheme (ARCHITECTURE §8).
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import jwt

from src.config.settings import settings
from src.domain.exceptions import AuthenticationError

logger = logging.getLogger(__name__)

# bcrypt only considers the first 72 bytes of a password. Reject longer inputs
# explicitly instead of letting them be silently truncated.
MAX_PASSWORD_BYTES: int = 72


@dataclass(frozen=True)
class AuthContext:
    """Verified identity of the caller.

    - `user_id` is None for anonymous/unauthenticated calls.
    - `tenant_id` is always present (defaults to DEFAULT_TENANT_ID).
    - `roles` drives authorization decisions.
    """

    user_id: Optional[str] = None
    tenant_id: str = field(default_factory=lambda: settings.DEFAULT_TENANT_ID)
    email: Optional[str] = None
    roles: tuple = field(default_factory=lambda: ("user",))
    is_authenticated: bool = False

    @property
    def is_anonymous(self) -> bool:
        return not self.is_authenticated

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def is_admin(self) -> bool:
        return "admin" in self.roles


def create_anonymous_context(tenant_id: Optional[str] = None) -> AuthContext:
    """Create an anonymous AuthContext for unauthenticated requests."""
    return AuthContext(
        tenant_id=tenant_id or settings.DEFAULT_TENANT_ID,
        is_authenticated=False,
    )


def validate_password_length(password: str) -> None:
    """Enforce bcrypt's 72-byte ceiling, counting UTF-8 bytes (Arabic-safe)."""
    if not password:
        raise AuthenticationError("Password cannot be empty.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise AuthenticationError(
            f"Password must not exceed {MAX_PASSWORD_BYTES} bytes "
            "(roughly 36 Arabic characters)."
        )


class PasswordHasher:
    """Password hashing using bcrypt via passlib."""

    def __init__(self, rounds: int = 12):
        self._rounds = rounds
        self._crypt = self._build_crypt()

    def _build_crypt(self):
        try:
            from passlib.context import CryptContext

            return CryptContext(
                schemes=["bcrypt"],
                deprecated="auto",
                bcrypt__rounds=self._rounds,
                bcrypt__truncate_error=True,
            )
        except ImportError as exc:
            raise AuthenticationError(
                "passlib is required for password hashing. "
                "Install with: pip install 'passlib[bcrypt]' 'bcrypt<5'"
            ) from exc

    def hash(self, password: str) -> str:
        validate_password_length(password)
        return self._crypt.hash(password)

    def verify(self, password: str, password_hash: str) -> bool:
        if not password or not password_hash:
            return False
        try:
            return bool(self._crypt.verify(password, password_hash))
        except Exception as exc:
            logger.warning("Password verification error: %s", exc)
            return False

    def needs_rehash(self, password_hash: str) -> bool:
        try:
            return bool(self._crypt.needs_update(password_hash))
        except Exception:
            return False


def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token.

    NOTE: this is a *generator* (API keys, reset nonces). It does NOT verify
    JWTs — use `decode_access_token` for that.
    """
    return secrets.token_urlsafe(length)


def _require_signing_key() -> str:
    if not settings.JWT_SECRET_KEY:
        raise AuthenticationError(
            "JWT_SECRET_KEY is not configured. Generate one with: "
            "python -c \"import secrets; print(secrets.token_urlsafe(64))\""
        )
    return settings.JWT_SECRET_KEY


def create_access_token(
    *,
    user_id: str | int,
    tenant_id: str,
    roles: Sequence[str] = ("user",),
    email: Optional[str] = None,
    expires_minutes: Optional[int] = None,
) -> str:
    """Issue a signed access token.

    `sub` is always a string: PyJWT >= 2.10 rejects non-string subjects.
    """
    key = _require_signing_key()
    minutes = expires_minutes or settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)

    payload: dict[str, Any] = {
        "sub": str(user_id),
        "user_id": str(user_id),
        "tenant_id": tenant_id,
        "roles": [str(r) for r in roles],
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    if email:
        payload["email"] = email

    return jwt.encode(payload, key, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify signature, expiry and required claims; return the claim set.

    Raises AuthenticationError on any failure so callers can map it to 401
    without knowing JWT internals.
    """
    if not token:
        raise AuthenticationError("Missing access token.")
    key = _require_signing_key()

    try:
        payload = jwt.decode(
            token,
            key,
            algorithms=[settings.JWT_ALGORITHM],  # pin alg: blocks alg=none
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Access token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        logger.warning("Token verification failed: %s", exc)
        raise AuthenticationError("Invalid access token.") from exc

    if not isinstance(payload, dict):
        raise AuthenticationError("Invalid token payload.")
    return payload