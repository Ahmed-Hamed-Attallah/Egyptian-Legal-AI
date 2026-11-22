"""Authentication service (Phase 12).

Business logic only: no HTTP concerns. Token issuance is delegated to
src.security.authentication so signing and verification stay in one place.
"""

from __future__ import annotations

import logging

from src.config.settings import settings
from src.domain.exceptions import AuthenticationError
from src.security.authentication import (
    PasswordHasher,
    create_access_token,
    validate_password_length,
)
from src.storage.database.database import Database
from src.storage.database.models.user import UserRecord
from src.storage.database.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)


def _normalise_email(email: str) -> str:
    return (email or "").strip().lower()


class AuthService:
    def __init__(self, database: Database):
        self._database = database
        self._hasher = PasswordHasher()

    def register(
        self,
        email: str,
        password: str,
        tenant_id: str | None = None,
    ) -> dict:
        """Create a user account.

        `tenant_id` is server-assigned; public registration must never let the
        caller pick a tenant (tenant isolation, ARCHITECTURE §7 / Phase 12).
        """
        email = _normalise_email(email)
        if not email:
            raise AuthenticationError("Email is required.")
        validate_password_length(password)

        tenant_id = tenant_id or settings.DEFAULT_TENANT_ID

        with self._database.session() as session:
            repo = UserRepository(session)
            if repo.get_by_email(email):
                raise AuthenticationError("Email already registered")

            user = UserRecord(
                email=email,
                password_hash=self._hasher.hash(password),
                tenant_id=tenant_id,
                role="user",
            )
            repo.create(user)
            session.commit()
            logger.info("Registered user %s in tenant %s", user.id, tenant_id)
            return {"message": "User created successfully", "user_id": user.id}

    def login(self, email: str, password: str) -> str:
        """Verify credentials and return a signed access token."""
        email = _normalise_email(email)

        with self._database.read_only_session() as session:
            repo = UserRepository(session)
            user = repo.get_by_email(email)
            if not user or not self._hasher.verify(password, user.password_hash):
                raise AuthenticationError("Invalid credentials")

            return create_access_token(
                user_id=user.id,
                tenant_id=user.tenant_id,
                roles=(user.role or "user",),
                email=user.email,
            )