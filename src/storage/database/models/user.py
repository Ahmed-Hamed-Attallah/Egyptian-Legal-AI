"""
User database model.

Stores authentication credentials and tenant association. Passwords are stored
as bcrypt hashes — never in plain text.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from src.storage.database.models.base import Base, TimestampMixin, _generate_uuid


class UserRecord(Base, TimestampMixin):
    """
    Application user.

    Notes:
    - `password_hash` is a bcrypt hash. Never store plain-text passwords.
    - `tenant_id` scopes the user to a tenant for isolation.
    - `is_active` supports soft-disable without deletion.
    - `role` is a simple string for now; a full RBAC system can be layered on.
    """
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True, default="default"
    )
    email: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    username: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, unique=True
    )
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<UserRecord id={self.id!r} email={self.email!r} role={self.role!r}>"
