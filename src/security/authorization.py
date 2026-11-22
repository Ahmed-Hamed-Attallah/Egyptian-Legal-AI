"""
Authorization primitives.

Provides role-based permission checks and ownership validation. These are
pure functions that operate on AuthContext — they don't know about HTTP or
databases.
"""
from __future__ import annotations

import enum
import logging
from typing import Iterable

from src.domain.exceptions import AuthorizationError
from src.security.authentication import AuthContext

logger = logging.getLogger(__name__)


class Permission(str, enum.Enum):
    """Application-level permissions."""
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    CONVERSATION_READ = "conversation:read"
    CONVERSATION_WRITE = "conversation:write"
    SEARCH = "search"
    ADMIN = "admin"


# Role -> permissions mapping
_ROLE_PERMISSIONS = {
    "admin": {p for p in Permission},
    "user": {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.CONVERSATION_READ,
        Permission.CONVERSATION_WRITE,
        Permission.SEARCH,
    },
    "viewer": {
        Permission.DOCUMENT_READ,
        Permission.CONVERSATION_READ,
        Permission.SEARCH,
    },
}


def permissions_for_role(role: str) -> set:
    return set(_ROLE_PERMISSIONS.get(role, set()))


def has_permission(auth: AuthContext, permission: Permission) -> bool:
    """Check if the caller has a specific permission."""
    if auth.is_admin:
        return True
    for role in auth.roles:
        if permission in permissions_for_role(role):
            return True
    return False


def require_permission(auth: AuthContext, permission: Permission) -> None:
    """Raise AuthorizationError if the caller lacks the permission."""
    if not has_permission(auth, permission):
        logger.warning(
            "Authorization denied: user=%s permission=%s roles=%s",
            auth.user_id, permission.value, auth.roles,
        )
        raise AuthorizationError(
            f"Missing required permission: {permission.value}",
            required_permission=permission.value,
        )


def require_any_permission(auth: AuthContext, permissions: Iterable[Permission]) -> None:
    """Raise if the caller lacks ALL of the listed permissions."""
    for perm in permissions:
        if has_permission(auth, perm):
            return
    raise AuthorizationError(
        f"Missing required permissions: {[p.value for p in permissions]}"
    )


def is_owner_or_admin(auth: AuthContext, resource_owner_id: str | None) -> bool:
    """Check if the caller owns the resource or is an admin."""
    if auth.is_admin():
        return True
    if resource_owner_id is None:
        # Unowned resources are accessible to authenticated users in the same tenant
        return auth.is_authenticated
    return auth.user_id == resource_owner_id


def assert_owner_or_admin(auth: AuthContext, resource_owner_id: str | None) -> None:
    """Raise AuthorizationError if the caller is not the owner or an admin."""
    if not is_owner_or_admin(auth, resource_owner_id):
        raise AuthorizationError("Not authorized to access this resource")
