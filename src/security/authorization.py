"""Authorization primitives.

Pure functions over AuthContext — no HTTP, no database.

SECURITY NOTES
--------------
* `AuthContext.is_admin` is a METHOD. `if auth.is_admin:` is always truthy and
  previously granted every permission to every caller. Always call it.
* An unowned private resource is NOT public. `owner_id is None` now means
  "admin only", so legacy rows fail closed instead of leaking.
"""

from __future__ import annotations

import enum
import logging
from typing import Iterable, Optional

from src.domain.exceptions import AuthorizationError
from src.domain.queries import QueryScope
from src.security.authentication import AuthContext

logger = logging.getLogger(__name__)

SCOPE_OFFICIAL = "official"
SCOPE_PRIVATE = "private"


class Permission(str, enum.Enum):
    DOCUMENT_READ = "document:read"
    DOCUMENT_WRITE = "document:write"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_PUBLISH = "document:publish"   # promote private → official
    CONVERSATION_READ = "conversation:read"
    CONVERSATION_WRITE = "conversation:write"
    SEARCH = "search"
    ADMIN = "admin"


_ROLE_PERMISSIONS = {
    "admin": {p for p in Permission},
    "user": {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_WRITE,
        Permission.DOCUMENT_DELETE,
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
    if auth.is_admin():                      # FIX: was `auth.is_admin` (always True)
        return True
    return any(permission in permissions_for_role(role) for role in auth.roles)


def require_permission(auth: AuthContext, permission: Permission) -> None:
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
    if any(has_permission(auth, p) for p in permissions):
        return
    raise AuthorizationError(
        f"Missing required permissions: {[p.value for p in permissions]}"
    )


def is_owner_or_admin(auth: AuthContext, resource_owner_id: Optional[str]) -> bool:
    """Ownership check. Unowned resources are admin-only (fail closed)."""
    if auth.is_admin():
        return True
    if resource_owner_id is None:
        return False
    return bool(auth.user_id) and auth.user_id == resource_owner_id


def assert_owner_or_admin(auth: AuthContext, resource_owner_id: Optional[str]) -> None:
    if not is_owner_or_admin(auth, resource_owner_id):
        raise AuthorizationError("Not authorized to access this resource")


# ---------------------------------------------------------------------------
# Document visibility policy — mirrors MetadataFilter.for_documents exactly.
# ---------------------------------------------------------------------------
def normalise_scope(scope: object) -> str:
    """Accept enum or string; unknown values become 'private' (fail closed)."""
    value = getattr(scope, "value", scope)
    text = str(value or "").strip().lower()
    return SCOPE_OFFICIAL if text == SCOPE_OFFICIAL else SCOPE_PRIVATE


def can_read_document(
    auth: AuthContext,
    *,
    owner_id: Optional[str],
    scope: object,
) -> bool:
    """official → any tenant member; private → owner or admin only."""
    if auth.is_admin():
        return True
    if normalise_scope(scope) == SCOPE_OFFICIAL:
        return True
    return bool(auth.user_id) and owner_id is not None and owner_id == auth.user_id


def assert_can_read_document(
    auth: AuthContext, *, owner_id: Optional[str], scope: object
) -> None:
    if not can_read_document(auth, owner_id=owner_id, scope=scope):
        logger.warning(
            "Document access denied: user=%s owner=%s scope=%s",
            auth.user_id, owner_id, normalise_scope(scope),
        )
        # Message deliberately opaque: do not confirm existence.
        raise AuthorizationError("Document not found")


def visible_scope_filter(
    auth: AuthContext, scope: QueryScope = QueryScope.ALL_SOURCES
):
    """Build the retrieval/SQL visibility filter for this caller."""
    from src.retrieval.metadata_filters import MetadataFilter

    return MetadataFilter.for_documents(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        scope=scope,
        is_admin=auth.is_admin(),
    )