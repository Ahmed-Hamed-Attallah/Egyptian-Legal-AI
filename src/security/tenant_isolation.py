"""
Tenant isolation enforcement.

Tenant boundaries are enforced at three layers:
1. Repository layer — every query is filtered by tenant_id.
2. Service layer — every operation validates the caller's tenant context.
3. Storage layer — document keys include the tenant path component.

This module provides the primitives for layer 2. Layer 1 is implemented
inside each repository; layer 3 inside LocalDocumentStore.

Important: the tenant_id in AuthContext is derived from a trusted source
(JWT, session, etc.) — never from a request body. That derivation happens
in Phase 13 (FastAPI).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.domain.exceptions import TenantIsolationError
from src.security.authentication import AuthContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TenantContext:
    """
    Tenant context for a request.

    Carries the verified tenant_id plus optional ownership information.
    """
    tenant_id: str
    user_id: Optional[str] = None
    is_admin: bool = False


def tenant_context_from_auth(auth: AuthContext) -> TenantContext:
    """Build a TenantContext from a verified AuthContext."""
    if not auth.tenant_id:
        raise TenantIsolationError("AuthContext has no tenant_id")
    return TenantContext(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        is_admin=auth.is_admin(),
    )


def assert_tenant_access(
    auth: AuthContext,
    resource_tenant_id: str,
    resource_label: str = "resource",
) -> None:
    """
    Verify that the caller's tenant matches the resource's tenant.

    Admins bypass tenant checks (they can access any tenant's resources
    for operational purposes).
    """
    if auth.is_admin():
        return
    if auth.tenant_id != resource_tenant_id:
        logger.warning(
            "Tenant isolation violation: caller_tenant=%s resource_tenant=%s label=%s",
            auth.tenant_id, resource_tenant_id, resource_label,
        )
        raise TenantIsolationError(
            f"Cannot access {resource_label} from another tenant",
            tenant_id=auth.tenant_id,
        )


def assert_authenticated(auth: AuthContext, action: str = "this action") -> None:
    """Require an authenticated caller."""
    if auth.is_anonymous:
        raise TenantIsolationError(f"Authentication required for {action}")


def cache_key_for_tenant(tenant_id: str, key: str) -> str:
    """Build a tenant-scoped cache key."""
    return f"tenant:{tenant_id}:{key}"
