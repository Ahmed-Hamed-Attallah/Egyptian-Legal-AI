"""Document repository — SQL access with tenant AND owner/scope enforcement."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Sequence

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import Session

from src.domain.exceptions import DatabaseError
from src.domain.queries import QueryScope
from src.storage.database.models.document import (
    DocumentRecord, DocumentScope, DocumentStatus,
)

logger = logging.getLogger(__name__)


class DocumentRepository:
    def __init__(self, session: Session):
        self._session = session

    # -- create --------------------------------------------------------------
    def create(self, document: DocumentRecord) -> DocumentRecord:
        try:
            self._session.add(document)
            self._session.flush()
            return document
        except Exception as exc:
            logger.exception("Failed to create document")
            raise DatabaseError(f"Failed to create document: {exc}") from exc

    # -- visibility predicate (mirrors MetadataFilter.for_documents) ---------
    @staticmethod
    def _visibility_clause(
        tenant_id: str,
        user_id: Optional[str],
        is_admin: bool,
        scope: QueryScope,
    ):
        if is_admin:
            return DocumentRecord.tenant_id == tenant_id

        official = DocumentRecord.scope == DocumentScope.OFFICIAL
        mine = and_(
            DocumentRecord.scope == DocumentScope.PRIVATE,
            DocumentRecord.owner_id.is_not(None),
            DocumentRecord.owner_id == user_id,
        )
        if isinstance(scope, str):
            scope = QueryScope(scope)

        if scope == QueryScope.OFFICIAL_ONLY:
            visible = official
        elif scope == QueryScope.USER_UPLOADS_ONLY:
            visible = mine
        else:
            visible = or_(official, mine)
        return and_(DocumentRecord.tenant_id == tenant_id, visible)

    # -- read ----------------------------------------------------------------
    def get_by_id(
        self, document_id: str, tenant_id: str, include_deleted: bool = False
    ) -> Optional[DocumentRecord]:
        """Unscoped fetch. INTERNAL ONLY — callers must apply visibility."""
        stmt = select(DocumentRecord).where(
            DocumentRecord.id == document_id,
            DocumentRecord.tenant_id == tenant_id,
        )
        if not include_deleted:
            stmt = stmt.where(DocumentRecord.deleted_at.is_(None))
        return self._session.execute(stmt).scalar_one_or_none()

    def get_visible_by_id(
        self,
        document_id: str,
        tenant_id: str,
        user_id: Optional[str],
        is_admin: bool = False,
    ) -> Optional[DocumentRecord]:
        """Fetch only if the caller may read it. Returns None otherwise."""
        stmt = select(DocumentRecord).where(
            DocumentRecord.id == document_id,
            DocumentRecord.deleted_at.is_(None),
            self._visibility_clause(tenant_id, user_id, is_admin, QueryScope.ALL_SOURCES),
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def list_by_tenant(
        self,
        tenant_id: str,
        status: Optional[DocumentStatus] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Sequence[DocumentRecord]:
        """Unscoped list. INTERNAL / admin only."""
        stmt = (
            select(DocumentRecord)
            .where(
                DocumentRecord.tenant_id == tenant_id,
                DocumentRecord.deleted_at.is_(None),
            )
            .order_by(DocumentRecord.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        if status is not None:
            stmt = stmt.where(DocumentRecord.status == status)
        return self._session.execute(stmt).scalars().all()

    def list_visible(
        self,
        tenant_id: str,
        user_id: Optional[str],
        is_admin: bool = False,
        scope: QueryScope = QueryScope.ALL_SOURCES,
        status: Optional[DocumentStatus] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Sequence[DocumentRecord]:
        stmt = (
            select(DocumentRecord)
            .where(
                DocumentRecord.deleted_at.is_(None),
                self._visibility_clause(tenant_id, user_id, is_admin, scope),
            )
            .order_by(DocumentRecord.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        if status is not None:
            stmt = stmt.where(DocumentRecord.status == status)
        return self._session.execute(stmt).scalars().all()

    def count_visible(
        self,
        tenant_id: str,
        user_id: Optional[str],
        is_admin: bool = False,
        scope: QueryScope = QueryScope.ALL_SOURCES,
        status: Optional[DocumentStatus] = None,
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(DocumentRecord)
            .where(
                DocumentRecord.deleted_at.is_(None),
                self._visibility_clause(tenant_id, user_id, is_admin, scope),
            )
        )
        if status is not None:
            stmt = stmt.where(DocumentRecord.status == status)
        return self._session.execute(stmt).scalar_one()

    def count_by_tenant(
        self, tenant_id: str, status: Optional[DocumentStatus] = None
    ) -> int:
        stmt = select(func.count()).select_from(DocumentRecord).where(
            DocumentRecord.tenant_id == tenant_id,
            DocumentRecord.deleted_at.is_(None),
        )
        if status is not None:
            stmt = stmt.where(DocumentRecord.status == status)
        return self._session.execute(stmt).scalar_one()

    def exists(self, document_id: str, tenant_id: str) -> bool:
        stmt = select(func.count()).select_from(DocumentRecord).where(
            DocumentRecord.id == document_id,
            DocumentRecord.tenant_id == tenant_id,
            DocumentRecord.deleted_at.is_(None),
        )
        return self._session.execute(stmt).scalar_one() > 0

    # -- update --------------------------------------------------------------
    def update_status(
        self,
        document_id: str,
        tenant_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None,
        chunk_count: Optional[int] = None,
        indexed: Optional[bool] = None,
    ) -> Optional[DocumentRecord]:
        values = {"status": status}
        if error_message is not None:
            values["error_message"] = error_message
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        if indexed is not None:
            values["indexed"] = indexed
        result = self._session.execute(
            update(DocumentRecord)
            .where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
            )
            .values(**values)
        )
        if result.rowcount == 0:
            return None
        return self.get_by_id(document_id, tenant_id)

    def update_metadata(
        self,
        document_id: str,
        tenant_id: str,
        law_number: Optional[str] = None,
        law_year: Optional[str] = None,
        title: Optional[str] = None,
        legal_metadata: Optional[dict] = None,
    ) -> Optional[DocumentRecord]:
        values = {}
        if law_number is not None:
            values["law_number"] = law_number
        if law_year is not None:
            values["law_year"] = law_year
        if title is not None:
            values["title"] = title
        if legal_metadata is not None:
            values["legal_metadata"] = legal_metadata
        if not values:
            return self.get_by_id(document_id, tenant_id)
        self._session.execute(
            update(DocumentRecord)
            .where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
            )
            .values(**values)
        )
        return self.get_by_id(document_id, tenant_id)

    def set_scope(
        self, document_id: str, tenant_id: str, scope: DocumentScope
    ) -> Optional[DocumentRecord]:
        """Admin promotion/demotion between official and private."""
        self._session.execute(
            update(DocumentRecord)
            .where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
            )
            .values(scope=scope)
        )
        return self.get_by_id(document_id, tenant_id)

    def set_owner(
        self, document_id: str, tenant_id: str, owner_id: Optional[str]
    ) -> Optional[DocumentRecord]:
        self._session.execute(
            update(DocumentRecord)
            .where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
            )
            .values(owner_id=owner_id)
        )
        return self.get_by_id(document_id, tenant_id)

    # -- delete --------------------------------------------------------------
    def soft_delete(self, document_id: str, tenant_id: str) -> bool:
        result = self._session.execute(
            update(DocumentRecord)
            .where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
                DocumentRecord.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.utcnow())
        )
        return result.rowcount > 0

    def hard_delete(self, document_id: str, tenant_id: str) -> bool:
        doc = self.get_by_id(document_id, tenant_id, include_deleted=True)
        if doc is None:
            return False
        self._session.delete(doc)
        return True