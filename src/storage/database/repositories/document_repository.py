"""
Document repository.

Encapsulates SQL operations for documents. The service layer uses this
instead of touching the ORM directly.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Sequence

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from src.domain.exceptions import DatabaseError
from src.storage.database.models.document import DocumentRecord, DocumentStatus

logger = logging.getLogger(__name__)


class DocumentRepository:
    """Data-access layer for DocumentRecord."""

    def __init__(self, session: Session):
        self._session = session

    # -------------------------------------------------------------------------
    # Create
    # -------------------------------------------------------------------------
    def create(self, document: DocumentRecord) -> DocumentRecord:
        try:
            self._session.add(document)
            self._session.flush()
            return document
        except Exception as exc:
            logger.exception("Failed to create document")
            raise DatabaseError(f"Failed to create document: {exc}") from exc

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------
    def get_by_id(
        self,
        document_id: str,
        tenant_id: str,
        include_deleted: bool = False,
    ) -> Optional[DocumentRecord]:
        stmt = select(DocumentRecord).where(
            DocumentRecord.id == document_id,
            DocumentRecord.tenant_id == tenant_id,
        )
        if not include_deleted:
            stmt = stmt.where(DocumentRecord.deleted_at.is_(None))
        return self._session.execute(stmt).scalar_one_or_none()

    def list_by_tenant(
        self,
        tenant_id: str,
        status: Optional[DocumentStatus] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Sequence[DocumentRecord]:
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

    def count_by_tenant(
        self,
        tenant_id: str,
        status: Optional[DocumentStatus] = None,
    ) -> int:
        stmt = select(func.count()).select_from(DocumentRecord).where(
            DocumentRecord.tenant_id == tenant_id,
            DocumentRecord.deleted_at.is_(None),
        )
        if status is not None:
            stmt = stmt.where(DocumentRecord.status == status)
        return self._session.execute(stmt).scalar_one()

    def exists(self, document_id: str, tenant_id: str) -> bool:
        stmt = (
            select(func.count())
            .select_from(DocumentRecord)
            .where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
                DocumentRecord.deleted_at.is_(None),
            )
        )
        return self._session.execute(stmt).scalar_one() > 0

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------
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

        stmt = (
            update(DocumentRecord)
            .where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
            )
            .values(**values)
        )

        result = self._session.execute(stmt)

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

        stmt = (
            update(DocumentRecord)
            .where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
            )
            .values(**values)
        )
        self._session.execute(stmt)
        return self.get_by_id(document_id, tenant_id)

    # -------------------------------------------------------------------------
    # Delete
    # -------------------------------------------------------------------------
    def soft_delete(self, document_id: str, tenant_id: str) -> bool:
        stmt = (
            update(DocumentRecord)
            .where(
                DocumentRecord.id == document_id,
                DocumentRecord.tenant_id == tenant_id,
                DocumentRecord.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.utcnow())
        )
        result = self._session.execute(stmt)
        return result.rowcount > 0

    def hard_delete(self, document_id: str, tenant_id: str) -> bool:
        doc = self.get_by_id(document_id, tenant_id, include_deleted=True)
        if doc is None:
            return False
        self._session.delete(doc)
        return True
