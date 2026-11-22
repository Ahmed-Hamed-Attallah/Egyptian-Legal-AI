"""
Document chunk repository.

Encapsulates SQL operations for document chunks.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from src.domain.exceptions import DatabaseError
from src.storage.database.models.chunk import DocumentChunkRecord

logger = logging.getLogger(__name__)


class ChunkRepository:
    """Data-access layer for DocumentChunkRecord."""

    def __init__(self, session: Session):
        self._session = session

    # -------------------------------------------------------------------------
    # Create
    # -------------------------------------------------------------------------
    def create_many(
        self,
        records: Sequence[DocumentChunkRecord],
    ) -> Sequence[DocumentChunkRecord]:
        try:
            self._session.add_all(list(records))
            self._session.flush()
            return records
        except Exception as exc:
            logger.exception("Failed to create document chunks")
            raise DatabaseError(f"Failed to create document chunks: {exc}") from exc

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------
    def list_by_document(
        self,
        document_id: str,
        tenant_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> Sequence[DocumentChunkRecord]:
        stmt = (
            select(DocumentChunkRecord)
            .where(
                DocumentChunkRecord.document_id == document_id,
                DocumentChunkRecord.tenant_id == tenant_id,
            )
            .order_by(DocumentChunkRecord.chunk_index.asc())
            .offset(offset)
            .limit(limit)
        )
        return self._session.execute(stmt).scalars().all()

    def count_by_document(
        self,
        document_id: str,
        tenant_id: str,
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(DocumentChunkRecord)
            .where(
                DocumentChunkRecord.document_id == document_id,
                DocumentChunkRecord.tenant_id == tenant_id,
            )
        )
        return self._session.execute(stmt).scalar_one()

    def get_by_id(
        self,
        chunk_id: str,
        tenant_id: str,
    ) -> Optional[DocumentChunkRecord]:
        stmt = select(DocumentChunkRecord).where(
            DocumentChunkRecord.id == chunk_id,
            DocumentChunkRecord.tenant_id == tenant_id,
        )
        return self._session.execute(stmt).scalar_one_or_none()

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------
    def update_embedding_result(
        self,
        chunk_id: str,
        tenant_id: str,
        embedding_status: str,
        vector_id: Optional[str] = None,
    ) -> bool:
        values = {"embedding_status": embedding_status}

        if vector_id is not None:
            values["vector_id"] = vector_id

        stmt = (
            update(DocumentChunkRecord)
            .where(
                DocumentChunkRecord.id == chunk_id,
                DocumentChunkRecord.tenant_id == tenant_id,
            )
            .values(**values)
        )

        result = self._session.execute(stmt)
        return result.rowcount > 0

    # -------------------------------------------------------------------------
    # Delete
    # -------------------------------------------------------------------------
    def delete_by_document(
        self,
        document_id: str,
        tenant_id: str,
    ) -> int:
        stmt = delete(DocumentChunkRecord).where(
            DocumentChunkRecord.document_id == document_id,
            DocumentChunkRecord.tenant_id == tenant_id,
        )
        result = self._session.execute(stmt)
        return result.rowcount