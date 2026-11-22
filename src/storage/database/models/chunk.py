"""
Document chunk database model.

Stores legal chunks produced during document ingestion.
This allows chunk inspection even if Qdrant indexing fails.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import (
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.storage.database.models.base import (
    Base,
    TimestampMixin,
    _generate_uuid,
)


class DocumentChunkRecord(Base, TimestampMixin):
    """
    A single retrieval-ready chunk belonging to a document.
    """

    __tablename__ = "document_chunks"

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_index",
        ),
        Index(
            "ix_document_chunks_document_index",
            "document_id",
            "chunk_index",
        ),
        Index(
            "ix_document_chunks_tenant_document",
            "tenant_id",
            "document_id",
        ),
        Index(
            "ix_document_chunks_embedding_status",
            "embedding_status",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_generate_uuid,
    )

    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        default="default",
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    chunk_type: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    page_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    section_title: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
    )

    law_name: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
    )

    law_number: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    law_year: Mapped[Optional[str]] = mapped_column(
        String(16),
        nullable=True,
    )

    article_number: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    embedding_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
    )

    vector_id: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    metadata_json: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
    )

    document: Mapped["DocumentRecord"] = relationship(
        "DocumentRecord",
        back_populates="chunks",
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunkRecord id={self.id!r} "
            f"document_id={self.document_id!r} "
            f"chunk_index={self.chunk_index!r}>"
        )