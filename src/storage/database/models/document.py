"""
Document database model.

Stores document metadata and processing state. The actual file content lives
in the document store (filesystem / object storage); the vector embeddings
live in Qdrant. This record is the coordination point between them.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
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
    SoftDeleteMixin,
    TimestampMixin,
    _generate_uuid,
)


class DocumentStatus(str, enum.Enum):
    """Processing status of a document."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentRecord(Base, TimestampMixin, SoftDeleteMixin):
    """
    Persistent record for a legal document.

    Design notes:
    - `storage_key` is the opaque reference used by the document store.
    - `legal_metadata` holds law_number, law_year, effective_date, etc. as JSON
      because these fields are heterogeneous and evolve with the domain.
    - `version` allows tracking multiple versions of the same logical document.
    - `chunk_count` is a denormalized counter to avoid joins for list views.
    - `indexed` indicates whether chunks were successfully inserted into Qdrant.
    """

    __tablename__ = "documents"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "storage_key",
            name="uq_documents_tenant_storage",
        ),
        Index("ix_documents_tenant_status", "tenant_id", "status"),
        Index("ix_documents_tenant_law", "tenant_id", "law_number", "law_year"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_generate_uuid,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        default="default",
    )

    owner_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    # File identity
    original_filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    storage_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    mime_type: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
    )

    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
    )

    # Processing state
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus),
        nullable=False,
        default=DocumentStatus.PENDING,
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    indexed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # Legal metadata — extracted during preprocessing
    law_number: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    law_year: Mapped[Optional[str]] = mapped_column(
        String(16),
        nullable=True,
    )

    title: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
    )

    effective_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    legal_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
    )

    # Versioning
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    source_url: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
    )

    # Relationships
    conversations: Mapped[list["ConversationRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    chunks: Mapped[list["DocumentChunkRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentChunkRecord.chunk_index",
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentRecord id={self.id!r} "
            f"status={self.status.value} "
            f"filename={self.original_filename!r}>"
        )


# Avoid circular import at module load time.
from src.storage.database.models.conversation import ConversationRecord  # noqa: E402,F401
from src.storage.database.models.chunk import DocumentChunkRecord  # noqa: E402,F401