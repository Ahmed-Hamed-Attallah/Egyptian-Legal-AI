"""
Conversation and message database models.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.storage.database.models.base import (
    Base,
    SoftDeleteMixin,
    TimestampMixin,
    _generate_uuid,
)


class ConversationRecord(Base, TimestampMixin, SoftDeleteMixin):
    """
    A conversation between a user (or tenant) and the system.

    Conversations are scoped to a tenant and optionally linked to a document.
    """

    __tablename__ = "conversations"
    __table_args__ = (
        Index("ix_conversations_tenant_owner", "tenant_id", "owner_id"),
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

    document_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    title: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
    )

    # Use standard SQLAlchemy JSON instead of SQLite-specific JSON.
    # This is safer for PostgreSQL and remains portable for tests/local dev.
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
    )

    # Relationships
    document: Mapped[Optional["DocumentRecord"]] = relationship(
        back_populates="conversations",
    )

    messages: Mapped[list["MessageRecord"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="MessageRecord.created_at",
    )

    def __repr__(self) -> str:
        return f"<ConversationRecord id={self.id!r} title={self.title!r}>"


class MessageRecord(Base, TimestampMixin):
    """
    A single message within a conversation.

    Stores both user and assistant messages, plus optional citations and
    retrieval metadata for traceability.
    """

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_generate_uuid,
    )

    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # "user" | "assistant" | "system"
    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Retrieval / generation traceability
    provider: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    model: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
    )

    tokens_used: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # Use standard SQLAlchemy JSON instead of SQLite-specific JSON.
    citations_json: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
    )

    retrieved_chunk_ids: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
    )

    conversation: Mapped["ConversationRecord"] = relationship(
        back_populates="messages",
    )

    def __repr__(self) -> str:
        return f"<MessageRecord id={self.id!r} role={self.role!r}>"


# Import at the bottom to avoid circular import issues while still making
# DocumentRecord available for SQLAlchemy relationship configuration.
from src.storage.database.models.document import DocumentRecord  # noqa: E402,F401