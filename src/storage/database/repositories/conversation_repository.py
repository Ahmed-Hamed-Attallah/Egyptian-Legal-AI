"""
Conversation repository.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Sequence

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, joinedload

from src.domain.exceptions import DatabaseError
from src.storage.database.models.conversation import ConversationRecord, MessageRecord

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Data-access layer for conversations and messages."""

    def __init__(self, session: Session):
        self._session = session

    # -------------------------------------------------------------------------
    # Conversations
    # -------------------------------------------------------------------------
    def create_conversation(self, conversation: ConversationRecord) -> ConversationRecord:
        try:
            self._session.add(conversation)
            self._session.flush()
            return conversation
        except Exception as exc:
            logger.exception("Failed to create conversation")
            raise DatabaseError(f"Failed to create conversation: {exc}") from exc

    def get_conversation(
        self,
        conversation_id: str,
        tenant_id: str,
        include_deleted: bool = False,
    ) -> Optional[ConversationRecord]:
        stmt = (
            select(ConversationRecord)
            .options(joinedload(ConversationRecord.messages))
            .where(
                ConversationRecord.id == conversation_id,
                ConversationRecord.tenant_id == tenant_id,
            )
        )
        if not include_deleted:
            stmt = stmt.where(ConversationRecord.deleted_at.is_(None))
        return self._session.execute(stmt).unique().scalar_one_or_none()

    def list_by_tenant(
        self,
        tenant_id: str,
        owner_id: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Sequence[ConversationRecord]:
        """List conversations with messages eager-loaded.

        Messages are eager-loaded here so callers (API serialization and the
        conversation-memory digest) can safely access ``conversation.messages``
        after the session closes, without triggering DetachedInstanceError.
        """
        stmt = (
            select(ConversationRecord)
            .options(joinedload(ConversationRecord.messages))
            .where(
                ConversationRecord.tenant_id == tenant_id,
                ConversationRecord.deleted_at.is_(None),
            )
            .order_by(ConversationRecord.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        if owner_id is not None:
            stmt = stmt.where(ConversationRecord.owner_id == owner_id)
        return self._session.execute(stmt).unique().scalars().all()

    def update_title(self, conversation_id: str, tenant_id: str, title: str) -> bool:
        """Update a conversation title (used for auto-titling from the first question)."""
        stmt = (
            update(ConversationRecord)
            .where(
                ConversationRecord.id == conversation_id,
                ConversationRecord.tenant_id == tenant_id,
                ConversationRecord.deleted_at.is_(None),
            )
            .values(title=title, updated_at=datetime.utcnow())
        )
        result = self._session.execute(stmt)
        return result.rowcount > 0

    def soft_delete_conversation(self, conversation_id: str, tenant_id: str) -> bool:
        stmt = (
            update(ConversationRecord)
            .where(
                ConversationRecord.id == conversation_id,
                ConversationRecord.tenant_id == tenant_id,
                ConversationRecord.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.utcnow())
        )
        result = self._session.execute(stmt)
        return result.rowcount > 0

    # -------------------------------------------------------------------------
    # Messages
    # -------------------------------------------------------------------------
    def add_message(self, message: MessageRecord) -> MessageRecord:
        try:
            self._session.add(message)
            self._session.flush()
            return message
        except Exception as exc:
            logger.exception("Failed to add message")
            raise DatabaseError(f"Failed to add message: {exc}") from exc

    def list_messages(
        self,
        conversation_id: str,
        tenant_id: str,
        limit: int = 100,
    ) -> Sequence[MessageRecord]:
        """Return the FIRST `limit` messages in chronological order."""
        # Verify conversation belongs to tenant
        conv = self.get_conversation(conversation_id, tenant_id)
        if conv is None:
            return []
        stmt = (
            select(MessageRecord)
            .where(MessageRecord.conversation_id == conversation_id)
            .order_by(MessageRecord.created_at.asc())
            .limit(limit)
        )
        return self._session.execute(stmt).scalars().all()

    def list_recent_messages(
        self,
        conversation_id: str,
        tenant_id: str,
        limit: int = 20,
    ) -> Sequence[MessageRecord]:
        """Return the LAST `limit` messages in chronological (oldest -> newest) order.

        Used by conversation memory: the model needs the most recent turns,
        not the first ones. ``list_messages`` cannot serve this because
        ``order_by asc + limit`` returns the oldest messages.
        """
        # Verify conversation belongs to tenant
        conv = self.get_conversation(conversation_id, tenant_id)
        if conv is None:
            return []
        stmt = (
            select(MessageRecord)
            .where(MessageRecord.conversation_id == conversation_id)
            .order_by(MessageRecord.created_at.desc())
            .limit(limit)
        )
        records = self._session.execute(stmt).scalars().all()
        return list(reversed(records))

    def count_messages(self, conversation_id: str) -> int:
        stmt = (
            select(func.count())
            .select_from(MessageRecord)
            .where(MessageRecord.conversation_id == conversation_id)
        )
        return self._session.execute(stmt).scalar_one()