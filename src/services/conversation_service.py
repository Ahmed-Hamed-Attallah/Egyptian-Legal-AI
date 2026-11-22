"""
Conversation lifecycle service.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from src.config.settings import settings
from src.domain.exceptions import DatabaseError
from src.security.authentication import AuthContext
from src.security.authorization import Permission, assert_owner_or_admin, require_permission
from src.security.tenant_isolation import assert_tenant_access
from src.storage.database.database import Database
from src.storage.database.models.conversation import ConversationRecord, MessageRecord
from src.storage.database.repositories.conversation_repository import ConversationRepository

logger = logging.getLogger(__name__)

# Max characters kept from the last assistant answer in the cross-chat digest.
_ANSWER_SNIPPET_LENGTH = 200

# Titles treated as "not meaningful" (candidates for auto-titling).
_DEFAULT_TITLES = {"", "محادثة جديدة", "محادثة بدون عنوان"}


class ConversationService:
    """Manages conversations and their messages."""

    def __init__(self, database: Database):
        self._database = database

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def create_conversation(
        self,
        auth: AuthContext,
        title: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ConversationRecord:
        require_permission(auth, Permission.CONVERSATION_WRITE)
        with self._database.session() as session:
            repo = ConversationRepository(session)
            record = ConversationRecord(
                tenant_id=auth.tenant_id,
                owner_id=auth.user_id,
                title=title,
                document_id=document_id,
                metadata_json=metadata,
            )
            created = repo.create_conversation(record)
            session.commit()
            conv_id = created.id
            return repo.get_conversation(conv_id, auth.tenant_id)

    def get_conversation(
        self, conversation_id: str, auth: AuthContext
    ) -> Optional[ConversationRecord]:
        require_permission(auth, Permission.CONVERSATION_READ)
        with self._database.read_only_session() as session:
            repo = ConversationRepository(session)
            conv = repo.get_conversation(conversation_id, auth.tenant_id)
            if conv is None:
                return None
            assert_tenant_access(auth, conv.tenant_id, "conversation")
            assert_owner_or_admin(auth, conv.owner_id)
            return conv

    def list_conversations(
        self,
        auth: AuthContext,
        offset: int = 0,
        limit: int = 50,
    ) -> Sequence[ConversationRecord]:
        require_permission(auth, Permission.CONVERSATION_READ)
        with self._database.read_only_session() as session:
            return ConversationRepository(session).list_by_tenant(
                auth.tenant_id, owner_id=auth.user_id, offset=offset, limit=limit
            )

    def rename_conversation(
        self, conversation_id: str, title: str, auth: AuthContext
    ) -> bool:
        """Rename a conversation (used to auto-title from the first question)."""
        require_permission(auth, Permission.CONVERSATION_WRITE)
        with self._database.session() as session:
            repo = ConversationRepository(session)
            conv = repo.get_conversation(conversation_id, auth.tenant_id)
            if conv is None:
                return False
            assert_tenant_access(auth, conv.tenant_id, "conversation")
            assert_owner_or_admin(auth, conv.owner_id)
            result = repo.update_title(conversation_id, auth.tenant_id, title.strip()[:256])
            session.commit()
            return result

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        auth: AuthContext,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[float] = None,
        citations: Optional[list] = None,
        retrieved_chunk_ids: Optional[list] = None,
    ) -> MessageRecord:
        require_permission(auth, Permission.CONVERSATION_WRITE)
        with self._database.session() as session:
            repo = ConversationRepository(session)
            conv = repo.get_conversation(conversation_id, auth.tenant_id)
            if conv is None:
                raise DatabaseError(f"Conversation not found: {conversation_id}")
            assert_tenant_access(auth, conv.tenant_id, "conversation")
            assert_owner_or_admin(auth, conv.owner_id)

            message = MessageRecord(
                conversation_id=conversation_id,
                role=role,
                content=content,
                provider=provider,
                model=model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                citations_json=citations,
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
            created = repo.add_message(message)
            session.commit()
            return created

    def delete_conversation(self, conversation_id: str, auth: AuthContext) -> bool:
        require_permission(auth, Permission.CONVERSATION_WRITE)
        with self._database.session() as session:
            repo = ConversationRepository(session)
            conv = repo.get_conversation(conversation_id, auth.tenant_id)
            if conv is None:
                return False
            assert_tenant_access(auth, conv.tenant_id, "conversation")
            assert_owner_or_admin(auth, conv.owner_id)
            result = repo.soft_delete_conversation(conversation_id, auth.tenant_id)
            session.commit()
            return result

    # -------------------------------------------------------------------------
    # Conversation memory
    # -------------------------------------------------------------------------
    def has_meaningful_title(self, conversation: ConversationRecord) -> bool:
        """Whether the conversation title is useful for cross-chat reference."""
        return (conversation.title or "").strip() not in _DEFAULT_TITLES

    def get_history_messages(
        self,
        conversation_id: str,
        auth: AuthContext,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Short-term memory: the last turns of the CURRENT conversation.

        Returns messages shaped for the LLM provider interface
        (LLM_STRATEGY.md §4): [{"role": "user"|"assistant", "content": "..."}].
        Plain dicts are materialized inside the session so callers never touch
        detached ORM instances.
        """
        require_permission(auth, Permission.CONVERSATION_READ)
        window = limit or settings.CHAT_MEMORY_MAX_MESSAGES
        with self._database.read_only_session() as session:
            repo = ConversationRepository(session)
            conv = repo.get_conversation(conversation_id, auth.tenant_id)
            if conv is None:
                return []
            assert_tenant_access(auth, conv.tenant_id, "conversation")
            assert_owner_or_admin(auth, conv.owner_id)

            records = repo.list_recent_messages(
                conversation_id, auth.tenant_id, limit=window
            )
            # Materialize while the session is still open.
            return [
                {"role": m.role, "content": m.content}
                for m in records
                if m.role in ("user", "assistant")
            ]

    def build_memory_digest(
        self,
        auth: AuthContext,
        current_conversation_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Long-term memory: a compact digest of the user's OTHER conversations.

        For each recent conversation we keep: title, creation date, the last
        few user questions, and a short snippet of the last assistant answer.
        This lets the model answer things like:
        "في المحادثة اللي اسمها X سألتك عن Y" by matching against the digest.
        """
        require_permission(auth, Permission.CONVERSATION_READ)
        max_convs = settings.CHAT_MEMORY_MAX_CONVERSATIONS
        questions_per_conv = settings.CHAT_MEMORY_QUESTIONS_PER_CONVERSATION

        with self._database.read_only_session() as session:
            repo = ConversationRepository(session)
            # list_by_tenant eager-loads messages, so access below is safe.
            conversations = repo.list_by_tenant(
                auth.tenant_id, owner_id=auth.user_id, offset=0, limit=max_convs
            )

            digest: List[Dict[str, object]] = []
            for conv in conversations:
                if current_conversation_id and conv.id == current_conversation_id:
                    continue  # skip the chat we are currently in

                questions = [m.content for m in conv.messages if m.role == "user"]
                questions = questions[:questions_per_conv]

                last_answer = next(
                    (m.content for m in reversed(conv.messages) if m.role == "assistant"),
                    None,
                )
                snippet: Optional[str] = None
                if last_answer:
                    snippet = (
                        last_answer
                        if len(last_answer) <= _ANSWER_SNIPPET_LENGTH
                        else last_answer[:_ANSWER_SNIPPET_LENGTH] + "..."
                    )

                digest.append(
                    {
                        "id": conv.id,
                        "title": (conv.title or "").strip() or "محادثة بدون عنوان",
                        "created_at": conv.created_at.isoformat() if conv.created_at else None,
                        "questions": questions,
                        "last_answer_snippet": snippet,
                    }
                )
            return digest