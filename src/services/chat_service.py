"""
Chat service.

Orchestrates RAGOrchestrator and records messages via ConversationService.

Memory:
- Short-term: last turns of the CURRENT conversation → `history`.
- Long-term: digest of the user's OTHER conversations → `memory_digest`.

Citations are enriched from the retrieved chunks BEFORE persistence, so both
the live response and the stored conversation history carry the quoted legal
text and pinpoint references.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional

from src.config.settings import settings
from src.generation.citation_formatter import chunk_identity, enrich_citation_payloads
from src.rag.orchestrator import RAGOrchestrator
from src.security.authentication import AuthContext
from src.services.conversation_service import ConversationService

logger = logging.getLogger(__name__)

_RETRIEVED_KEYS = (
    "retrieved_chunks", "reranked_chunks", "context_chunks", "chunks", "sources",
)


def _extract_retrieved(result: Dict[str, Any]) -> List[Any]:
    for key in _RETRIEVED_KEYS:
        value = result.get(key)
        if isinstance(value, (list, tuple)) and value:
            return list(value)
    return []


def _collect_retrieved_chunk_ids(result: Dict[str, Any]) -> Optional[List[str]]:
    explicit = result.get("retrieved_chunk_ids")
    if explicit:
        return [str(cid) for cid in explicit if cid is not None] or None
    ids = [chunk_identity(item) for item in _extract_retrieved(result)]
    return [i for i in ids if i] or None


class ChatService:
    """Orchestrates RAG question answering and conversation persistence."""

    def __init__(
        self,
        orchestrator: RAGOrchestrator,
        conversation_service: Optional[ConversationService] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.conversation_service = conversation_service

    def _orchestrator_supports_memory(self) -> bool:
        try:
            parameters = inspect.signature(self.orchestrator.ask).parameters
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                return True
            return "history" in parameters and "memory_digest" in parameters
        except (TypeError, ValueError):
            return False

    def _load_history(self, conversation_id: Optional[str], auth: AuthContext) -> List[Dict[str, str]]:
        if not conversation_id or self.conversation_service is None:
            return []
        try:
            return self.conversation_service.get_history_messages(
                conversation_id=conversation_id, auth=auth,
            )
        except Exception:
            logger.exception("Failed to load conversation history; continuing without it.")
            return []

    def _load_memory_digest(
        self, auth: AuthContext, current_conversation_id: Optional[str],
    ) -> List[Dict[str, object]]:
        if self.conversation_service is None:
            return []
        try:
            return self.conversation_service.build_memory_digest(
                auth=auth, current_conversation_id=current_conversation_id,
            )
        except Exception:
            logger.exception("Failed to build memory digest; continuing without it.")
            return []

    def ask(
        self,
        question: str,
        auth: AuthContext,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = 3,
    ) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        history: List[Dict[str, str]] = []
        memory_digest: List[Dict[str, object]] = []
        if getattr(settings, "CHAT_MEMORY_ENABLED", True) and self.conversation_service is not None:
            history = self._load_history(conversation_id, auth)
            memory_digest = self._load_memory_digest(auth, conversation_id)

        if self._orchestrator_supports_memory():
            result = self.orchestrator.ask(
                question, top_k=top_k, history=history, memory_digest=memory_digest,
            )
        else:
            logger.warning("RAGOrchestrator.ask has no memory support; running stateless.")
            result = self.orchestrator.ask(question, top_k=top_k)

        retrieved = _extract_retrieved(result)
        citations = enrich_citation_payloads(result.get("citations"), retrieved)
        result["citations"] = citations

        with_text = sum(1 for c in citations if c.get("citation_text"))
        logger.info(
            "Citations: %d total, %d with quoted text, %d retrieved chunks.",
            len(citations), with_text, len(retrieved),
        )
        if retrieved and with_text == 0:
            logger.error(
                "Retrieved chunks carry no content. Chunk payload keys: %s",
                sorted(getattr(retrieved[0], "chunk", retrieved[0]).__dict__.keys())
                if hasattr(getattr(retrieved[0], "chunk", retrieved[0]), "__dict__") else "unknown",
            )

        if self.conversation_service is not None:
            answer = str(result.get("answer", ""))
            if conversation_id:
                self.conversation_service.get_conversation(conversation_id=conversation_id, auth=auth)
            else:
                conversation = self.conversation_service.create_conversation(
                    auth=auth, title=question[:120],
                )
                conversation_id = conversation.id

            self.conversation_service.add_message(
                conversation_id=conversation_id, role="user", content=question, auth=auth,
            )
            assistant_message = self.conversation_service.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
                auth=auth,
                provider=result.get("provider"),
                model=result.get("model"),
                tokens_used=result.get("tokens_used"),
                latency_ms=result.get("latency_ms"),
                citations=citations or None,
                retrieved_chunk_ids=_collect_retrieved_chunk_ids(result),
            )
            result["conversation_id"] = conversation_id
            result["message_id"] = assistant_message.id

        return result