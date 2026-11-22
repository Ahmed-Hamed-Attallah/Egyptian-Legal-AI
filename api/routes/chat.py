"""Chat endpoint (Phase 13).

Mounted under `settings.API_PREFIX` by api/main.py, giving:
    POST {prefix}/chat
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import AuthContext, get_auth_context, get_chat_service
from api.schemas.chat import ChatRequest, ChatResponse
from api.serializers import build_citation_schemas
from src.services.chat_service import ChatService
from src.utils.pipeline_logger import trace_step

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, summary="Ask a legal question")
def chat(
    payload: ChatRequest,
    auth_context: AuthContext = Depends(get_auth_context),
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    with trace_step("CHAT API - SERVICE CALL"):
        result = chat_service.ask(
            question=payload.question,
            auth=auth_context,
            conversation_id=payload.conversation_id,
            top_k=payload.top_k,
        )

        # Join citations against the chunks the pipeline actually used, so the
        # response always carries the quoted legal text and pinpoint references.
        citations = build_citation_schemas(
            result.get("citations"),
            result.get("retrieved_chunks"),
        )

        return ChatResponse(
            answer=str(result.get("answer", "")),
            citations=citations,
            provider=result.get("provider"),
            model=result.get("model"),
            tokens_used=result.get("tokens_used"),
            latency_ms=result.get("latency_ms"),
            conversation_id=result.get("conversation_id"),
            message_id=result.get("message_id"),
            validation=result.get("validation"),
        )