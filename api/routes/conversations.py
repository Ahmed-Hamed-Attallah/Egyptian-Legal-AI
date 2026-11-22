# api/routes/conversations.py
from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import AuthContext, get_auth_context, get_conversation_service
from api.schemas.conversations import (
    ConversationCreateRequest,
    ConversationDetailSchema,
    ConversationSchema,
)
from src.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])

def _serialize_conversation(record, message_count: int = 0) -> dict:
    """Helper to map SQLAlchemy ConversationRecord to Pydantic Schema dict."""
    return {
        "id": record.id,
        "tenant_id": record.tenant_id,
        "title": record.title or "محادثة جديدة",
        "status": "active" if getattr(record, "deleted_at", None) is None else "deleted",
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "message_count": message_count,
    }

@router.post("", response_model=ConversationSchema, status_code=201)
def create_conversation(
    payload: ConversationCreateRequest,
    auth_context: AuthContext = Depends(get_auth_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationSchema:
    result = conversation_service.create_conversation(
        auth=auth_context,
        title=payload.title or "محادثة جديدة",
    )
    return ConversationSchema.model_validate(_serialize_conversation(result, 0))

@router.get("", response_model=List[ConversationSchema])
def list_conversations(
    auth_context: AuthContext = Depends(get_auth_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> List[ConversationSchema]:
    results = conversation_service.list_conversations(auth=auth_context)
    
    serialized = []
    for item in results:
        # Safely count messages if the relationship is loaded
        msg_count = len(item.messages) if hasattr(item, "messages") and item.messages else 0
        serialized.append(_serialize_conversation(item, msg_count))
        
    return [ConversationSchema.model_validate(item) for item in serialized]

@router.get("/{conversation_id}", response_model=ConversationDetailSchema)
def get_conversation(
    conversation_id: str,
    auth_context: AuthContext = Depends(get_auth_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetailSchema:
    result = conversation_service.get_conversation(
        conversation_id=conversation_id,
        auth=auth_context,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
        
    msg_count = len(result.messages) if hasattr(result, "messages") and result.messages else 0
    base_data = _serialize_conversation(result, msg_count)
    
    messages_data = []
    if hasattr(result, "messages") and result.messages:
        for msg in result.messages:
            messages_data.append({
                "id": msg.id,
                "conversation_id": msg.conversation_id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at,
                "metadata": {
                    "provider": msg.provider,
                    "model": msg.model,
                    "tokens_used": msg.tokens_used,
                    "latency_ms": msg.latency_ms,
                    "citations": msg.citations_json,
                    "retrieved_chunk_ids": msg.retrieved_chunk_ids,
                }
            })
            
    base_data["messages"] = messages_data
    return ConversationDetailSchema.model_validate(base_data)

@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    auth_context: AuthContext = Depends(get_auth_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> dict:
    success = conversation_service.delete_conversation(
        conversation_id=conversation_id,
        auth=auth_context,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"status": "deleted", "id": conversation_id}