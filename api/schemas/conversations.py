# api/schemas/conversations.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)


class MessageSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    conversation_id: str
    role: str
    content: str
    created_at: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    tenant_id: str
    title: str
    status: str
    created_at: Any
    updated_at: Any
    message_count: int = 0


class ConversationDetailSchema(ConversationSchema):
    messages: List[MessageSchema] = Field(default_factory=list)