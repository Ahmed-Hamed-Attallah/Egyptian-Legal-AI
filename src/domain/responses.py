"""Domain models for AI-generated responses.

A Response is an auditable artifact linking the generated answer back to the
query, its supporting citations, and generation metadata.

Refusal contract (LLM_STRATEGY §7 / ARCHITECTURE §9)
---------------------------------------------------
When the retrieved context is insufficient, the response is marked
`insufficient_context` with status NO_ANSWER and **citations are cleared**.
A refusal must never be presented as source-grounded.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class ResponseStatus(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    NO_ANSWER = "no_answer"


class ResponseConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------
class Response(BaseModel):
    """An auditable record of one generated answer."""

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    query_id: uuid.UUID = Field(...)
    conversation_id: Optional[uuid.UUID] = Field(default=None)

    answer_text: Optional[str] = Field(default=None)
    status: ResponseStatus = Field(default=ResponseStatus.PENDING)
    confidence: ResponseConfidence = Field(default=ResponseConfidence.UNCERTAIN)

    # Supporting evidence. Empty whenever insufficient_context is True.
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    insufficient_context: bool = Field(default=False)

    model_name: Optional[str] = Field(default=None)
    model_version: Optional[str] = Field(default=None)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    generation_time_ms: Optional[int] = Field(default=None, ge=0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)

    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)

    # -- state ---------------------------------------------------------------
    def is_complete(self) -> bool:
        return self.status == ResponseStatus.COMPLETED

    def has_answer(self) -> bool:
        return bool(self.answer_text) and bool(self.answer_text.strip())

    def is_grounded(self) -> bool:
        """A usable answer must have evidence and not be a refusal."""
        return self.is_complete() and self.has_answer() and bool(self.citations)

    def get_token_cost_estimate(self, cost_per_1k_tokens: float) -> float:
        return (self.total_tokens / 1000.0) * cost_per_1k_tokens

    # -- transitions ---------------------------------------------------------
    def mark_completed(
        self,
        answer: str,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: str,
        generation_time_ms: Optional[int] = None,
        confidence: ResponseConfidence = ResponseConfidence.MEDIUM,
        citations: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.answer_text = answer
        self.status = ResponseStatus.COMPLETED
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.model_name = model_name
        self.generation_time_ms = generation_time_ms
        self.confidence = confidence
        self.citations = list(citations or [])
        self.insufficient_context = False
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error: str) -> None:
        self.status = ResponseStatus.FAILED
        self.citations = []
        self.completed_at = datetime.utcnow()
        errors = self.metadata.setdefault("errors", [])
        errors.append({"error": error, "timestamp": datetime.utcnow().isoformat()})

    def mark_no_answer(self, reason: str = "No relevant information found") -> None:
        self.status = ResponseStatus.NO_ANSWER
        self.citations = []
        self.completed_at = datetime.utcnow()
        self.metadata["no_answer_reason"] = reason

    def mark_insufficient_context(
        self,
        answer: str,
        reason: str = "Retrieved context did not support an answer.",
    ) -> None:
        """Refusal path: keep the explanation, drop every citation.

        Prevents the UI showing "insufficient information" alongside N cited
        sources, which would imply the refusal was source-grounded.
        """
        self.answer_text = answer
        self.status = ResponseStatus.NO_ANSWER
        self.confidence = ResponseConfidence.UNCERTAIN
        self.insufficient_context = True
        self.citations = []
        self.metadata["no_answer_reason"] = reason
        self.completed_at = datetime.utcnow()


# -----------------------------------------------------------------------------
# LLM Generation Contracts
# -----------------------------------------------------------------------------
class GenerationRequest(BaseModel):
    """Request payload for LLM providers.

    Two modes:
      1. Legacy single-turn: system_prompt + user_prompt
      2. Multi-turn with memory: `messages`

    Providers must call `get_messages()` for the canonical format.
    Sampling fields are Optional: None means "use the configured default".
    """

    system_prompt: str
    user_prompt: str
    messages: Optional[List[Dict[str, str]]] = None

    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    do_sample: Optional[bool] = None
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0)

    def get_messages(self) -> List[Dict[str, str]]:
        if self.messages:
            return self.messages
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]

    def has_memory(self) -> bool:
        """True when more than the canonical system+user pair was supplied."""
        return bool(self.messages) and len(self.messages) > 2


class GenerationResponse(BaseModel):
    """Response payload from LLM providers."""

    content: str
    provider: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.content or not self.content.strip()