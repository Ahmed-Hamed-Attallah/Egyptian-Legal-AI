"""
Domain models for conversations.

This module defines the data structures for conversation threads.
A Conversation groups multiple Queries and Responses, enabling 
multi-turn legal research and context-aware retrieval.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class ConversationStatus(str, Enum):
    """
    Lifecycle status of a conversation thread.
    """
    ACTIVE = "active"       # Currently in use
    ARCHIVED = "archived"   # Read-only, no new queries
    DELETED = "deleted"     # Soft-deleted


# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------

class Conversation(BaseModel):
    """
    Represents a multi-turn conversation thread between a user and the AI.
    
    Groups related Queries and Responses, maintaining context for 
    coherent legal research sessions.
    """

    # Pydantic V2 configuration
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    # Core Identifiers
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for this conversation."
    )
    user_id: uuid.UUID = Field(
        ..., 
        description="ID of the user who owns this conversation."
    )

    # Content & Context
    title: str = Field(
        default="New Conversation",
        max_length=200,
        description="Short summary or title of the conversation."
    )
    context_summary: Optional[str] = Field(
        default=None,
        description=(
            "Compressed summary of conversation history for LLM context. "
            "Used to maintain coherence without exceeding token limits."
        )
    )

    # Status & Metrics
    status: ConversationStatus = Field(
        default=ConversationStatus.ACTIVE,
        description="Current lifecycle status."
    )
    turn_count: int = Field(
        default=0,
        ge=0,
        description="Number of query-response pairs in this conversation."
    )

    # Legal Context (Optional but useful)
    legal_domains: List[str] = Field(
        default_factory=list,
        description="List of legal domains detected (e.g., 'Civil', 'Penal', 'Commercial')."
    )

    # Extensibility & Timestamps
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra data (e.g., tags, source preferences)."
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of creation."
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of last update."
    )
    last_activity_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the most recent query/response."
    )

    # -------------------------------------------------------------------------
    # Business Logic Methods
    # -------------------------------------------------------------------------

    def is_active(self) -> bool:
        """
        Checks if the conversation is open for new queries.
        """
        return self.status == ConversationStatus.ACTIVE

    def increment_turn(self) -> None:
        """
        Increments the turn count and updates activity timestamp.
        Called after a successful query-response cycle.
        """
        self.turn_count += 1
        self.last_activity_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def archive(self) -> None:
        """
        Archives the conversation, making it read-only.
        """
        self.status = ConversationStatus.ARCHIVED
        self.updated_at = datetime.utcnow()

    def update_summary(self, summary: str) -> None:
        """
        Updates the context summary used for LLM prompting.
        """
        self.context_summary = summary
        self.updated_at = datetime.utcnow()

    def add_legal_domain(self, domain: str) -> None:
        """
        Adds a legal domain tag if not already present.
        """
        if domain not in self.legal_domains:
            self.legal_domains.append(domain)
            self.updated_at = datetime.utcnow()