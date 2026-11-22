"""Domain models for legal documents in the Egyptian Legal AI system.

This module defines the core document entities, their lifecycle statuses,
document types, and source authorities as used across the application.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class DocumentStatus(str, Enum):
    """Lifecycle status of a legal document."""

    PENDING = "pending"
    LOADED = "loaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Classification of legal document types."""

    LEGISLATION = "legislation"
    REGULATION = "regulation"
    DECREE = "decree"
    COURT_RULING = "court_ruling"
    CONSTITUTION = "constitution"
    INTERNATIONAL_TREATY = "international_treaty"
    MINISTERIAL_DECISION = "ministerial_decision"
    OTHER = "other"


class SourceAuthority(str, Enum):
    """Authority that issued the legal document."""

    PARLIAMENT = "parliament"
    PRESIDENT = "president"
    PRIME_MINISTER = "prime_minister"
    MINISTRY = "ministry"
    COURT = "court"
    CONSTITUTIONAL_ASSEMBLY = "constitutional_assembly"
    INTERNATIONAL_BODY = "international_body"
    OTHER = "other"


class Document(BaseModel):
    """Core domain model representing a legal document.

    Fields like file_hash, file_size_bytes, and mime_type are populated
    during the ingestion pipeline and may be None at initial construction.
    """

    model_config = {"populate_by_name": True}

    id: uuid.UUID = Field(default_factory=uuid.uuid4, validation_alias="document_id")
    title: str = Field(..., min_length=1, max_length=500)
    document_type: DocumentType = Field(default=DocumentType.OTHER)
    source_authority: SourceAuthority = Field(default=SourceAuthority.OTHER)
    file_path: Optional[str] = Field(default=None, validation_alias="source_path")
    file_hash: Optional[str] = Field(default=None)
    file_size_bytes: Optional[int] = Field(default=None, ge=0)
    mime_type: Optional[str] = Field(default=None)
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    owner_id: Optional[str] = Field(default=None)
    issued_date: Optional[date] = Field(default=None)
    published_date: Optional[date] = Field(default=None)
    effective_date: Optional[date] = Field(default=None)
    language: str = Field(default="ar")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # ── Convenience aliases for backward compatibility ──────────────

    @property
    def document_id(self) -> uuid.UUID:
        """Alias for id, used in tests and some service layers."""
        return self.id

    @property
    def source_path(self) -> Optional[str]:
        """Alias for file_path, used in tests and registry."""
        return self.file_path

    @field_validator("file_size_bytes")
    @classmethod
    def validate_file_size(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("file_size_bytes cannot be negative")
        return v

    def mark_status(self, new_status: DocumentStatus) -> None:
        """Transition the document to a new status."""
        self.status = new_status
        self.updated_at = datetime.utcnow()

    def mark_completed(self) -> None:
        """Mark the document as successfully processed."""
        self.mark_status(DocumentStatus.COMPLETED)

    def mark_failed(self) -> None:
        """Mark the document as failed."""
        self.mark_status(DocumentStatus.FAILED)

    def mark_loaded(self) -> None:
        """Mark the document as loaded from storage."""
        self.mark_status(DocumentStatus.LOADED)

    def mark_processing(self) -> None:
        """Mark the document as being processed."""
        self.mark_status(DocumentStatus.PROCESSING)

        # ── Document-level API (used by services and tests) ─────────────

    def register(self, document: "Document") -> None:
        """Register a Document domain object in the registry.

        Args:
            document: The Document to register.
        """
        doc_id = str(document.id)
        self._documents[doc_id] = document

    def get(self, document_id: str | uuid.UUID) -> "Document | None":
        """Retrieve a document by its ID.

        Args:
            document_id: The document's unique identifier.

        Returns:
            The Document if found, None otherwise.
        """
        return self._documents.get(str(document_id))

    def list_all(self) -> list["Document"]:
        """List all registered documents.

        Returns:
            List of all Document objects in the registry.
        """
        return list(self._documents.values())

    def remove(self, document_id: str | uuid.UUID) -> bool:
        """Remove a document from the registry.

        Args:
            document_id: The document's unique identifier.

        Returns:
            True if the document was found and removed, False otherwise.
        """
        key = str(document_id)
        if key in self._documents:
            del self._documents[key]
            return True
        return False