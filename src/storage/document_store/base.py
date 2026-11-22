"""
Abstract document storage interface.

Keeps the storage abstraction independent of filesystem, S3, or any other
backend. The ingestion service depends on this interface, not on concrete
implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Optional


@dataclass(frozen=True)
class StoredDocument:
    """Reference to a stored document."""
    storage_key: str
    original_filename: str
    size_bytes: int
    mime_type: Optional[str] = None


class BaseDocumentStore:
    """Abstract document storage backend."""

    def save(
        self,
        file_content: BinaryIO,
        original_filename: str,
        tenant_id: str,
        mime_type: Optional[str] = None,
    ) -> StoredDocument:
        """Persist a document and return a stable reference."""
        raise NotImplementedError

    def load(self, storage_key: str) -> BinaryIO:
        """Retrieve a document as a binary stream."""
        raise NotImplementedError

    def delete(self, storage_key: str) -> bool:
        """Delete a document. Returns True if it existed."""
        raise NotImplementedError

    def exists(self, storage_key: str) -> bool:
        """Check whether a document exists."""
        raise NotImplementedError

    def size(self, storage_key: str) -> int:
        """Return document size in bytes."""
        raise NotImplementedError
