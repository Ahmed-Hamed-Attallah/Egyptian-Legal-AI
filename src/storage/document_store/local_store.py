"""
Local filesystem document store.

Documents are stored under `<storage_dir>/<tenant_id>/<storage_key>`. The
storage_key is a UUID-based name, not the original filename, which prevents
collisions and path-traversal attacks.
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Optional

from src.config.settings import settings
from src.domain.exceptions import DocumentStorageError
from src.storage.document_store.base import BaseDocumentStore, StoredDocument

logger = logging.getLogger(__name__)

# Safe filename characters — everything else is stripped
_SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _sanitize_filename(name: str) -> str:
    """Strip path components and unsafe characters."""
    # Take only the basename to prevent path traversal
    name = Path(name).name
    # Replace unsafe characters
    sanitized = "".join(c if c in _SAFE_CHARS else "_" for c in name)
    # Collapse multiple underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("._") or "document"


class LocalDocumentStore(BaseDocumentStore):
    """Store documents on the local filesystem."""

    def __init__(self, base_dir: Optional[Path] = None):
        self._base_dir = Path(base_dir or settings.DOCUMENT_STORAGE_DIR)
        try:
            self._base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DocumentStorageError(
                f"Cannot create storage directory {self._base_dir}: {exc}"
            ) from exc

    def _tenant_dir(self, tenant_id: str) -> Path:
        # Tenant ID is validated upstream; sanitize defensively
        safe_tenant = "".join(c if c in _SAFE_CHARS else "_" for c in tenant_id)
        path = self._base_dir / safe_tenant
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_path(self, storage_key: str) -> Path:
        """Resolve a storage_key to an absolute path, validating it stays within base_dir."""
        # storage_key format: "<tenant>/<uuid>"
        candidate = (self._base_dir / storage_key).resolve()
        base_resolved = self._base_dir.resolve()
        try:
            candidate.relative_to(base_resolved)
        except ValueError as exc:
            raise DocumentStorageError(
                f"Path traversal detected for storage_key: {storage_key}"
            ) from exc
        return candidate

    def save(
        self,
        file_content: BinaryIO,
        original_filename: str,
        tenant_id: str,
        mime_type: Optional[str] = None,
    ) -> StoredDocument:
        if not original_filename:
            raise DocumentStorageError("original_filename is required")

        sanitized = _sanitize_filename(original_filename)
        extension = Path(sanitized).suffix.lower()
        unique_name = f"{uuid.uuid4().hex}{extension}"

        tenant_dir = self._tenant_dir(tenant_id)
        target = tenant_dir / unique_name
        storage_key = f"{Path(tenant_dir).name}/{unique_name}"

        try:
            with target.open("wb") as out:
                shutil.copyfileobj(file_content, out)
            size = target.stat().st_size
            logger.info(
                "Saved document: tenant=%s key=%s size=%d",
                tenant_id, storage_key, size,
            )
            return StoredDocument(
                storage_key=storage_key,
                original_filename=sanitized,
                size_bytes=size,
                mime_type=mime_type,
            )
        except OSError as exc:
            raise DocumentStorageError(
                f"Failed to save document: {exc}", storage_key=storage_key
            ) from exc

    def load(self, storage_key: str) -> BinaryIO:
        path = self._resolve_path(storage_key)
        if not path.is_file():
            raise DocumentStorageError(
                f"Document not found: {storage_key}", storage_key=storage_key
            )
        try:
            return path.open("rb")
        except OSError as exc:
            raise DocumentStorageError(
                f"Failed to load document: {exc}", storage_key=storage_key
            ) from exc

    def delete(self, storage_key: str) -> bool:
        path = self._resolve_path(storage_key)
        if not path.exists():
            return False
        try:
            path.unlink()
            logger.info("Deleted document: %s", storage_key)
            return True
        except OSError as exc:
            raise DocumentStorageError(
                f"Failed to delete document: {exc}", storage_key=storage_key
            ) from exc

    def exists(self, storage_key: str) -> bool:
        try:
            return self._resolve_path(storage_key).is_file()
        except DocumentStorageError:
            return False

    def size(self, storage_key: str) -> int:
        path = self._resolve_path(storage_key)
        if not path.is_file():
            raise DocumentStorageError(
                f"Document not found: {storage_key}", storage_key=storage_key
            )
        return path.stat().st_size
