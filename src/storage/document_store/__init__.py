"""Document storage package."""
from src.storage.document_store.base import BaseDocumentStore, StoredDocument
from src.storage.document_store.local_store import LocalDocumentStore

__all__ = ["BaseDocumentStore", "StoredDocument", "LocalDocumentStore"]
