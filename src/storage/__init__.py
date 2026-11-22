"""
Storage Module — Phase 11.

Provides database, document storage, and cache abstractions.
"""
from src.storage.database.database import (
    Database,
    get_database,
    get_session,
)
from src.storage.document_store.base import BaseDocumentStore
from src.storage.document_store.local_store import LocalDocumentStore
from src.storage.cache.base import BaseCache
from src.storage.cache.redis_cache import RedisCache, NoOpCache

__all__ = [
    "Database",
    "get_database",
    "get_session",
    "BaseDocumentStore",
    "LocalDocumentStore",
    "BaseCache",
    "RedisCache",
    "NoOpCache",
]
