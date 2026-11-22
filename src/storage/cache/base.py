"""Abstract cache interface."""
from __future__ import annotations

from typing import Any, Optional


class BaseCache:
    """Abstract cache backend."""

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        raise NotImplementedError

    def clear_namespace(self, namespace: str) -> int:
        """Delete all keys under a namespace. Returns count deleted."""
        raise NotImplementedError
