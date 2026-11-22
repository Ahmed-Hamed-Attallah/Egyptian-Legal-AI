"""
Redis cache implementation with graceful degradation.

When Redis is unavailable or unconfigured, the application falls back to a
NoOpCache that logs a warning once and otherwise does nothing. This keeps
the system usable in development without Redis while still providing clear
feedback that caching is disabled.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.config.settings import settings
from src.domain.exceptions import CacheError
from src.storage.cache.base import BaseCache

logger = logging.getLogger(__name__)


def _default_serializer(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _default_deserializer(raw: str) -> Any:
    return json.loads(raw)


class RedisCache(BaseCache):
    """Redis-backed cache."""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: Optional[int] = None,
        key_prefix: str = "egyptian_legal:",
    ):
        self._url = redis_url or settings.REDIS_URL
        self._default_ttl = default_ttl if default_ttl is not None else settings.CACHE_DEFAULT_TTL
        self._prefix = key_prefix
        self._client = None
        self._connect()

    def _connect(self) -> None:
        try:
            import redis  # local import — optional dependency
            self._client = redis.from_url(self._url, decode_responses=True)
            self._client.ping()
            logger.info("Redis cache connected: %s", self._mask_url())
        except Exception as exc:
            logger.warning("Redis unavailable, cache disabled: %s", exc)
            self._client = None

    def _mask_url(self) -> str:
        if "@" not in self._url:
            return self._url
        prefix, suffix = self._url.rsplit("@", 1)
        scheme_end = prefix.find("://")
        if scheme_end == -1:
            return self._url
        return f"{prefix[:scheme_end + 3]}***@{suffix}"

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _ensure_connected(self) -> None:
        if self._client is None:
            raise CacheError("Redis cache is not connected")

    def get(self, key: str) -> Optional[Any]:
        self._ensure_connected()
        try:
            raw = self._client.get(self._full_key(key))
            if raw is None:
                return None
            return _default_deserializer(raw)
        except Exception as exc:
            logger.warning("Cache GET failed for %s: %s", key, exc)
            raise CacheError(f"Cache get failed: {exc}", cache_key=key) from exc

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self._ensure_connected()
        effective_ttl = ttl if ttl is not None else self._default_ttl
        try:
            payload = _default_serializer(value)
            if effective_ttl and effective_ttl > 0:
                return bool(self._client.setex(self._full_key(key), effective_ttl, payload))
            return bool(self._client.set(self._full_key(key), payload))
        except Exception as exc:
            logger.warning("Cache SET failed for %s: %s", key, exc)
            raise CacheError(f"Cache set failed: {exc}", cache_key=key) from exc

    def delete(self, key: str) -> bool:
        self._ensure_connected()
        try:
            return bool(self._client.delete(self._full_key(key)))
        except Exception as exc:
            logger.warning("Cache DELETE failed for %s: %s", key, exc)
            raise CacheError(f"Cache delete failed: {exc}", cache_key=key) from exc

    def exists(self, key: str) -> bool:
        self._ensure_connected()
        try:
            return bool(self._client.exists(self._full_key(key)))
        except Exception as exc:
            logger.warning("Cache EXISTS failed for %s: %s", key, exc)
            return False

    def clear_namespace(self, namespace: str) -> int:
        self._ensure_connected()
        pattern = f"{self._prefix}{namespace}*"
        try:
            keys = list(self._client.scan_iter(match=pattern, count=100))
            if not keys:
                return 0
            return int(self._client.delete(*keys))
        except Exception as exc:
            logger.warning("Cache CLEAR_NAMESPACE failed for %s: %s", namespace, exc)
            raise CacheError(f"Cache clear failed: {exc}") from exc


class NoOpCache(BaseCache):
    """
    No-op cache used when Redis is unavailable.

    Logs a single warning on first use, then stays silent to avoid spam.
    """

    _warned = False

    def _warn_once(self) -> None:
        if not NoOpCache._warned:
            logger.warning(
                "Cache operations are no-ops: Redis is not configured or unavailable. "
                "Set REDIS_URL and CACHE_ENABLED=true to enable caching."
            )
            NoOpCache._warned = True

    def get(self, key: str) -> Optional[Any]:
        self._warn_once()
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        self._warn_once()
        return False

    def delete(self, key: str) -> bool:
        self._warn_once()
        return False

    def exists(self, key: str) -> bool:
        self._warn_once()
        return False

    def clear_namespace(self, namespace: str) -> int:
        self._warn_once()
        return 0


_cache_instance: Optional[BaseCache] = None


def get_cache() -> BaseCache:
    """Return the application-wide cache, initializing as needed."""
    global _cache_instance
    if _cache_instance is not None:
        return _cache_instance

    if not settings.CACHE_ENABLED or not settings.REDIS_URL:
        logger.info("Cache disabled by configuration — using NoOpCache")
        _cache_instance = NoOpCache()
        return _cache_instance

    try:
        _cache_instance = RedisCache()
    except Exception as exc:
        logger.warning("Failed to initialize Redis cache: %s — using NoOpCache", exc)
        _cache_instance = NoOpCache()
    return _cache_instance


def reset_cache() -> None:
    """Reset the cache singleton. Intended for tests."""
    global _cache_instance
    _cache_instance = None
