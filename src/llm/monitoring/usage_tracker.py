import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Any

@dataclass
class UsageStats:
    provider: str
    model: str
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    fallback_count: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    last_success: Optional[float] = None
    last_error_category: Optional[str] = None

class UsageTracker:
    def __init__(self):
        self._stats: Dict[str, UsageStats] = {}
        self._lock = threading.Lock()

    def record_request(self, provider: str, model: str, success: bool, 
                       tokens: int = 0, latency_ms: float = 0.0, 
                       error_category: Optional[str] = None, is_fallback: bool = False):
        key = f"{provider}:{model}"
        with self._lock:
            if key not in self._stats:
                self._stats[key] = UsageStats(provider=provider, model=model)
            stats = self._stats[key]
            stats.request_count += 1
            if is_fallback:
                stats.fallback_count += 1
            if success:
                stats.success_count += 1
                stats.total_tokens += tokens
                stats.total_latency_ms += latency_ms
                stats.last_success = time.time()
            else:
                stats.failure_count += 1
                stats.last_error_category = error_category

    def get_stats(self, provider: str, model: str) -> Optional[Dict[str, Any]]:
        key = f"{provider}:{model}"
        with self._lock:
            stats = self._stats.get(key)
            if not stats: return None
            return {
                "requests": stats.request_count,
                "successes": stats.success_count,
                "failures": stats.failure_count,
                "fallbacks": stats.fallback_count,
                "total_tokens": stats.total_tokens,
                "avg_latency_ms": stats.total_latency_ms / stats.success_count if stats.success_count > 0 else 0,
                "last_success": stats.last_success,
                "last_error_category": stats.last_error_category
            }