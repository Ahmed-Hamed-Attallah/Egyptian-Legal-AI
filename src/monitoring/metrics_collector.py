# src/monitoring/metrics_collector.py
"""
Lightweight in-process metrics collector.

Design goals:

- No mandatory external dependency.
- Optional JSONL export.
- Safe for FastAPI, workers, and evaluation runner.
- Suitable for later export to Prometheus/OpenTelemetry without changing callers.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from src.config.settings import settings

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tag_key(name: str, tags: Optional[Dict[str, Any]]) -> str:
    if not tags:
        return name

    normalized = ",".join(
        f"{key}={value}"
        for key, value in sorted(tags.items(), key=lambda item: str(item[0]))
        if value is not None
    )
    return f"{name}|{normalized}" if normalized else name


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[k])


@dataclass
class MetricSnapshot:
    counters: Dict[str, int] = field(default_factory=dict)
    histograms: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(default_factory=_utc_now_iso)


class MetricsCollector:
    def __init__(
        self,
        enabled: bool = True,
        export_path: Optional[str] = None,
        max_events: int = 1000,
    ) -> None:
        self.enabled = enabled
        self.export_path = Path(export_path) if export_path else None
        self.max_events = max_events

        self._lock = threading.RLock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._events: Deque[Dict[str, Any]] = deque(maxlen=max_events)

        if self.export_path is not None:
            self.export_path.parent.mkdir(parents=True, exist_ok=True)

    def inc(self, name: str, value: int = 1, tags: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return

        key = _tag_key(name, tags)
        with self._lock:
            self._counters[key] += value

    def observe(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        key = _tag_key(name, tags)
        with self._lock:
            self._histograms[key].append(float(value))

    def record_event(
        self,
        event: str,
        tags: Optional[Dict[str, Any]] = None,
        values: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        payload = {
            "timestamp": _utc_now_iso(),
            "event": event,
            "tags": tags or {},
            "values": values or {},
        }

        with self._lock:
            self._events.append(payload)
            self._append_event_to_file(payload)

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            counters = dict(self._counters)
            histograms = {
                key: {
                    "count": len(values),
                    "mean": sum(values) / len(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "p50": _percentile(values, 50),
                    "p90": _percentile(values, 90),
                    "p95": _percentile(values, 95),
                    "p99": _percentile(values, 99),
                }
                for key, values in self._histograms.items()
            }
            events = list(self._events)

        return MetricSnapshot(
            counters=counters,
            histograms=histograms,
            events=events,
        )

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._events.clear()

    def _append_event_to_file(self, payload: Dict[str, Any]) -> None:
        if self.export_path is None:
            return

        try:
            with self.export_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to export metric event: %s", exc)


_monitor: Optional[MetricsCollector] = None


def get_monitor() -> MetricsCollector:
    global _monitor

    if _monitor is None:
        enabled = bool(
            getattr(settings, "MONITORING_ENABLED", False)
            or getattr(settings, "METRICS_ENABLED", False)
        )
        export_path = getattr(settings, "MONITORING_EXPORT_PATH", "") or None

        _monitor = MetricsCollector(
            enabled=enabled,
            export_path=export_path,
        )

    return _monitor


def reset_monitor() -> None:
    global _monitor
    _monitor = None