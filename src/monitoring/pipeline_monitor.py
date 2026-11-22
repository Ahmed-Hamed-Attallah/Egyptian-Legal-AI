
"""
Pipeline stage timing and terminal logging helpers.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

from src.monitoring.metrics_collector import get_monitor

logger = logging.getLogger("rag_pipeline")


@contextmanager
def stage_timer(
    stage: str,
    **tags: Any,
) -> Iterator[Dict[str, Any]]:

    monitor = get_monitor()
    started = time.perf_counter()

    context: Dict[str, Any] = {
        "stage": stage,
        "tags": tags,
        "started_at": started,
    }

    logger.info("▶ START: %s", stage)

    try:
        yield context

    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000

        logger.exception(
            "✗ FAILED: %s | %.3f seconds",
            stage,
            duration_ms / 1000,
        )

        monitor.inc(
            f"{stage}.error",
            tags={"error_type": exc.__class__.__name__, **tags},
        )

        monitor.observe(
            f"{stage}.duration_ms",
            duration_ms,
            tags={"status": "error", **tags},
        )

        monitor.record_event(
            event=f"{stage}.failed",
            tags=tags,
            values={
                "duration_ms": duration_ms,
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            },
        )

        raise

    else:
        duration_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "✓ SUCCESS: %s | %.3f seconds",
            stage,
            duration_ms / 1000,
        )

        monitor.inc(f"{stage}.success", tags=tags)

        monitor.observe(
            f"{stage}.duration_ms",
            duration_ms,
            tags={"status": "success", **tags},
        )

        monitor.record_event(
            event=f"{stage}.completed",
            tags=tags,
            values={"duration_ms": duration_ms},
        )

        context["duration_ms"] = duration_ms
