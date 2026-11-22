"""Unified logging configuration for Egyptian Legal AI.

`setup_logging()` MUST be called once at process entry (API lifespan, worker
main, CLI scripts). `get_logger()` only fetches a logger — it does not configure
handlers. Without setup, the root logger has no handlers and Python's
`lastResort` handler silently drops everything below WARNING, which makes all
INFO-level request/pipeline telemetry disappear.

Records are handed to a QueueHandler and written by a background listener
thread, so rotating-file I/O never runs on a request thread.
"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import queue
import sys
import threading
from pathlib import Path
from typing import Optional

from src.config.settings import settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False
_lock = threading.Lock()
_listener: Optional[logging.handlers.QueueListener] = None

# Third-party loggers that would otherwise flood the file at INFO.
_NOISY = (
    "transformers", "sentence_transformers", "urllib3", "qdrant_client",
    "httpx", "httpcore", "huggingface_hub", "filelock", "PIL",
)


class _TracebackPreservingQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler that keeps tracebacks.

    The stdlib implementation clears `exc_info` after formatting; with no
    formatter attached that would silently discard every stack trace. We fold
    it into `exc_text` instead, which the downstream formatter still renders.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        rec = logging.makeLogRecord(record.__dict__)
        formatter = logging.Formatter()
        if rec.exc_info:
            rec.exc_text = formatter.formatException(rec.exc_info)
            rec.exc_info = None
        if rec.stack_info:
            rec.stack_info = formatter.formatStack(rec.stack_info)
        return rec


def _build_sinks(level: int) -> list[logging.Handler]:
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(level)

    logs_dir: Path = settings.LOG_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=logs_dir / "app.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",          # Arabic text must survive the round-trip
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    return [console, file_handler]


def setup_logging(force: bool = False) -> None:
    """Install handlers exactly once per process. Safe to call repeatedly."""
    global _configured, _listener

    with _lock:
        if _configured and not force:
            return

        if _listener is not None:
            _listener.stop()
            _listener = None

        level = getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO)
        sinks = _build_sinks(level)

        log_queue: queue.Queue = queue.Queue(-1)
        _listener = logging.handlers.QueueListener(
            log_queue, *sinks, respect_handler_level=True
        )
        _listener.start()

        root = logging.getLogger()
        for handler in list(root.handlers):
            root.removeHandler(handler)
        root.setLevel(level)
        root.addHandler(_TracebackPreservingQueueHandler(log_queue))

        # Route Uvicorn through the same queue instead of duplicating handlers.
        # propagate=True keeps `access_logger.hasHandlers()` truthy, which is
        # what Uvicorn checks to decide whether to emit access logs at all.
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            uv_logger = logging.getLogger(name)
            for handler in list(uv_logger.handlers):
                uv_logger.removeHandler(handler)
            uv_logger.setLevel(level)
            uv_logger.propagate = True

        for name in _NOISY:
            logging.getLogger(name).setLevel(logging.WARNING)

        logging.captureWarnings(True)
        _configured = True

        root.info(
            "Logging initialized | level=%s | env=%s | file=%s",
            logging.getLevelName(level), settings.APP_ENV, settings.LOG_DIR / "app.log",
        )


def shutdown_logging() -> None:
    """Flush and stop the listener. Registered with atexit."""
    global _listener, _configured
    with _lock:
        if _listener is not None:
            _listener.stop()
            _listener = None
        _configured = False


atexit.register(shutdown_logging)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Does NOT configure handlers. Call `setup_logging()` at process entry first,
    otherwise INFO records are dropped by `logging.lastResort`.
    """
    return logging.getLogger(name)