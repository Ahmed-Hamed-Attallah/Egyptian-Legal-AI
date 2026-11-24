# workers/ingestion_worker.py
"""Standalone ingestion worker entry point.

This module is a CLI wrapper. The IngestionWorker CLASS lives in
src/workers/ingestion_worker.py and is reused verbatim — no duplicated logic.

Why a separate process (PROJECT_ROADMAP Phase 15)
-------------------------------------------------
Parsing, Arabic normalization, legal chunking, embedding and Qdrant writes are
CPU-bound. Running them inside the API process contends with HTTP handlers under
the GIL. This entry point gives them their own process and their own core budget.

Why it builds its own services
------------------------------
It needs job_manager + document_service, which factory.py wires together. It
does NOT need chat/search, so those are left unbuilt to save RAM.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from src.config.logging_config import setup_logging
from src.config.settings import settings
from src.services.factory import build_application_services
from src.workers.ingestion_worker import IngestionWorker

logger = logging.getLogger("workers.ingestion_worker")


def _install_signal_handlers(worker: IngestionWorker) -> None:
    """Graceful stop on SIGTERM/SIGINT: finish the current job, then exit.

    Only the standalone process installs handlers. The embedded worker in the API
    is constructed with install_signal_handlers=False so it cannot hijack
    Uvicorn's own SIGINT/SIGTERM — that was the original shutdown bug.
    """
    loop = asyncio.get_running_loop()

    def _stop(signame: str) -> None:
        logger.info("Received %s; stopping after the current job.", signame)
        worker.request_stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _stop, sig.name)
        except NotImplementedError:  # pragma: no cover - non-POSIX
            signal.signal(sig, lambda *_: _stop(sig.name))


async def _run() -> int:
    setup_logging()

    if not settings.WORKER_ENABLED:
        logger.error(
            "WORKER_ENABLED=false. The worker will not poll jobs. "
            "Set WORKER_ENABLED=true or leave WORKER_EMBEDDED=true in the API."
        )
        return 2

    logger.info(
        "Starting ingestion worker | store=%s | concurrency=%d | poll=%ds | timeout=%ds",
        settings.WORKER_JOB_STORE,
        settings.WORKER_CONCURRENCY,
        settings.WORKER_POLL_INTERVAL_SECONDS,
        settings.WORKER_JOB_TIMEOUT_SECONDS,
    )

    # Builds embedding model + BM25 index + document_service. Expect 10-30 s on
    # a cold cache; that cost is paid here, not in the API process.
    services = build_application_services()

    if services.job_manager is None:
        logger.error(
            "job_manager unavailable. Check WORKER_ENABLED and that "
            "WORKER_JOB_STORE is 'sql' or 'memory'."
        )
        return 3

    if services.document_service is None:
        logger.error("document_service unavailable — ingestion cannot run.")
        return 4

    worker = IngestionWorker(
        job_manager=services.job_manager,
        install_signal_handlers=False,   # we install our own, async-safe, below
    )
    _install_signal_handlers(worker)

    try:
        await worker.run()
    except Exception:
        logger.exception("Worker terminated unexpectedly.")
        return 1

    logger.info("Worker stopped cleanly.")
    return 0


def main() -> None:
    try:
        code = asyncio.run(_run())
    except KeyboardInterrupt:  # pragma: no cover
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()