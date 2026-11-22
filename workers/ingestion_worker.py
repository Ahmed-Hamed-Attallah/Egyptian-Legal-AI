# workers/ingestion_worker.py
"""
Standalone ingestion worker process.

Run with:

    python -m workers.ingestion_worker
"""

from __future__ import annotations

import asyncio
import logging

from src.config.logging_config import get_logger as configure_logging
configure_logging(__name__)
from src.services.factory import build_application_services
from src.workers.ingestion_worker import IngestionWorker

logger = logging.getLogger(__name__)


async def main() -> None:
    configure_logging(__name__)

    services = build_application_services()

    if services.job_manager is None:
        raise RuntimeError("Job manager could not be initialized.")

    worker = IngestionWorker(job_manager=services.job_manager)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())