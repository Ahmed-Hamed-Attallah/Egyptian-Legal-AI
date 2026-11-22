"""API lifespan: construct long-lived services once and expose them on app.state."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from src.config.logging_config import setup_logging
from src.config.settings import settings
from src.services.auth_service import AuthService
from src.services.factory import ApplicationServices, build_application_services
from src.storage.database.database import get_database

# Configure handlers BEFORE any logger is used. `get_logger` does not do this.
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()  # idempotent; covers reloader-spawned children
    app.state.settings = settings
    app.state.startup_errors: list[str] = []

    # Service construction is synchronous and includes Qdrant scroll + BM25
    # index build. Time it explicitly: this is the phase that blocks startup.
    t0 = time.perf_counter()
    services: ApplicationServices = build_application_services()
    logger.info("Service construction completed in %.2f s", time.perf_counter() - t0)

    app.state.conversation_service = services.conversation_service
    app.state.chat_service = services.chat_service
    app.state.search_service = services.search_service
    app.state.document_service = services.document_service
    app.state.admin_service = services.admin_service
    app.state.job_manager = services.job_manager
    app.state.startup_errors = services.startup_errors

    # --- Authentication (Phase 12/13) -------------------------------------
    app.state.auth_service = None
    try:
        app.state.auth_service = AuthService(database=get_database())
        logger.info("Authentication service provisioned.")
    except Exception as exc:
        logger.error("Authentication service failed to start: %s", exc)
        app.state.startup_errors.append(f"auth_service: {exc}")

    # --- Embedded worker (development only) -------------------------------
    embedded_worker = None
    embedded_worker_task: Optional[asyncio.Task] = None
    if settings.WORKER_EMBEDDED and services.job_manager is not None:
        try:
            from src.workers.ingestion_worker import IngestionWorker

            embedded_worker = IngestionWorker(
                job_manager=services.job_manager,
                install_signal_handlers=False,  # Uvicorn owns SIGINT/SIGTERM
            )
            embedded_worker_task = asyncio.create_task(embedded_worker.run())
            logger.info("Embedded ingestion worker started in API process.")
        except Exception as exc:
            logger.error("Embedded worker failed to start: %s", exc)
            app.state.startup_errors.append(f"embedded_worker: {exc}")

    logger.info(
        "API startup completed in %.2f s", time.perf_counter() - t0
    )
    if app.state.startup_errors:
        logger.warning(
            "Startup completed with degraded components: %s", app.state.startup_errors
        )

    try:
        yield
    finally:
        if embedded_worker is not None:
            logger.info("Stopping embedded ingestion worker.")
            embedded_worker.request_stop()
        if embedded_worker_task is not None:
            try:
                await asyncio.wait_for(embedded_worker_task, timeout=20.0)
            except asyncio.TimeoutError:
                logger.warning("Embedded worker did not stop in time; cancelling.")
                embedded_worker_task.cancel()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error("Embedded worker shutdown error: %s", exc)
        logger.info("Egyptian Legal AI API shutdown completed.")