# src/workers/ingestion_worker.py
"""
Background ingestion worker.

Executes existing document ingestion services. Does not implement parsing,
chunking, embedding, or vector indexing logic.

Signal handling
---------------
`install_signal_handlers` must be False when the worker is embedded in the API
process. Registering SIGINT/SIGTERM handlers on a shared event loop replaces
Uvicorn's own handlers and prevents the server from shutting down or reloading.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from src.config.settings import settings
from src.domain.exceptions import (
    DocumentChunkingError,
    EmbeddingError,
    IngestionError,
    ProviderUnavailableError,
    UnsupportedFormatError,
    VectorDatabaseError,
)
from src.monitoring.pipeline_monitor import stage_timer
from src.workers.job_manager import JobManager
from src.workers.job_status import JobRecord, JobStatus

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (
    ProviderUnavailableError,
    VectorDatabaseError,
    EmbeddingError,
    TimeoutError,
    ConnectionError,
    OSError,
)

PERMANENT_EXCEPTIONS = (
    UnsupportedFormatError,
    ValueError,
    DocumentChunkingError,
    IngestionError,
)

# Upper bound on waiting for in-flight jobs during shutdown. Threads cannot be
# cancelled, so we stop waiting instead of blocking the host process forever.
SHUTDOWN_GRACE_SECONDS = 15.0


class IngestionWorker:
    def __init__(
        self,
        job_manager: JobManager,
        concurrency: Optional[int] = None,
        poll_interval: Optional[float] = None,
        job_timeout: Optional[int] = None,
        reclaim_stale_after: Optional[int] = None,
        install_signal_handlers: bool = True,
    ) -> None:
        self.job_manager = job_manager
        self.concurrency = concurrency or settings.WORKER_CONCURRENCY
        self.poll_interval = poll_interval or settings.WORKER_POLL_INTERVAL_SECONDS
        self.job_timeout = job_timeout or settings.WORKER_JOB_TIMEOUT_SECONDS
        self.reclaim_stale_after = (
            reclaim_stale_after or settings.WORKER_RECLAIM_STALE_AFTER_SECONDS
        )
        self._install_signal_handlers = install_signal_handlers
        self._stop_event = asyncio.Event()
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._tasks: set[asyncio.Task] = set()

    def request_stop(self, *_args: Any) -> None:
        logger.info("Ingestion worker stop requested.")
        self._stop_event.set()

    async def _update_status(self, **kwargs: Any) -> None:
        """Job-store writes are synchronous — keep them off the event loop."""
        await asyncio.to_thread(self.job_manager.store.update_status, **kwargs)

    async def run(self) -> None:
        if self._install_signal_handlers:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, self.request_stop)
                except NotImplementedError:
                    pass
        else:
            logger.info(
                "Embedded mode: worker will not install signal handlers "
                "(host process owns SIGINT/SIGTERM)."
            )

        logger.info(
            "Starting ingestion worker concurrency=%s poll_interval=%s job_timeout=%s",
            self.concurrency,
            self.poll_interval,
            self.job_timeout,
        )

        try:
            while not self._stop_event.is_set():
                reclaimed = await asyncio.to_thread(
                    self.job_manager.store.reclaim_stale, self.reclaim_stale_after
                )
                if reclaimed:
                    logger.warning("Reclaimed %s stale processing jobs.", reclaimed)

                launched_any = await self._launch_available_jobs()
                if not launched_any:
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(), timeout=self.poll_interval
                        )
                    except asyncio.TimeoutError:
                        pass
        finally:
            await self._shutdown()

    async def _launch_available_jobs(self) -> bool:
        launched = False
        while not self._stop_event.is_set():
            job = await asyncio.to_thread(self.job_manager.store.claim_next)
            if job is None:
                break
            task = asyncio.create_task(self._run_job_with_semaphore(job))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            launched = True
        return launched

    async def _run_job_with_semaphore(self, job: JobRecord) -> None:
        async with self._semaphore:
            try:
                await asyncio.wait_for(self.process_job(job), timeout=self.job_timeout)
            except asyncio.TimeoutError:
                logger.error("Job timed out job_id=%s", job.id)
                await self._fail_job(
                    job=job,
                    exc=TimeoutError(
                        f"Job exceeded timeout of {self.job_timeout} seconds."
                    ),
                    error_code="TIMEOUT",
                )
            except asyncio.CancelledError:
                logger.warning("Job cancelled by worker shutdown job_id=%s", job.id)
                await self._update_status(
                    job_id=job.id,
                    status=JobStatus.PENDING,
                    error_code="WORKER_SHUTDOWN",
                    error_message="Job returned to queue because worker shut down.",
                    next_run_at=datetime.utcnow(),
                )
                raise
            except Exception as exc:
                logger.exception("Unexpected worker error job_id=%s", job.id)
                await self._fail_job(job=job, exc=exc, error_code=exc.__class__.__name__)

    async def process_job(self, job: JobRecord) -> None:
        logger.info(
            "Processing job job_id=%s tenant_id=%s document_id=%s attempt=%s",
            job.id, job.tenant_id, job.document_id, job.attempt,
        )
        await self._update_status(
            job_id=job.id, status=JobStatus.PROCESSING, progress=0.05, heartbeat=True
        )

        path = Path(job.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Stored upload file not found: {job.file_path}")

        file_bytes = await asyncio.to_thread(path.read_bytes)

        document_service = self.job_manager.document_service
        if document_service is None:
            raise RuntimeError("Document service is not configured for worker execution.")

        ingest_method = getattr(document_service, "ingest_upload", None)
        if not callable(ingest_method):
            raise RuntimeError(
                "Configured document service does not expose ingest_upload()."
            )

        try:
            with stage_timer(
                "worker.ingestion",
                job_id=job.id,
                tenant_id=job.tenant_id,
                document_id=job.document_id,
            ):
                result = await asyncio.to_thread(
                    ingest_method,
                    file_bytes=file_bytes,
                    filename=job.filename,
                    tenant_id=job.tenant_id,
                    document_id=job.document_id,
                    owner_id=job.requested_by, 
                )
        except RETRYABLE_EXCEPTIONS as exc:
            await self._retry_or_fail_job(job=job, exc=exc, error_code=exc.__class__.__name__)
            return
        except PERMANENT_EXCEPTIONS as exc:
            await self._fail_job(job=job, exc=exc, error_code=exc.__class__.__name__)
            return
        except Exception as exc:
            if self._looks_retryable(exc):
                await self._retry_or_fail_job(job=job, exc=exc, error_code="TRANSIENT_ERROR")
            else:
                await self._fail_job(job=job, exc=exc, error_code="UNEXPECTED_ERROR")
            return

        await self._update_status(
            job_id=job.id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            result=result if isinstance(result, dict) else {"result": result},
            document_id=job.document_id,
            heartbeat=True,
        )
        logger.info("Job completed job_id=%s document_id=%s", job.id, job.document_id)

    async def _retry_or_fail_job(
        self, job: JobRecord, exc: Exception, error_code: str
    ) -> None:
        if job.attempt < job.max_attempts:
            delay = settings.WORKER_RETRY_BACKOFF_SECONDS * max(1, job.attempt)
            next_run = datetime.utcnow() + timedelta(seconds=delay)
            await self._update_status(
                job_id=job.id,
                status=JobStatus.PENDING,
                error_code=error_code,
                error_message=str(exc),
                next_run_at=next_run,
                progress=min(0.95, job.progress + 0.05),
            )
            logger.warning(
                "Job scheduled for retry job_id=%s attempt=%s/%s next_run_at=%s error=%s",
                job.id, job.attempt, job.max_attempts, next_run.isoformat(), exc,
            )
            return
        await self._fail_job(job=job, exc=exc, error_code=error_code)

    async def _fail_job(self, job: JobRecord, exc: Exception, error_code: str) -> None:
        await self._update_status(
            job_id=job.id,
            status=JobStatus.FAILED,
            error_code=error_code,
            error_message=str(exc),
            progress=job.progress,
        )
        logger.error(
            "Job failed permanently job_id=%s error_code=%s error=%s",
            job.id, error_code, exc,
        )

    @staticmethod
    def _looks_retryable(exc: Exception) -> bool:
        text = str(exc).lower()
        retry_markers = (
            "timeout", "timed out", "connection", "temporarily unavailable",
            "rate limit", "too many requests", "service unavailable",
            "bad gateway", "gateway timeout",
        )
        return any(marker in text for marker in retry_markers)

    async def _shutdown(self) -> None:
        logger.info("Shutting down ingestion worker. Active tasks=%s", len(self._tasks))
        if self._tasks:
            done, pending = await asyncio.wait(
                self._tasks, timeout=SHUTDOWN_GRACE_SECONDS
            )
            if pending:
                logger.warning(
                    "%s job(s) still running after %.0fs grace period; "
                    "they will be reclaimed as stale by the next worker.",
                    len(pending), SHUTDOWN_GRACE_SECONDS,
                )
                for task in pending:
                    task.cancel()
        logger.info("Ingestion worker shutdown complete.")