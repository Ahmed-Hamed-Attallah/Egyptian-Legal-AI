# src/workers/job_manager.py
"""
Application-facing job manager.

This module is used by the API to submit and inspect jobs. It does not execute
ingestion itself. Execution happens in IngestionWorker.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from src.config.settings import settings
from src.workers.job_status import ACTIVE_STATUSES, JobRecord, JobStatus
from src.workers.job_store import JobStore

logger = logging.getLogger(__name__)

try:
    from src.security.constants import MAX_FILE_SIZE_BYTES, SUPPORTED_EXTENSIONS
except ImportError:
    MAX_FILE_SIZE_BYTES = getattr(settings, "MAX_UPLOAD_SIZE_BYTES", 50 * 1024 * 1024)
    SUPPORTED_EXTENSIONS = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in getattr(settings, "ALLOWED_FILE_TYPES", [".pdf", ".docx", ".txt", ".md"])
    }


class JobManager:
    def __init__(
        self,
        store: JobStore,
        document_service: Optional[object] = None,
    ) -> None:
        self.store = store
        self.document_service = document_service
        self.store.ensure_schema()

    def submit_document_job(
        self,
        file_bytes: bytes,
        filename: str,
        tenant_id: str,
        requested_by: Optional[str] = None,
    ) -> JobRecord:
        self._validate_upload(file_bytes=file_bytes, filename=filename)

        file_hash = hashlib.sha256(file_bytes).hexdigest()

        existing = self.store.find_duplicate(
            tenant_id=tenant_id,
            file_hash=file_hash,
            statuses=list(ACTIVE_STATUSES) + [JobStatus.COMPLETED],
        )
        if existing is not None:
            logger.info(
                "Returning existing job for duplicate upload tenant_id=%s file_hash=%s job_id=%s",
                tenant_id,
                file_hash,
                existing.id,
            )
            return existing

        job_id = str(uuid4())
        document_id = str(uuid4())
        file_path = self._store_upload(
            file_bytes=file_bytes,
            tenant_id=tenant_id,
            job_id=job_id,
            filename=filename,
        )

        now = datetime.utcnow()
        job = JobRecord(
            id=job_id,
            tenant_id=tenant_id,
            requested_by=requested_by,
            job_type="document_ingestion",
            status=JobStatus.PENDING,
            filename=filename,
            file_path=str(file_path),
            file_hash=file_hash,
            document_id=document_id,
            attempt=0,
            max_attempts=max(1, settings.WORKER_MAX_RETRIES + 1),
            progress=0.0,
            created_at=now,
            updated_at=now,
            next_run_at=now,
            metadata={
                "source": "api_upload_job",
                "size_bytes": len(file_bytes),
            },
        )

        return self.store.create(job)

    def get_job(self, job_id: str, tenant_id: str) -> Optional[JobRecord]:
        return self.store.get(job_id, tenant_id=tenant_id)

    def list_jobs(
        self,
        tenant_id: str,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[JobRecord]:
        return self.store.list(tenant_id=tenant_id, status=status, limit=limit, offset=offset)

    def cancel_job(self, job_id: str, tenant_id: str) -> Optional[JobRecord]:
        return self.store.cancel(job_id=job_id, tenant_id=tenant_id)

    def _validate_upload(self, file_bytes: bytes, filename: str) -> None:
        if not file_bytes:
            raise ValueError("Uploaded file is empty.")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES} bytes."
            )

        suffix = Path(filename).suffix.lower()
        if suffix and suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {suffix}. Allowed types: {sorted(SUPPORTED_EXTENSIONS)}"
            )

    def _store_upload(
        self,
        file_bytes: bytes,
        tenant_id: str,
        job_id: str,
        filename: str,
    ) -> Path:
        safe_tenant = self._safe_component(tenant_id, default="default")
        safe_job = self._safe_component(job_id, default="job")
        safe_name = self._safe_component(Path(filename).name, default="upload")

        base_dir = settings.DOCUMENT_STORAGE_DIR / "jobs" / safe_tenant / safe_job
        base_dir.mkdir(parents=True, exist_ok=True)

        path = base_dir / safe_name
        path.write_bytes(file_bytes)
        return path

    @staticmethod
    def _safe_component(value: str, default: str = "default") -> str:
        value = (value or default).strip()
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
        return safe[:128] or default