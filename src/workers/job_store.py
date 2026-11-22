# src/workers/job_store.py
"""
Persistent job store implementations.

The default production implementation uses the existing application database
through SQLAlchemy Core. This avoids introducing a second independent database.

If the project later exposes a canonical SQLAlchemy ORM Base/repository layer,
this store can be refactored to use it without changing JobManager or worker
logic.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    func,
    or_,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config.settings import settings
from src.workers.job_status import ACTIVE_STATUSES, JobRecord, JobStatus

logger = logging.getLogger(__name__)

JOB_TABLE_NAME = "processing_jobs"


def _utcnow() -> datetime:
    return datetime.utcnow()


def _normalize_sync_database_url(url: str) -> str:
    """
    Convert async SQLAlchemy URLs to sync URLs for the worker/job store.

    The application may use asyncpg for async DB access, while the background
    worker uses synchronous SQLAlchemy for operational simplicity.
    """
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql+aiopg://"):
        return url.replace("postgresql+aiopg://", "postgresql+psycopg://", 1)
    if url.startswith("sqlite+aiosqlite://"):
        return url.replace("sqlite+aiosqlite://", "sqlite://", 1)
    return url


class JobStore(ABC):
    @abstractmethod
    def ensure_schema(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def create(self, job: JobRecord) -> JobRecord:
        raise NotImplementedError

    @abstractmethod
    def get(self, job_id: str, tenant_id: Optional[str] = None) -> Optional[JobRecord]:
        raise NotImplementedError

    @abstractmethod
    def list(
        self,
        tenant_id: str,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[JobRecord]:
        raise NotImplementedError

    @abstractmethod
    def find_duplicate(
        self,
        tenant_id: str,
        file_hash: str,
        statuses: Sequence[JobStatus],
    ) -> Optional[JobRecord]:
        raise NotImplementedError

    @abstractmethod
    def claim_next(self) -> Optional[JobRecord]:
        raise NotImplementedError

    @abstractmethod
    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        progress: Optional[float] = None,
        document_id: Optional[str] = None,
        next_run_at: Optional[datetime] = None,
        heartbeat: bool = False,
    ) -> Optional[JobRecord]:
        raise NotImplementedError

    @abstractmethod
    def reclaim_stale(self, stale_after_seconds: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, job_id: str, tenant_id: str) -> Optional[JobRecord]:
        raise NotImplementedError


class InMemoryJobStore(JobStore):
    """
    Development/test-friendly in-memory job store.

    Not suitable for production because jobs are lost on process restart.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: Dict[str, JobRecord] = {}

    def ensure_schema(self) -> None:
        return None

    def create(self, job: JobRecord) -> JobRecord:
        with self._lock:
            self._items[job.id] = job
            return job

    def get(self, job_id: str, tenant_id: Optional[str] = None) -> Optional[JobRecord]:
        with self._lock:
            job = self._items.get(job_id)
            if job is None:
                return None
            if tenant_id and job.tenant_id != tenant_id:
                return None
            return job

    def list(
        self,
        tenant_id: str,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[JobRecord]:
        with self._lock:
            items = [job for job in self._items.values() if job.tenant_id == tenant_id]
            if status is not None:
                items = [job for job in items if job.status == status]
            items = sorted(items, key=lambda x: x.created_at, reverse=True)
            return items[offset : offset + limit]

    def find_duplicate(
        self,
        tenant_id: str,
        file_hash: str,
        statuses: Sequence[JobStatus],
    ) -> Optional[JobRecord]:
        status_set = set(statuses)
        with self._lock:
            for job in self._items.values():
                if (
                    job.tenant_id == tenant_id
                    and job.file_hash == file_hash
                    and job.status in status_set
                ):
                    return job
        return None

    def claim_next(self) -> Optional[JobRecord]:
        now = _utcnow()
        with self._lock:
            candidates = [
                job
                for job in self._items.values()
                if job.status in {JobStatus.PENDING, JobStatus.QUEUED}
                and job.attempt < job.max_attempts
                and (job.next_run_at is None or job.next_run_at <= now)
            ]
            if not candidates:
                return None

            candidates.sort(key=lambda x: x.created_at)
            job = candidates[0]
            job.status = JobStatus.PROCESSING
            job.attempt += 1
            job.started_at = now
            job.heartbeat_at = now
            job.updated_at = now
            job.error_code = None
            job.error_message = None
            return job

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        progress: Optional[float] = None,
        document_id: Optional[str] = None,
        next_run_at: Optional[datetime] = None,
        heartbeat: bool = False,
    ) -> Optional[JobRecord]:
        with self._lock:
            job = self._items.get(job_id)
            if job is None:
                return None

            job.status = status
            job.updated_at = _utcnow()

            if error_code is not None:
                job.error_code = error_code
            if error_message is not None:
                job.error_message = error_message
            if result is not None:
                job.result = result
            if progress is not None:
                job.progress = progress
            if document_id is not None:
                job.document_id = document_id
            if next_run_at is not None:
                job.next_run_at = next_run_at
            if heartbeat:
                job.heartbeat_at = job.updated_at

            if status == JobStatus.PROCESSING and job.started_at is None:
                job.started_at = job.updated_at
            if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
                job.finished_at = job.updated_at

            return job

    def reclaim_stale(self, stale_after_seconds: int) -> int:
        cutoff = _utcnow() - timedelta(seconds=stale_after_seconds)
        count = 0
        with self._lock:
            for job in self._items.values():
                if (
                    job.status == JobStatus.PROCESSING
                    and job.heartbeat_at is not None
                    and job.heartbeat_at < cutoff
                ):
                    job.status = JobStatus.PENDING
                    job.next_run_at = _utcnow()
                    job.heartbeat_at = None
                    job.updated_at = _utcnow()
                    job.error_code = "STALE_RECLAIMED"
                    job.error_message = "Job was reclaimed because heartbeat expired."
                    count += 1
        return count

    def cancel(self, job_id: str, tenant_id: str) -> Optional[JobRecord]:
        with self._lock:
            job = self._items.get(job_id)
            if job is None or job.tenant_id != tenant_id:
                return None
            if job.status not in {JobStatus.PENDING, JobStatus.QUEUED}:
                return job
            job.status = JobStatus.CANCELLED
            job.finished_at = _utcnow()
            job.updated_at = job.finished_at
            return job


class SqlJobStore(JobStore):
    """
    SQLAlchemy Core backed job store.

    Uses the same DATABASE_URL as the application. It intentionally avoids
    depending on a specific ORM Base because the exact existing database model
    registry is not fully visible in the supplied combined source.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        url = _normalize_sync_database_url(database_url or settings.DATABASE_URL)
        connect_args: Dict[str, Any] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self.engine: Engine = create_engine(
            url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self.metadata = MetaData()
        self.table = self._define_table()

    def _define_table(self) -> Table:
        return Table(
            JOB_TABLE_NAME,
            self.metadata,
            Column("id", String(36), primary_key=True),
            Column("tenant_id", String(128), nullable=False),
            Column("requested_by", String(128), nullable=True),
            Column("job_type", String(64), nullable=False, default="document_ingestion"),
            Column("document_id", String(36), nullable=True),
            Column("filename", String(512), nullable=False),
            Column("file_path", Text, nullable=False),
            Column("file_hash", String(64), nullable=False),
            Column("status", String(32), nullable=False),
            Column("attempt", Integer, nullable=False, default=0),
            Column("max_attempts", Integer, nullable=False, default=3),
            Column("progress", Float, nullable=False, default=0.0),
            Column("error_code", String(64), nullable=True),
            Column("error_message", Text, nullable=True),
            Column("result_json", Text, nullable=True),
            Column("metadata_json", Text, nullable=True),
            Column("created_at", DateTime, nullable=False),
            Column("updated_at", DateTime, nullable=False),
            Column("started_at", DateTime, nullable=True),
            Column("finished_at", DateTime, nullable=True),
            Column("heartbeat_at", DateTime, nullable=True),
            Column("next_run_at", DateTime, nullable=True),
            Index("ix_processing_jobs_tenant_status", "tenant_id", "status"),
            Index("ix_processing_jobs_status_created", "status", "created_at"),
            Index("ix_processing_jobs_tenant_hash", "tenant_id", "file_hash"),
            Index("ix_processing_jobs_document_id", "document_id"),
        )

    def ensure_schema(self) -> None:
        try:
            self.table.create(self.engine, checkfirst=True)
        except SQLAlchemyError as exc:
            logger.error("Failed to create job table: %s", exc)
            raise

    def create(self, job: JobRecord) -> JobRecord:
        self.ensure_schema()
        values = self._job_to_values(job)
        with self.engine.begin() as conn:
            conn.execute(self.table.insert().values(**values))
        return job

    def get(self, job_id: str, tenant_id: Optional[str] = None) -> Optional[JobRecord]:
        self.ensure_schema()
        stmt = select(self.table).where(self.table.c.id == job_id)
        if tenant_id:
            stmt = stmt.where(self.table.c.tenant_id == tenant_id)
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return self._row_to_job(row) if row else None

    def list(
        self,
        tenant_id: str,
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[JobRecord]:
        self.ensure_schema()
        stmt = select(self.table).where(self.table.c.tenant_id == tenant_id)
        if status is not None:
            stmt = stmt.where(self.table.c.status == status.value)
        stmt = stmt.order_by(self.table.c.created_at.desc()).limit(limit).offset(offset)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [self._row_to_job(row) for row in rows]

    def find_duplicate(
        self,
        tenant_id: str,
        file_hash: str,
        statuses: Sequence[JobStatus],
    ) -> Optional[JobRecord]:
        self.ensure_schema()
        status_values = [s.value for s in statuses]
        stmt = (
            select(self.table)
            .where(
                and_(
                    self.table.c.tenant_id == tenant_id,
                    self.table.c.file_hash == file_hash,
                    self.table.c.status.in_(status_values),
                )
            )
            .order_by(self.table.c.created_at.desc())
            .limit(1)
        )
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        return self._row_to_job(row) if row else None

    def claim_next(self) -> Optional[JobRecord]:
        self.ensure_schema()
        now = _utcnow()

        selectable = (
            select(self.table.c.id)
            .where(
                and_(
                    self.table.c.status.in_([JobStatus.PENDING.value, JobStatus.QUEUED.value]),
                    self.table.c.attempt < self.table.c.max_attempts,
                    or_(
                        self.table.c.next_run_at.is_(None),
                        self.table.c.next_run_at <= now,
                    ),
                )
            )
            .order_by(self.table.c.created_at.asc())
            .limit(1)
        )

        dialect = self.engine.dialect.name
        if dialect == "postgresql":
            selectable = selectable.with_for_update(skip_locked=True)

        with self.engine.begin() as conn:
            row = conn.execute(selectable).first()
            if row is None:
                return None

            job_id = row.id

            upd = (
                update(self.table)
                .where(
                    and_(
                        self.table.c.id == job_id,
                        self.table.c.status.in_(
                            [JobStatus.PENDING.value, JobStatus.QUEUED.value]
                        ),
                    )
                )
                .values(
                    status=JobStatus.PROCESSING.value,
                    attempt=self.table.c.attempt + 1,
                    started_at=now,
                    heartbeat_at=now,
                    updated_at=now,
                    error_code=None,
                    error_message=None,
                )
            )

            result = conn.execute(upd)
            if result.rowcount == 0:
                return None

        return self.get(job_id)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        progress: Optional[float] = None,
        document_id: Optional[str] = None,
        next_run_at: Optional[datetime] = None,
        heartbeat: bool = False,
    ) -> Optional[JobRecord]:
        self.ensure_schema()
        now = _utcnow()

        values: Dict[str, Any] = {
            "status": status.value,
            "updated_at": now,
        }

        if error_code is not None:
            values["error_code"] = error_code
        if error_message is not None:
            values["error_message"] = error_message
        if result is not None:
            values["result_json"] = json.dumps(result, ensure_ascii=False)
        if progress is not None:
            values["progress"] = progress
        if document_id is not None:
            values["document_id"] = document_id
        if next_run_at is not None:
            values["next_run_at"] = next_run_at
        if heartbeat:
            values["heartbeat_at"] = now

        if status == JobStatus.PROCESSING:
            values["started_at"] = now

        if status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            values["finished_at"] = now

        with self.engine.begin() as conn:
            conn.execute(update(self.table).where(self.table.c.id == job_id).values(**values))

        return self.get(job_id)

    def reclaim_stale(self, stale_after_seconds: int) -> int:
        self.ensure_schema()
        cutoff = _utcnow() - timedelta(seconds=stale_after_seconds)
        now = _utcnow()

        with self.engine.begin() as conn:
            result = conn.execute(
                update(self.table)
                .where(
                    and_(
                        self.table.c.status == JobStatus.PROCESSING.value,
                        self.table.c.heartbeat_at.is_not(None),
                        self.table.c.heartbeat_at < cutoff,
                    )
                )
                .values(
                    status=JobStatus.PENDING.value,
                    next_run_at=now,
                    updated_at=now,
                    heartbeat_at=None,
                    error_code="STALE_RECLAIMED",
                    error_message="Job was reclaimed because heartbeat expired.",
                )
            )
            return int(result.rowcount or 0)

    def cancel(self, job_id: str, tenant_id: str) -> Optional[JobRecord]:
        self.ensure_schema()
        now = _utcnow()

        with self.engine.begin() as conn:
            result = conn.execute(
                update(self.table)
                .where(
                    and_(
                        self.table.c.id == job_id,
                        self.table.c.tenant_id == tenant_id,
                        self.table.c.status.in_([JobStatus.PENDING.value, JobStatus.QUEUED.value]),
                    )
                )
                .values(
                    status=JobStatus.CANCELLED.value,
                    finished_at=now,
                    updated_at=now,
                )
            )
            if result.rowcount == 0:
                return self.get(job_id, tenant_id=tenant_id)

        return self.get(job_id, tenant_id=tenant_id)

    def _job_to_values(self, job: JobRecord) -> Dict[str, Any]:
        return {
            "id": job.id,
            "tenant_id": job.tenant_id,
            "requested_by": job.requested_by,
            "job_type": job.job_type,
            "document_id": job.document_id,
            "filename": job.filename,
            "file_path": job.file_path,
            "file_hash": job.file_hash,
            "status": job.status.value,
            "attempt": job.attempt,
            "max_attempts": job.max_attempts,
            "progress": job.progress,
            "error_code": job.error_code,
            "error_message": job.error_message,
            "result_json": json.dumps(job.result, ensure_ascii=False) if job.result else None,
            "metadata_json": json.dumps(job.metadata, ensure_ascii=False) if job.metadata else None,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "heartbeat_at": job.heartbeat_at,
            "next_run_at": job.next_run_at,
        }

    def _row_to_job(self, row: Any) -> JobRecord:
        result_json = row.get("result_json")
        metadata_json = row.get("metadata_json")

        return JobRecord(
            id=row["id"],
            tenant_id=row["tenant_id"],
            requested_by=row.get("requested_by"),
            job_type=row.get("job_type", "document_ingestion"),
            document_id=row.get("document_id"),
            filename=row["filename"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            status=JobStatus(row["status"]),
            attempt=row.get("attempt", 0),
            max_attempts=row.get("max_attempts", 3),
            progress=row.get("progress", 0.0),
            error_code=row.get("error_code"),
            error_message=row.get("error_message"),
            result=json.loads(result_json) if result_json else None,
            metadata=json.loads(metadata_json) if metadata_json else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            heartbeat_at=row.get("heartbeat_at"),
            next_run_at=row.get("next_run_at"),
        )


def create_job_store(store_type: Optional[str] = None) -> JobStore:
    kind = (store_type or settings.WORKER_JOB_STORE).lower()

    if kind == "memory":
        logger.warning("Using in-memory job store. Jobs will not survive restarts.")
        return InMemoryJobStore()

    if kind == "sql":
        store = SqlJobStore()
        store.ensure_schema()
        return store

    raise ValueError(f"Unsupported job store type: {kind}")