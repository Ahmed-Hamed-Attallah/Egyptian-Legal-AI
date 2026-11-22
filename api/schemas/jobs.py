# api/schemas/jobs.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.workers.job_status import JobRecord, JobStatus


class JobResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    tenant_id: str
    job_type: str
    status: JobStatus
    filename: str
    document_id: Optional[str] = None
    requested_by: Optional[str] = None
    attempt: int
    max_attempts: int
    progress: float
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None

    @classmethod
    def from_record(cls, record: JobRecord) -> "JobResponse":
        return cls(
            id=record.id,
            tenant_id=record.tenant_id,
            job_type=record.job_type,
            status=record.status,
            filename=record.filename,
            document_id=record.document_id,
            requested_by=record.requested_by,
            attempt=record.attempt,
            max_attempts=record.max_attempts,
            progress=record.progress,
            error_code=record.error_code,
            error_message=record.error_message,
            result=record.result,
            created_at=record.created_at,
            updated_at=record.updated_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            next_run_at=record.next_run_at,
        )


class JobListResponse(BaseModel):
    items: List[JobResponse] = Field(default_factory=list)
    total: int


class JobCancelResponse(BaseModel):
    id: str
    status: JobStatus
    message: str