# src/workers/job_status.py
"""
Job status models for background processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}


ACTIVE_STATUSES = {
    JobStatus.PENDING,
    JobStatus.QUEUED,
    JobStatus.PROCESSING,
}


@dataclass
class JobRecord:
    id: str
    tenant_id: str
    job_type: str
    status: JobStatus
    filename: str
    file_path: str
    file_hash: str
    document_id: Optional[str] = None
    requested_by: Optional[str] = None

    attempt: int = 0
    max_attempts: int = 3
    progress: float = 0.0

    error_code: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    heartbeat_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def can_transition_to(self, new_status: JobStatus) -> bool:
        if self.is_terminal():
            return False

        if self.status == JobStatus.PENDING:
            return new_status in {
                JobStatus.QUEUED,
                JobStatus.PROCESSING,
                JobStatus.CANCELLED,
                JobStatus.FAILED,
            }

        if self.status == JobStatus.QUEUED:
            return new_status in {
                JobStatus.PROCESSING,
                JobStatus.CANCELLED,
                JobStatus.FAILED,
            }

        if self.status == JobStatus.PROCESSING:
            return new_status in {
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.PENDING,
            }

        return False