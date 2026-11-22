# src/workers/__init__.py
"""
Background worker primitives for Egyptian Legal AI.

This package contains job lifecycle, persistence, and worker execution logic.
It must not contain ingestion, retrieval, embedding, chunking, or RAG business
logic. Those responsibilities remain in their original modules.
"""

from src.workers.job_status import JobStatus, JobRecord
from src.workers.job_store import (
    JobStore,
    InMemoryJobStore,
    SqlJobStore,
    create_job_store,
)
from src.workers.job_manager import JobManager
from src.workers.ingestion_worker import IngestionWorker

__all__ = [
    "JobStatus",
    "JobRecord",
    "JobStore",
    "InMemoryJobStore",
    "SqlJobStore",
    "create_job_store",
    "JobManager",
    "IngestionWorker",
]