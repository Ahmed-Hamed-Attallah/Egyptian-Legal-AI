# api/routes/jobs.py
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

from api.dependencies import AuthContext, get_auth_context, get_job_manager
from api.schemas.jobs import JobCancelResponse, JobListResponse, JobResponse
from src.workers.job_manager import JobManager
from src.workers.job_status import JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/documents", response_model=JobResponse, status_code=202)
async def submit_document_job(
    file: UploadFile,
    auth_context: AuthContext = Depends(get_auth_context),
    job_manager: JobManager = Depends(get_job_manager),
) -> JobResponse:
    contents = await file.read()

    job = job_manager.submit_document_job(
        file_bytes=contents,
        filename=file.filename or "upload",
        tenant_id=auth_context.tenant_id,
        requested_by=getattr(auth_context, "user_id", None),
    )

    return JobResponse.from_record(job)


@router.get("", response_model=JobListResponse)
def list_jobs(
    status: Optional[JobStatus] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    auth_context: AuthContext = Depends(get_auth_context),
    job_manager: JobManager = Depends(get_job_manager),
) -> JobListResponse:
    jobs = job_manager.list_jobs(
        tenant_id=auth_context.tenant_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    items = [JobResponse.from_record(job) for job in jobs]
    return JobListResponse(items=items, total=len(items))


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    auth_context: AuthContext = Depends(get_auth_context),
    job_manager: JobManager = Depends(get_job_manager),
) -> JobResponse:
    job = job_manager.get_job(job_id=job_id, tenant_id=auth_context.tenant_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobResponse.from_record(job)


@router.post("/{job_id}/cancel", response_model=JobCancelResponse)
def cancel_job(
    job_id: str,
    auth_context: AuthContext = Depends(get_auth_context),
    job_manager: JobManager = Depends(get_job_manager),
) -> JobCancelResponse:
    job = job_manager.cancel_job(job_id=job_id, tenant_id=auth_context.tenant_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    if job.status == JobStatus.CANCELLED:
        message = "Job cancelled successfully."
    elif job.status in {JobStatus.PROCESSING, JobStatus.COMPLETED, JobStatus.FAILED}:
        message = f"Job cannot be cancelled in status: {job.status.value}"
    else:
        message = "Cancellation request accepted."

    return JobCancelResponse(
        id=job.id,
        status=job.status,
        message=message,
    )