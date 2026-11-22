# api/routes/health.py
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_admin_service
from api.schemas.admin import HealthResponse
from src.services.admin_service import AdminService

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def live() -> dict:
    return {"status": "alive"}


@router.get("/ready", response_model=HealthResponse)
def ready(admin_service: AdminService = Depends(get_admin_service)) -> HealthResponse:
    return HealthResponse.model_validate(admin_service.health())