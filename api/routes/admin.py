# api/routes/admin.py
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, HTTPException

from api.dependencies import AuthContext, require_admin
from api.schemas.admin import AdminStatusResponse, HealthResponse
from api.dependencies import get_admin_service
from src.services.admin_service import AdminService
from src.monitoring.metrics_collector import get_monitor

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


@router.get("/status", response_model=AdminStatusResponse)
def status(
    admin_service: AdminService = Depends(get_admin_service),
) -> AdminStatusResponse:
    return AdminStatusResponse.model_validate(admin_service.status())


@router.get("/health", response_model=HealthResponse)
def health(
    admin_service: AdminService = Depends(get_admin_service),
) -> HealthResponse:
    return HealthResponse.model_validate(admin_service.health())


@router.get("/metrics")
def metrics() -> dict:
    monitor = get_monitor()
    snapshot = monitor.snapshot()
    return {
        "enabled": monitor.enabled,
        "generated_at": snapshot.generated_at,
        "counters": snapshot.counters,
        "histograms": snapshot.histograms,
        "recent_events": snapshot.events[-200:],
    }

@router.get("/providers")
def get_providers(
    request: Request,
    admin_service: AdminService = Depends(get_admin_service),
) -> dict:
    router = getattr(request.app.state, "provider_router", None)
    if not router:
        return {"providers": [], "message": "Provider router not initialized"}
    return {"providers": router.get_provider_statuses()}

@router.post("/providers/{provider_name}/reset")
def reset_provider(
    provider_name: str,
    request: Request,
    admin_service: AdminService = Depends(get_admin_service),
) -> dict:
    router = getattr(request.app.state, "provider_router", None)
    if not router or provider_name not in router.fallback_manager.circuit_breakers:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    router.fallback_manager.circuit_breakers[provider_name].reset()
    return {"status": "success", "message": f"Circuit breaker for {provider_name} reset."}