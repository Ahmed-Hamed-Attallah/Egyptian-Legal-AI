"""FastAPI application entrypoint.

Routing contract
----------------
Business routers declare their own sub-prefix (e.g. /auth, /chat) and are
mounted here under `settings.API_PREFIX` exactly once. Operational probes
(health) stay unprefixed so container/nginx healthchecks remain stable.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.exceptions import register_exception_handlers
from api.lifespan import lifespan
from api.middleware.request_context import RequestContextMiddleware
from api.routes import (
    admin,
    auth,
    chat,
    conversations,
    documents,
    health,
    jobs,
    search,
)
from src.config.settings import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description="Production-oriented Arabic legal RAG API for Egyptian law.",
    version="0.1.0",
    lifespan=lifespan,
)

# --- Middleware -------------------------------------------------------------
# Starlette applies the LAST-added middleware as the OUTERMOST layer.
# CORS must be outermost so error responses still carry CORS headers.
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=settings.CORS_ALLOW_ORIGIN_REGEX or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

register_exception_handlers(app)

# --- Routers ----------------------------------------------------------------
# Versioned business surface. Each router already owns its sub-prefix.
_API_ROUTERS: tuple[APIRouter, ...] = (
    auth.router,
    chat.router,
    search.router,
    documents.router,
    conversations.router,
    jobs.router,
    admin.router,
)

for _router in _API_ROUTERS:
    app.include_router(_router, prefix=settings.API_PREFIX)

# Operational probes: intentionally NOT versioned.
app.include_router(health.router)

logger.info(
    "Mounted %d routers under prefix '%s'; health probes unprefixed.",
    len(_API_ROUTERS),
    settings.API_PREFIX or "/",
)


@app.get("/", tags=["root"])
def root() -> dict:
    prefix = settings.API_PREFIX
    return {
        "service": settings.APP_NAME,
        "status": "running",
        "api_prefix": prefix,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health_live": "/health/live",
        "health_ready": "/health/ready",
        "register": f"{prefix}/auth/register",
        "login": f"{prefix}/auth/login",
    }