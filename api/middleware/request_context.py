"""Request context, timing, metrics and CORS-safe error conversion.

This middleware is added BEFORE CORSMiddleware in api/main.py, which makes it
the INNER layer. Converting unhandled exceptions to a JSONResponse here means
the response still passes out through CORSMiddleware and receives its
Access-Control-* headers — unlike a handler registered for `Exception`, which
Starlette installs on ServerErrorMiddleware, outside CORS.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from api.exceptions import error_payload
from src.monitoring.metrics_collector import get_monitor

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "request_id=%s method=%s path=%s status=500 duration_ms=%.2f",
                request_id, request.method, request.url.path, duration_ms,
            )
            # Returned, not raised: CORS headers are applied on the way out.
            response = JSONResponse(
                status_code=500,
                content=error_payload(
                    message="Internal server error.",
                    error_code="INTERNAL_ERROR",
                    request_id=request_id,
                ),
            )

        duration_ms = (time.perf_counter() - start) * 1000
        self._record_metrics(request, response.status_code, duration_ms)

        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id, request.method, request.url.path,
            response.status_code, duration_ms,
        )
        return response

    @staticmethod
    def _record_metrics(request: Request, status_code: int, duration_ms: float) -> None:
        """Metrics must never break or delay a response."""
        try:
            monitor = get_monitor()
            tags = {
                "method": request.method,
                "path": request.url.path,
                "status": status_code,
            }
            monitor.observe("http.request.duration_ms", duration_ms, tags=tags)
            if status_code >= 500:
                monitor.inc("http.request.server_error", tags=tags)
            elif status_code >= 400:
                monitor.inc("http.request.client_error", tags=tags)
            else:
                monitor.inc("http.request.success", tags=tags)
        except Exception as exc:
            logger.warning("Metrics recording failed: %s", exc)