import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        started = time.perf_counter()

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - started) * 1000

            logger.info(
                "method=%s path=%s status=%s duration_ms=%.2f",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )

            response.headers["X-Process-Time-Ms"] = (
                f"{duration_ms:.2f}"
            )
            return response

        except Exception:
            logger.exception(
                "Request failed: %s %s",
                request.method,
                request.url.path,
            )
            raise