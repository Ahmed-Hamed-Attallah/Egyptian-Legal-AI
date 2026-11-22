import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

class LLMErrorCategory(str, Enum):
    MISSING_API_KEY = "MISSING_API_KEY"
    INVALID_API_KEY = "INVALID_API_KEY"
    RATE_LIMITED = "RATE_LIMITED"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    BILLING_LIMIT_REACHED = "BILLING_LIMIT_REACHED"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CONTEXT_LENGTH_EXCEEDED = "CONTEXT_LENGTH_EXCEEDED"
    INVALID_REQUEST = "INVALID_REQUEST"
    DEPENDENCY_MISSING = "DEPENDENCY_MISSING"
    UNKNOWN_PROVIDER_ERROR = "UNKNOWN_PROVIDER_ERROR"

class ClassifiedLLMError(Exception):
    def __init__(self, category: LLMErrorCategory, message: str, provider: str, 
                 retry_after: Optional[int] = None, original_exc: Optional[Exception] = None):
        self.category = category
        self.provider = provider
        self.retry_after = retry_after
        self.original_exc = original_exc
        super().__init__(f"[{provider}] {category.value}: {message}")



def classify_exception(exc: Exception, provider: str) -> ClassifiedLLMError:
    retry_after = None

    # Inspect wrapped exceptions as well as the outer exception.
    exceptions = []
    current = exc
    seen = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        exceptions.append(current)

        current = (
            getattr(current, "original_exc", None)
            or getattr(current, "__cause__", None)
            or getattr(current, "__context__", None)
        )

    msg = " ".join(str(item).lower() for item in exceptions)

    status_code = None

    for item in exceptions:
        status_code = getattr(item, "status_code", None)

        if status_code is None:
            response = getattr(item, "response", None)
            status_code = getattr(response, "status_code", None)

        if status_code is not None:
            break

    # Groq and other SDKs sometimes expose model errors in the body/message.
    if status_code is None and any(
        marker in msg
        for marker in (
            "model_not_found",
            "model does not exist",
            "model_unavailable",
        )
    ):
        status_code = 404

    if "no module named" in msg or "cannot import" in msg:
        category = LLMErrorCategory.DEPENDENCY_MISSING

    elif "missing api key" in msg or "api key not configured" in msg:
        category = LLMErrorCategory.MISSING_API_KEY

    elif status_code in (401, 403):
        if "quota" in msg or "billing" in msg:
            category = LLMErrorCategory.BILLING_LIMIT_REACHED
        else:
            category = LLMErrorCategory.INVALID_API_KEY

    elif status_code == 404:
        category = LLMErrorCategory.MODEL_UNAVAILABLE

    elif status_code == 429:
        if "quota" in msg or "billing" in msg:
            category = LLMErrorCategory.QUOTA_EXHAUSTED
        else:
            category = LLMErrorCategory.RATE_LIMITED

            for item in exceptions:
                response = getattr(item, "response", None)
                headers = getattr(response, "headers", {}) or {}
                value = headers.get("retry-after")

                if value:
                    try:
                        retry_after = int(value)
                    except (TypeError, ValueError):
                        pass
                    break

    elif status_code == 400:
        if "context length" in msg or "maximum context" in msg:
            category = LLMErrorCategory.CONTEXT_LENGTH_EXCEEDED
        else:
            category = LLMErrorCategory.INVALID_REQUEST

    elif status_code is not None and status_code >= 500:
        category = LLMErrorCategory.SERVER_ERROR

    elif "timeout" in msg or "timed out" in msg:
        category = LLMErrorCategory.REQUEST_TIMEOUT

    elif "connection" in msg or "network" in msg:
        category = LLMErrorCategory.CONNECTION_ERROR

    else:
        category = LLMErrorCategory.UNKNOWN_PROVIDER_ERROR

    return ClassifiedLLMError(
        category=category,
        message=str(exc),
        provider=provider,
        retry_after=retry_after,
        original_exc=exc,
    )

