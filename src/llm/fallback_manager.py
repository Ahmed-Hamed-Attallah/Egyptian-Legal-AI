
import logging
import time

from typing import Dict, List, Optional

from src.llm.base_provider import BaseLLMProvider
from src.llm.error_classifier import (
    classify_exception,
    LLMErrorCategory,
    ClassifiedLLMError,
)
from src.llm.monitoring.circuit_breaker import CircuitBreaker
from src.llm.monitoring.usage_tracker import UsageTracker
from src.domain.responses import GenerationRequest, GenerationResponse
from src.domain.exceptions import LLMError
from src.config.settings import settings

logger = logging.getLogger(__name__)


class FallbackManager:

    def __init__(
        self,
        providers: Dict[str, BaseLLMProvider],
        priority_order: List[str],
    ):

        self.providers = providers
        self.priority_order = priority_order

        self.circuit_breakers = {
            name: CircuitBreaker(name)
            for name in providers
        }

        self.usage_tracker = UsageTracker()

    def _get_model_name(self, provider):
        try:
            return provider.get_provider_info().get(
                "model",
                "unknown",
            )
        except Exception:
            return "unknown"

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:

        last_errors = []
        is_fallback = False

        for provider_name in self.priority_order:

            provider = self.providers.get(provider_name)

            if provider is None:
                continue

            breaker = self.circuit_breakers[provider_name]

            if not breaker.allow_request():
                logger.warning(
                    "[LLM] %s skipped: circuit breaker is open",
                    provider_name,
                )
                is_fallback = True
                continue

            model_name = self._get_model_name(provider)

            retries = max(
                0,
                int(getattr(settings, "LLM_MAX_RETRIES", 1)),
            )

            for attempt in range(retries + 1):

                started = time.time()

                try:

                    logger.info(
                        "[LLM] Trying %s (%s), attempt %s",
                        provider_name,
                        model_name,
                        attempt + 1,
                    )

                    response = provider.generate(request)

                    if response is None:
                        raise RuntimeError(
                            "Provider returned None"
                        )

                    if not isinstance(response, GenerationResponse):
                        raise TypeError(
                            "Invalid GenerationResponse type"
                        )

                    if not response.content or not response.content.strip():
                        raise RuntimeError(
                            "Provider returned an empty response"
                        )

                    breaker.record_success()

                    latency = (time.time() - started) * 1000

                    self.usage_tracker.record_request(
                        provider=provider_name,
                        model=model_name,
                        success=True,
                        tokens=response.tokens_used or 0,
                        latency_ms=latency,
                        is_fallback=is_fallback,
                    )

                    logger.info(
                        "[LLM] SUCCESS: %s using %s",
                        provider_name,
                        model_name,
                    )

                    return response

                except Exception as exc:

                    classified = classify_exception(
                        exc,
                        provider_name,
                    )

                    logger.warning(
                        "[LLM DEBUG] Provider=%s | Category=%s | Error=%s",
                        provider_name,
                        classified.category.value,
                        exc,
                    )

                    last_errors.append(
                        f"{provider_name}: {classified}"
                    )

                    logger.exception(
                        "[LLM] %s failed: %s",
                        provider_name,
                        classified.category.value,
                    )

                    breaker.record_failure(
                        classified.category.value,
                        classified.retry_after,
                    )

                    self.usage_tracker.record_request(
                        provider=provider_name,
                        model=model_name,
                        success=False,
                        error_category=classified.category.value,
                        is_fallback=is_fallback,
                    )

                    # Do not waste retries on errors that will not
                    # be fixed by retrying the same provider.
                    fatal = {
                        LLMErrorCategory.MISSING_API_KEY,
                        LLMErrorCategory.INVALID_API_KEY,
                        LLMErrorCategory.QUOTA_EXHAUSTED,
                        LLMErrorCategory.BILLING_LIMIT_REACHED,
                        LLMErrorCategory.INVALID_REQUEST,
                        LLMErrorCategory.CONTEXT_LENGTH_EXCEEDED,
                        LLMErrorCategory.MODEL_UNAVAILABLE,
                        LLMErrorCategory.DEPENDENCY_MISSING,
                    }

                    if classified.category in fatal:
                        break

                    if attempt < retries:

                        backoff = min(
                            getattr(
                                settings,
                                "LLM_RETRY_BACKOFF_SECONDS",
                                1,
                            ) * (2 ** attempt),
                            getattr(
                                settings,
                                "LLM_RETRY_MAX_BACKOFF_SECONDS",
                                5,
                            ),
                        )

                        time.sleep(backoff)

            # Crucial: move to the next provider.
            is_fallback = True

        logger.error(
            "[LLM] ALL PROVIDERS FAILED: %s",
            last_errors,
        )

        raise LLMError(
            "All configured LLM providers failed. "
            "No reliable response could be generated."
        )

    def get_provider_statuses(self):

        statuses = {}

        for name, provider in self.providers.items():

            try:
                model = self._get_model_name(provider)

                statuses[name] = {
                    "provider": name,
                    "model": model,
                    "initialized": True,
                    "circuit_breaker": (
                        self.circuit_breakers[name].get_state()
                    ),
                    "usage": self.usage_tracker.get_stats(
                        name,
                        model,
                    ),
                }

            except Exception as exc:

                statuses[name] = {
                    "provider": name,
                    "initialized": False,
                    "error": str(exc),
                }

        return statuses
