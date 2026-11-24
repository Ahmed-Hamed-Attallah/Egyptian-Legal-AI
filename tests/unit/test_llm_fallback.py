"""Fallback, retry, cooldown and circuit-breaker behaviour.

Settings are patched per-test via a fixture. The previous version mutated the
process-wide singleton at import time, which leaked into every later test and
made results depend on collection order.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.domain.exceptions import LLMError
from src.domain.responses import GenerationRequest, GenerationResponse
from src.llm.fallback_manager import FallbackManager


@pytest.fixture(autouse=True)
def fast_retry_policy(monkeypatch: pytest.MonkeyPatch, app_settings):
    monkeypatch.setattr(app_settings, "LLM_MAX_RETRIES", 1, raising=False)
    monkeypatch.setattr(app_settings, "LLM_RETRY_BACKOFF_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(app_settings, "LLM_PROVIDER_COOLDOWN_SECONDS", 1, raising=False)
    monkeypatch.setattr(app_settings, "LLM_QUOTA_COOLDOWN_SECONDS", 2, raising=False)
    monkeypatch.setattr(app_settings, "LLM_CIRCUIT_BREAKER_THRESHOLD", 2, raising=False)
    return app_settings


def provider(name: str, model: str = "test-model") -> MagicMock:
    mock = MagicMock()
    mock.get_name.return_value = name
    mock.get_provider_info.return_value = {"model": model}
    return mock


def ok(name: str) -> GenerationResponse:
    return GenerationResponse(
        content="OK", provider=name, model="test", tokens_used=10, latency_ms=50
    )


def request() -> GenerationRequest:
    return GenerationRequest(system_prompt="s", user_prompt="u")


def http_error(message: str, status: int) -> Exception:
    exc = Exception(message)
    exc.status_code = status  # type: ignore[attr-defined]
    return exc


def test_falls_through_to_next_available_provider():
    openai = provider("openai")
    openai.generate.return_value = ok("openai")
    manager = FallbackManager({"openai": openai}, ["groq", "openai"])
    assert manager.generate(request()).provider == "openai"


def test_rate_limit_retries_then_switches_and_trips_breaker():
    groq, openai = provider("groq"), provider("openai")
    groq.generate.side_effect = http_error("Rate limit exceeded", 429)
    openai.generate.return_value = ok("openai")

    manager = FallbackManager({"groq": groq, "openai": openai}, ["groq", "openai"])
    assert manager.generate(request()).provider == "openai"
    assert manager.circuit_breakers["groq"].state.value == "open"


def test_quota_exhaustion_switches_without_retrying():
    groq, openai = provider("groq"), provider("openai")
    groq.generate.side_effect = http_error("Quota exceeded for today", 429)
    openai.generate.return_value = ok("openai")

    manager = FallbackManager({"groq": groq, "openai": openai}, ["groq", "openai"])
    assert manager.generate(request()).provider == "openai"
    assert groq.generate.call_count == 1, "quota exhaustion must not be retried"


def test_invalid_key_disables_provider():
    openai, gemini = provider("openai"), provider("gemini")
    openai.generate.side_effect = http_error("Invalid API Key", 401)
    gemini.generate.return_value = ok("gemini")

    manager = FallbackManager({"openai": openai, "gemini": gemini}, ["openai", "gemini"])
    assert manager.generate(request()).provider == "gemini"
    assert manager.circuit_breakers["openai"].state.value == "open"


def test_all_providers_failing_raises_llm_error():
    groq = provider("groq")
    groq.generate.side_effect = Exception("Server Error")
    manager = FallbackManager({"groq": groq}, ["groq"])
    with pytest.raises(LLMError):
        manager.generate(request())


def test_cooldown_prevents_repeated_calls_to_a_tripped_provider():
    groq = provider("groq")
    groq.generate.side_effect = http_error("Quota exceeded", 429)
    manager = FallbackManager({"groq": groq}, ["groq"])

    with pytest.raises(LLMError):
        manager.generate(request())
    with pytest.raises(LLMError):
        manager.generate(request())

    assert groq.generate.call_count == 1