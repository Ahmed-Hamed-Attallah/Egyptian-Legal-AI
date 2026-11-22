import pytest
from unittest.mock import MagicMock, patch
from src.llm.fallback_manager import FallbackManager
from src.llm.error_classifier import LLMErrorCategory, ClassifiedLLMError
from src.domain.responses import GenerationRequest, GenerationResponse
from src.domain.exceptions import LLMError
from src.config.settings import settings

# Mock Settings
settings.LLM_MAX_RETRIES = 1
settings.LLM_RETRY_BACKOFF_SECONDS = 0.01
settings.LLM_PROVIDER_COOLDOWN_SECONDS = 1
settings.LLM_QUOTA_COOLDOWN_SECONDS = 2
settings.LLM_CIRCUIT_BREAKER_THRESHOLD = 2

def make_mock_provider(name, model="test-model"):
    mock = MagicMock()
    mock.get_name.return_value = name
    mock.get_provider_info.return_value = {"model": model}
    return mock

def make_req():
    return GenerationRequest(system_prompt="test", user_prompt="test")

def test_scenario_1_groq_missing_openai_available():
    # Groq missing, OpenAI available
    providers = {"openai": make_mock_provider("openai")}
    fm = FallbackManager(providers, ["groq", "openai"])
    fm.providers["openai"].generate.return_value = GenerationResponse(content="OK", provider="openai", model="test", tokens_used=10, latency_ms=50)
    
    resp = fm.generate(make_req())
    assert resp.provider == "openai"

def test_scenario_5_rate_limit_retry_then_switch():
    # Groq rate limited, OpenAI available
    groq = make_mock_provider("groq")
    openai = make_mock_provider("openai")
    
    # Groq fails with 429 Rate Limit
    err = Exception("Rate limit exceeded")
    err.status_code = 429
    groq.generate.side_effect = err
    
    openai.generate.return_value = GenerationResponse(content="OK", provider="openai", model="test", tokens_used=10, latency_ms=50)
    
    fm = FallbackManager({"groq": groq, "openai": openai}, ["groq", "openai"])
    resp = fm.generate(make_req())
    
    assert resp.provider == "openai"
    assert fm.circuit_breakers["groq"].state.value == "open" # Tripped after max retries

def test_scenario_6_quota_exhausted_immediate_switch():
    # Groq quota exhausted, OpenAI available
    groq = make_mock_provider("groq")
    openai = make_mock_provider("openai")
    
    err = Exception("Quota exceeded for today")
    err.status_code = 429
    groq.generate.side_effect = err
    
    openai.generate.return_value = GenerationResponse(content="OK", provider="openai", model="test", tokens_used=10, latency_ms=50)
    
    fm = FallbackManager({"groq": groq, "openai": openai}, ["groq", "openai"])
    resp = fm.generate(make_req())
    
    assert resp.provider == "openai"
    # Should only call Groq ONCE because quota exhaustion breaks the retry loop immediately
    assert groq.generate.call_count == 1 

def test_scenario_7_invalid_api_key_disables_provider():
    # OpenAI 401 Invalid Key
    openai = make_mock_provider("openai")
    gemini = make_mock_provider("gemini")
    
    err = Exception("Invalid API Key")
    err.status_code = 401
    openai.generate.side_effect = err
    
    gemini.generate.return_value = GenerationResponse(content="OK", provider="gemini", model="test", tokens_used=10, latency_ms=50)
    
    fm = FallbackManager({"openai": openai, "gemini": gemini}, ["openai", "gemini"])
    resp = fm.generate(make_req())
    
    assert resp.provider == "gemini"
    assert fm.circuit_breakers["openai"].state.value == "open"

def test_scenario_12_all_unavailable():
    # All fail
    groq = make_mock_provider("groq")
    groq.generate.side_effect = Exception("Server Error")
    
    fm = FallbackManager({"groq": groq}, ["groq"])
    
    with pytest.raises(LLMError):
        fm.generate(make_req())

def test_scenario_18_quota_cooldown_prevents_repeated_calls():
    groq = make_mock_provider("groq")
    err = Exception("Quota exceeded")
    err.status_code = 429
    groq.generate.side_effect = err
    
    fm = FallbackManager({"groq": groq}, ["groq"])
    
    # First call trips the breaker
    with pytest.raises(LLMError):
        fm.generate(make_req())
        
    # Second call should NOT invoke groq.generate again due to cooldown
    with pytest.raises(LLMError):
        fm.generate(make_req())
        
    assert groq.generate.call_count == 1