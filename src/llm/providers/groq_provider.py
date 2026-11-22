"""
Groq API provider implementation.
"""
import logging
import time
from typing import Dict, Any

from groq import Groq, APIConnectionError, RateLimitError, APIStatusError

from src.llm.base_provider import BaseLLMProvider
from src.domain.responses import GenerationRequest, GenerationResponse
from src.domain.exceptions import ProviderError, ProviderUnavailableError
from src.config.settings import settings

logger = logging.getLogger(__name__)


class GroqProvider(BaseLLMProvider):
    """
    Groq API integration for fast LLM inference.

    Supports conversation memory via ``request.get_messages()``, which returns
    the full multi-turn message list (system prompt + memory digest + history
    + current user turn).
    """

    def __init__(self, api_key: str = None, model: str = None, timeout: int = None):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.LLM_MODEL
        self.timeout = timeout or settings.LLM_REQUEST_TIMEOUT_SECONDS

        if not self.api_key:
            raise ProviderError("Groq API key not configured", provider="groq")

        try:
            self.client = Groq(api_key=self.api_key, timeout=self.timeout)
        except Exception as e:
            raise ProviderError(f"Failed to initialize Groq client: {e}", provider="groq")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()
        try:
            # Full conversation: system (+ memory digest) + history + user turn.
            # Falls back to [system, user] when no messages list is provided.
            messages = request.get_messages()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=request.temperature or settings.LLM_TEMPERATURE,
                max_tokens=request.max_tokens or settings.LLM_MAX_TOKENS,
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0

            return GenerationResponse(
                content=content,
                provider="groq",
                model=self.model,
                tokens_used=tokens,
                latency_ms=latency_ms,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "message_count": len(messages),
                },
            )
        except RateLimitError as e:
            logger.error(f"Groq rate limit exceeded: {e}")
            raise ProviderError(f"Rate limit exceeded: {e}", provider="groq")
        except APIConnectionError as e:
            logger.error(f"Groq connection error: {e}")
            raise ProviderUnavailableError(f"Connection failed: {e}", provider="groq")
        except APIStatusError as e:
            logger.error(f"Groq API error: {e}")
            raise ProviderError(f"API error: {e}", provider="groq")
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise ProviderError(f"Generation failed: {e}", provider="groq")

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider": "groq",
            "model": self.model,
            "supports_streaming": True,
            "max_tokens": 8192,
        }

    def health_check(self) -> bool:
        try:
            # Simple health check - just verify client is initialized
            return self.client is not None
        except Exception:
            return False

    def get_name(self) -> str:
        return "groq"