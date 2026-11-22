"""
OpenAI API provider implementation.
"""
import logging
import time
from typing import Dict, Any

from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

from src.llm.base_provider import BaseLLMProvider
from src.domain.responses import GenerationRequest, GenerationResponse
from src.domain.exceptions import ProviderError, ProviderUnavailableError
from src.config.settings import settings

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI API integration.

    Supports conversation memory via ``request.get_messages()``.
    """

    def __init__(self, api_key: str = None, model: str = None, timeout: int = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or "gpt-4o-mini"
        self.timeout = timeout or settings.LLM_REQUEST_TIMEOUT_SECONDS

        if not self.api_key:
            raise ProviderError("OpenAI API key not configured", provider="openai")

        try:
            self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        except Exception as e:
            raise ProviderError(f"Failed to initialize OpenAI client: {e}", provider="openai")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()
        try:
            # Full conversation: system (+ memory digest) + history + user turn.
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
                provider="openai",
                model=self.model,
                tokens_used=tokens,
                latency_ms=latency_ms,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "message_count": len(messages),
                },
            )
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise ProviderError(f"Rate limit exceeded: {e}", provider="openai")
        except APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            raise ProviderUnavailableError(f"Connection failed: {e}", provider="openai")
        except APIStatusError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ProviderError(f"API error: {e}", provider="openai")
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise ProviderError(f"Generation failed: {e}", provider="openai")

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.model,
            "supports_streaming": True,
            "max_tokens": 16384,
        }

    def health_check(self) -> bool:
        try:
            return self.client is not None
        except Exception:
            return False

    def get_name(self) -> str:
        return "openai"