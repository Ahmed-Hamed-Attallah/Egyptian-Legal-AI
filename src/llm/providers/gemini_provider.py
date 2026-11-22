"""
Google Gemini API provider implementation.
"""
import logging
import time
from typing import Any, Dict, List, Tuple

from src.llm.base_provider import BaseLLMProvider
from src.domain.responses import GenerationRequest, GenerationResponse
from src.domain.exceptions import ProviderError, ProviderUnavailableError
from src.config.settings import settings

logger = logging.getLogger(__name__)

# Lazy import guard for optional dependency
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False
    logger.debug("google-generativeai not installed. GeminiProvider will be unavailable.")


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API integration.

    Supports conversation memory via ``request.get_messages()``.
    Gemini's chat API uses roles "user" and "model" and has no dedicated
    system role, so the system prompt is prepended to the first user turn.
    """

    def __init__(self, api_key: str = None, model: str = None, timeout: int = None):
        if not GEMINI_AVAILABLE:
            raise ProviderUnavailableError(
                "Google Generative AI package is not installed. "
                "Install with: pip install google-generativeai",
                provider="gemini",
            )

        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model or "gemini-2.0-flash-exp"
        self.timeout = timeout or settings.LLM_REQUEST_TIMEOUT_SECONDS

        if not self.api_key:
            raise ProviderError("Gemini API key not configured", provider="gemini")

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            raise ProviderError(f"Failed to initialize Gemini client: {e}", provider="gemini")

    @staticmethod
    def _prepare_gemini_chat(
        messages: List[Dict[str, str]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Convert OpenAI-style messages into Gemini chat history + final turn.

        - system messages  -> prepended to the first user turn (no system role)
        - assistant        -> "model"
        - the LAST user message is returned separately for ``send_message``.
        """
        system_parts: List[str] = []
        turns: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                turns.append({"role": "user", "parts": [content]})
            elif role in ("assistant", "model"):
                turns.append({"role": "model", "parts": [content]})

        if not turns or turns[-1]["role"] != "user":
            raise ProviderError(
                "The last message in the conversation must be from the user.",
                provider="gemini",
            )

        last_user_turn = turns.pop()

        # Gemini has no system role: attach the system prompt to the first
        # user turn so it still frames the whole conversation.
        system_text = "\n\n".join(p for p in system_parts if p)
        if system_text:
            if turns:
                turns[0]["parts"][0] = f"{system_text}\n\n{turns[0]['parts'][0]}"
            else:
                last_user_turn["parts"][0] = f"{system_text}\n\n{last_user_turn['parts'][0]}"

        return turns, last_user_turn

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        start_time = time.time()
        try:
            # Full conversation: system (+ memory digest) + history + user turn
            messages = request.get_messages()
            history, last_user_turn = self._prepare_gemini_chat(messages)

            chat = self.model.start_chat(history=history)
            response = chat.send_message(
                last_user_turn["parts"][0],
                generation_config=genai.types.GenerationConfig(
                    temperature=request.temperature or settings.LLM_TEMPERATURE,
                    max_output_tokens=request.max_tokens or settings.LLM_MAX_TOKENS,
                ),
            )

            latency_ms = (time.time() - start_time) * 1000
            content = response.text
            tokens = (
                response.usage_metadata.total_token_count
                if getattr(response, "usage_metadata", None)
                else 0
            )

            return GenerationResponse(
                content=content,
                provider="gemini",
                model=self.model_name,
                tokens_used=tokens,
                latency_ms=latency_ms,
                metadata={"message_count": len(messages)},
            )
        except ProviderError:
            raise
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise ProviderError(f"Generation failed: {e}", provider="gemini")

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider": "gemini",
            "model": self.model_name,
            "supports_streaming": True,
            "max_tokens": 8192,
        }

    def health_check(self) -> bool:
        try:
            return self.model is not None
        except Exception:
            return False

    def get_name(self) -> str:
        return "gemini"