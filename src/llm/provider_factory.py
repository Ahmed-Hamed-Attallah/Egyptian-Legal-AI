
import logging
from typing import Optional, List

from src.llm.base_provider import BaseLLMProvider
from src.llm.providers import (
    GroqProvider,
    OpenAIProvider,
    GeminiProvider,
    TransformersProvider,
)
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ProviderFactory:

    PROVIDER_ORDER = [
        "groq",
        "openai",
        "gemini",
        "transformers",
    ]

    @staticmethod
    def create_provider(
        provider_name: str,
    ) -> Optional[BaseLLMProvider]:
        name = (provider_name or "").strip().lower()

        # Pre-check configuration to avoid noisy tracebacks for optional providers
        if name == "groq" and not getattr(settings, "GROQ_API_KEY", None):
            logger.info("[LLM] %s skipped: Missing API key", name)
            return None
        if name == "openai" and not getattr(settings, "OPENAI_API_KEY", None):
            logger.info("[LLM] %s skipped: Missing API key", name)
            return None
        if name == "gemini" and not getattr(settings, "GEMINI_API_KEY", None):
            logger.info("[LLM] %s skipped: Missing API key", name)
            return None
        if name == "transformers" and not getattr(settings, "LOCAL_MODEL_NAME", None):
            logger.info("[LLM] %s skipped: Missing local model name", name)
            return None

        try:
            if name == "groq":
                key = getattr(settings, "GROQ_API_KEY", None)
                model = getattr(settings, "GROQ_MODEL", None)
                if not model:
                    logger.warning("[LLM] %s skipped: Missing model configuration", name)
                    return None
                provider = GroqProvider(api_key=key, model=model)

            elif name == "openai":
                key = getattr(settings, "OPENAI_API_KEY", None)
                model = getattr(settings, "OPENAI_MODEL", None)
                if not model:
                    logger.warning("[LLM] %s skipped: Missing model configuration", name)
                    return None
                provider = OpenAIProvider(api_key=key, model=model)

            elif name == "gemini":
                key = getattr(settings, "GEMINI_API_KEY", None)
                model = getattr(settings, "GEMINI_MODEL", None)
                if GeminiProvider is None:
                    logger.warning("[LLM] %s skipped: Dependency unavailable", name)
                    return None
                if not model:
                    logger.warning("[LLM] %s skipped: Missing model configuration", name)
                    return None
                provider = GeminiProvider(api_key=key, model=model)

            elif name == "transformers":
                model = getattr(settings, "LOCAL_MODEL_NAME", None)
                provider = TransformersProvider(model_name=model)

            else:
                logger.warning("[LLM] Unknown provider requested: %s", name)
                return None

            if provider is None:
                raise RuntimeError("Provider returned None")

            actual_name = provider.get_name().strip().lower()
            if actual_name != name:
                raise RuntimeError(
                    f"Provider identity mismatch: requested {name}, "
                    f"received {actual_name}"
                )

            logger.info("[LLM] %s initialized successfully", name)
            return provider

        except Exception:
            # Only log full tracebacks for unexpected runtime errors
            logger.exception("[LLM] Failed to initialize provider: %s", name)
            return None

    @classmethod
    def create_all(
        cls,
        priority_order: Optional[List[str]] = None,
    ) -> List[BaseLLMProvider]:
        """
        Independently initialize configured providers.
        """

        order = priority_order or cls.PROVIDER_ORDER
        providers = []

        for name in order:
            provider = cls.create_provider(name)

            if provider is not None:
                providers.append(provider)

        logger.info(
            "[LLM] Successfully initialized %s/%s providers",
            len(providers),
            len(order),
        )

        return providers

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Configuration-only availability check.
        """

        available = []

        if getattr(settings, "GROQ_API_KEY", None):
            available.append("groq")

        if getattr(settings, "OPENAI_API_KEY", None):
            available.append("openai")

        if (
            getattr(settings, "GEMINI_API_KEY", None)
            and GeminiProvider is not None
        ):
            available.append("gemini")

        if getattr(settings, "LOCAL_MODEL_NAME", None):
            available.append("transformers")

        return available
