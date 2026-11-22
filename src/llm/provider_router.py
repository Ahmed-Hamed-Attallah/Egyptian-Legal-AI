
import logging
from typing import Dict, List

from src.llm.base_provider import BaseLLMProvider
from src.llm.provider_factory import ProviderFactory
from src.llm.fallback_manager import FallbackManager
from src.domain.responses import GenerationRequest, GenerationResponse
from src.domain.exceptions import LLMError
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ProviderRouter:

    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.priority_order: List[str] = []

        self._discover_and_initialize()

        self.fallback_manager = FallbackManager(
            providers=self.providers,
            priority_order=self.priority_order,
        )

    def _discover_and_initialize(self):

        primary = (
            getattr(settings, "LLM_PRIMARY_PROVIDER", "auto")
            or "auto"
        ).strip().lower()

        fallback_value = getattr(
            settings,
            "LLM_FALLBACK_ORDER",
            "openai,gemini,transformers",
        ) or ""

        fallbacks = [
            item.strip().lower()
            for item in fallback_value.split(",")
            if item.strip()
        ]

        if primary == "auto" or primary not in ProviderFactory.PROVIDER_ORDER:
            candidates = ProviderFactory.PROVIDER_ORDER.copy()
        else:
            candidates = [primary]

        for name in fallbacks:
            if name in ProviderFactory.PROVIDER_ORDER:
                if name not in candidates:
                    candidates.append(name)

        # Always include Transformers as the final fallback.
        if "transformers" not in candidates:
            candidates.append("transformers")

        for name in candidates:

            try:
                provider = ProviderFactory.create_provider(name)

                if provider is None:
                    logger.warning(
                        "[LLM] %s: SKIPPED",
                        name,
                    )
                    continue

                actual_name = provider.get_name().lower()

                if actual_name in self.providers:
                    logger.warning(
                        "[LLM] Duplicate provider skipped: %s",
                        actual_name,
                    )
                    continue

                self.providers[actual_name] = provider
                self.priority_order.append(actual_name)

                logger.info(
                    "[LLM] %s: AVAILABLE | Model: %s",
                    actual_name,
                    provider.get_provider_info().get("model"),
                )

            except Exception:
                logger.exception(
                    "[LLM] Unexpected initialization error for %s",
                    name,
                )

        if not self.providers:
            logger.error(
                "[LLM] No providers initialized successfully."
            )

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:

        return self.fallback_manager.generate(request)

    def get_provider_statuses(self) -> Dict[str, dict]:

        return self.fallback_manager.get_provider_statuses()
