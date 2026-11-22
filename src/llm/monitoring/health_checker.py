# src/llm/monitoring/health_checker.py
"""
Health checking for LLM providers.
"""
import logging
from typing import Dict
from src.llm.base_provider import BaseLLMProvider

logger = logging.getLogger(__name__)

class HealthChecker:
    """
    Monitors health status of LLM providers.
    """
    _health_status: Dict[str, bool] = {}

    @classmethod
    def check_provider(cls, provider: BaseLLMProvider) -> bool:
        """Check if a provider is healthy."""
        provider_name = provider.get_name()
        try:
            is_healthy = provider.health_check()
            cls._health_status[provider_name] = is_healthy
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed for {provider_name}: {e}")
            cls._health_status[provider_name] = False
            return False

    @classmethod
    def get_health_status(cls, provider_name: str) -> bool:
        """Get cached health status for a provider."""
        return cls._health_status.get(provider_name, False)

    @classmethod
    def get_all_status(cls) -> Dict[str, bool]:
        """Get health status for all providers."""
        return cls._health_status.copy()