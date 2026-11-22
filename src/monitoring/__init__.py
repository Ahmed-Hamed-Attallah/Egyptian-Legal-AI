# src/monitoring/__init__.py
"""
Application monitoring primitives.

This package provides lightweight structured metrics and pipeline stage timing.
It does not replace existing logging or LLM provider monitoring. It is designed
to integrate with them optionally.
"""

from src.monitoring.metrics_collector import (
    MetricsCollector,
    get_monitor,
    reset_monitor,
)
from src.monitoring.pipeline_monitor import stage_timer

__all__ = [
    "MetricsCollector",
    "get_monitor",
    "reset_monitor",
    "stage_timer",
]