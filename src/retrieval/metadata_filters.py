# src/retrieval/metadata_filters.py
"""
Backend-agnostic metadata filtering abstractions.
"""
from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class FilterCondition:
    """Represents a single filtering condition."""
    field: str
    value: Any
    operator: str = "eq"  # Supports: eq, neq, gt, gte, lt, lte

@dataclass
class MetadataFilter:
    """
    A collection of filter conditions combined with a logical operator.
    """
    conditions: List[FilterCondition] = field(default_factory=list)
    operator: str = "and"  # Supports: and, or

    def add_condition(self, field: str, value: Any, operator: str = "eq"):
        self.conditions.append(FilterCondition(field=field, value=value, operator=operator))