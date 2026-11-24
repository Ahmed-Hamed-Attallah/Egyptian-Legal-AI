# src/retrieval/metadata_filters.py
"""Backend-agnostic metadata filtering.

A filter is a boolean tree: field conditions combined with and/or, plus nested
groups, so access policies like

    tenant_id == T AND (scope == "official" OR (scope == "private" AND owner_id == me))

are expressed once and evaluated identically by dense retrieval, sparse
retrieval and SQL.

Evaluation lives HERE. The previous version declared `operator` but never
honoured it, made `or` behave as `and`, and silently skipped every comparison
operator — unacceptable for an access-control filter.

Payload convention: `scope` is the lowercase enum VALUE ("official"/"private").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence
from uuid import UUID

from src.domain.queries import QueryScope

SUPPORTED_CONDITION_OPERATORS = frozenset(
    {"eq", "neq", "gt", "gte", "lt", "lte", "in", "nin", "contains", "exists"}
)
SUPPORTED_GROUP_OPERATORS = frozenset({"and", "or"})

SCOPE_OFFICIAL = "official"
SCOPE_PRIVATE = "private"

_MISSING = object()


def _values_equal(actual: Any, expected: Any) -> bool:
    """Equality tolerant of UUID/str mixing (domain UUID vs payload string)."""
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(actual, UUID) or isinstance(expected, UUID):
        return str(actual) == str(expected)
    return bool(actual == expected)


def _as_sequence(value: Any) -> Sequence[Any]:
    return list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]


def _comparable(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


@dataclass
class FilterCondition:
    """A single comparison against a metadata field."""

    field: str
    value: Any = None
    operator: str = "eq"

    def __post_init__(self) -> None:
        op = (self.operator or "eq").strip().lower()
        if op not in SUPPORTED_CONDITION_OPERATORS:
            raise ValueError(
                f"Unsupported filter operator {self.operator!r}. "
                f"Supported: {sorted(SUPPORTED_CONDITION_OPERATORS)}"
            )
        self.operator = op

    def evaluate(self, metadata: Mapping[str, Any]) -> bool:
        """Fail closed: an absent field never satisfies a positive condition."""
        actual = metadata.get(self.field, _MISSING)
        present = actual is not _MISSING and actual is not None
        op = self.operator

        if op == "exists":
            return present
        if op == "eq":
            return present and _values_equal(actual, self.value)
        if op == "neq":
            return not (present and _values_equal(actual, self.value))
        if op in {"in", "nin"}:
            hit = present and any(_values_equal(actual, v) for v in _as_sequence(self.value))
            return hit if op == "in" else not hit
        if op == "contains":
            if not present:
                return False
            if isinstance(actual, (list, tuple, set, frozenset)):
                return any(_values_equal(i, self.value) for i in actual)
            if isinstance(actual, str):
                return str(self.value) in actual
            return False

        if not present:
            return False
        left, right = _comparable(actual), _comparable(self.value)
        if left is None or right is None:
            return False
        if op == "gt":
            return left > right
        if op == "gte":
            return left >= right
        if op == "lt":
            return left < right
        return left <= right


@dataclass
class MetadataFilter:
    """Conditions plus nested groups, combined with one logical operator."""

    conditions: list[FilterCondition] = field(default_factory=list)
    operator: str = "and"
    groups: list["MetadataFilter"] = field(default_factory=list)

    def __post_init__(self) -> None:
        op = (self.operator or "and").strip().lower()
        if op not in SUPPORTED_GROUP_OPERATORS:
            raise ValueError("MetadataFilter.operator must be 'and' or 'or'.")
        self.operator = op

    # -- construction --------------------------------------------------------
    def add_condition(
        self, field_name: str, value: Any = None, operator: str = "eq"
    ) -> "MetadataFilter":
        """Fluent append. Returns self so calls can chain."""
        self.conditions.append(FilterCondition(field_name, value, operator))
        return self

    def add_group(self, group: "MetadataFilter") -> "MetadataFilter":
        self.groups.append(group)
        return self

    @classmethod
    def all_of(cls, *parts: "FilterCondition | MetadataFilter") -> "MetadataFilter":
        return cls._compose("and", parts)

    @classmethod
    def any_of(cls, *parts: "FilterCondition | MetadataFilter") -> "MetadataFilter":
        return cls._compose("or", parts)

    @classmethod
    def _compose(cls, operator: str, parts: Iterable[Any]) -> "MetadataFilter":
        instance = cls(operator=operator)
        for part in parts:
            if isinstance(part, FilterCondition):
                instance.conditions.append(part)
            elif isinstance(part, MetadataFilter):
                instance.groups.append(part)
            else:
                raise TypeError(f"Unsupported filter part: {type(part).__name__}")
        return instance

    # -- access policy (single source of truth) ------------------------------
    @classmethod
    def for_documents(
        cls,
        *,
        tenant_id: str,
        user_id: Optional[str],
        scope: QueryScope = QueryScope.ALL_SOURCES,
        is_admin: bool = False,
    ) -> "MetadataFilter":
        """Visibility policy shared by dense, sparse and SQL.

        official             → every authenticated member of the tenant
        private + owner=me   → me only
        private + owner=None → admins only (fail closed)

        ALL_SOURCES means "official plus my own", NEVER "everyone's uploads".
        """
        tenant = FilterCondition("tenant_id", tenant_id, "eq")
        if is_admin:
            return cls.all_of(tenant)

        official = FilterCondition("scope", SCOPE_OFFICIAL, "eq")
        mine = cls.all_of(
            FilterCondition("scope", SCOPE_PRIVATE, "eq"),
            FilterCondition("owner_id", user_id, "eq"),
        )

        if isinstance(scope, str):
            scope = QueryScope(scope)

        if scope == QueryScope.OFFICIAL_ONLY:
            visible = cls.all_of(official)
        elif scope == QueryScope.USER_UPLOADS_ONLY:
            visible = mine
        else:
            visible = cls.any_of(official, mine)

        return cls.all_of(tenant, visible)

    # -- evaluation ----------------------------------------------------------
    def is_empty(self) -> bool:
        return not self.conditions and not self.groups

    def matches(self, metadata: Optional[Mapping[str, Any]]) -> bool:
        if self.is_empty():
            return True
        data: Mapping[str, Any] = metadata or {}
        results = [c.evaluate(data) for c in self.conditions]
        results.extend(g.matches(data) for g in self.groups)
        if not results:
            return True
        return all(results) if self.operator == "and" else any(results)

    def describe(self) -> str:
        parts = [f"{c.field} {c.operator} {c.value!r}" for c in self.conditions]
        parts.extend(f"({g.describe()})" for g in self.groups)
        joiner = " AND " if self.operator == "and" else " OR "
        return joiner.join(parts) if parts else "*"