"""JSON-safe serialization primitives.

PostgreSQL JSON/JSONB columns and Qdrant payloads are encoded with the standard
`json` module, which rejects UUID, datetime, date, Decimal, Enum, bytes and set
values. Domain metadata legitimately contains all of these (document IDs, legal
effective dates, relevance weights), so every persistence boundary must use this
encoder instead of bare `json.dumps`.

`json_dumps` is wired into the SQLAlchemy engine via `json_serializer`, which the
psycopg dialect forwards to `psycopg.types.json.set_json_dumps` — covering both
`json` and `jsonb` columns for all models.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

__all__ = ["json_default", "json_dumps", "to_jsonable"]


def json_default(value: Any) -> Any:
    """Convert non-JSON-native domain values into JSON-native equivalents.

    Used as the `default=` hook, so it applies at any nesting depth.
    """
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Enum):
        return json_default(value.value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")

    # Pydantic v2 models
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable. "
        "Add a conversion in src/utils/json_utils.json_default()."
    )


def json_dumps(value: Any, **kwargs: Any) -> str:
    """Canonical encoder for persistence boundaries.

    `ensure_ascii=False` keeps Arabic text literal rather than \\uXXXX escapes
    (Arabic-first system, ARCHITECTURE.md §2). JSONB stores decoded text either
    way, so no migration is required.
    """
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("default", json_default)
    return json.dumps(value, **kwargs)


def to_jsonable(value: Any) -> Any:
    """Return a plain JSON-native structure (dicts/lists/str/int/float/bool/None).

    Use where an API client needs a real object rather than a string — e.g.
    Qdrant payload upserts or worker job results.
    """
    return json.loads(json_dumps(value))