# src/evaluation/dataset.py
"""
Evaluation dataset loading and normalization.

Supported formats:

- .json
- .jsonl
- .csv
- .tsv

The loader intentionally supports partial annotations. Metrics should only be
computed when the required ground truth exists.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


@dataclass
class EvalRecord:
    id: str
    question: str
    expected_answer: Optional[str] = None
    relevant_document_ids: List[str] = field(default_factory=list)
    relevant_chunk_ids: List[str] = field(default_factory=list)
    relevant_articles: List[str] = field(default_factory=list)
    expected_citations: List[Dict[str, Any]] = field(default_factory=list)
    query_category: Optional[str] = None
    language: str = "ar"
    relevance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def _as_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in text.split(";") if part.strip()]
    return [str(value)]


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _as_relevance(value: Any) -> Dict[str, float]:
    raw = _as_dict(value)
    result: Dict[str, float] = {}
    for key, val in raw.items():
        try:
            result[str(key)] = float(val)
        except (TypeError, ValueError):
            continue
    return result


def _record_from_mapping(data: Dict[str, Any], fallback_id: str) -> EvalRecord:
    return EvalRecord(
        id=str(data.get("id") or fallback_id),
        question=str(data.get("question") or data.get("query") or "").strip(),
        expected_answer=data.get("expected_answer") or data.get("answer"),
        relevant_document_ids=_as_list(data.get("relevant_document_ids")),
        relevant_chunk_ids=_as_list(data.get("relevant_chunk_ids")),
        relevant_articles=_as_list(data.get("relevant_articles")),
        expected_citations=[
            _as_dict(item)
            for item in (data.get("expected_citations") or [])
            if _as_dict(item)
        ],
        query_category=data.get("query_category") or data.get("category"),
        language=str(data.get("language") or "ar"),
        relevance=_as_relevance(data.get("relevance")),
        metadata=_as_dict(data.get("metadata")),
    )


def load_dataset(path: Union[str, Path]) -> List[EvalRecord]:
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()

    if suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict):
            records_data = payload.get("records") or payload.get("items") or []
        elif isinstance(payload, list):
            records_data = payload
        else:
            raise ValueError("JSON dataset must be a list or an object with records/items.")

        return [
            _record_from_mapping(item, fallback_id=str(idx))
            for idx, item in enumerate(records_data, start=1)
        ]

    if suffix == ".jsonl":
        records: List[EvalRecord] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                records.append(_record_from_mapping(item, fallback_id=str(idx)))
        return records

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        records = []
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for idx, row in enumerate(reader, start=1):
                records.append(_record_from_mapping(dict(row), fallback_id=str(idx)))
        return records

    raise ValueError(f"Unsupported dataset format: {suffix}")