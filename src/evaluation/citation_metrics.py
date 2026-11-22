# src/evaluation/citation_metrics.py
"""
Citation evaluation metrics.

Citation correctness is evaluated against expected citations and retrieved
evidence. Formatting alone is not treated as correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

from src.evaluation.generation_metrics import normalize_digits

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN_DIGITS = "0123456789"
DIGIT_TRANS = str.maketrans(ARABIC_DIGITS, LATIN_DIGITS)

CITATION_TEXT_RE = re.compile(
    r"(قانون|مرسوم|قرار)\s+رقم\s+([\d٠-٩]+)\s+(?:لسنة|لعام)\s+([\d٠-٩]{4})(?:.*?المادة\s+([\d٠-٩]+))?",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class CitationMetricResult:
    metric: str
    value: Optional[float]
    evaluated_count: int
    skipped_count: int


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_citation(citation: Any) -> Optional[str]:
    if citation is None:
        return None

    if isinstance(citation, str):
        match = CITATION_TEXT_RE.search(normalize_digits(citation))
        if not match:
            return None
        instrument, number, year, article = match.groups()
        key = f"{instrument}:{number}:{year}"
        if article:
            key += f":article:{article}"
        return key

    law_number = _normalize_number(_get(citation, "law_number"))
    law_year = _normalize_number(_get(citation, "law_year"))
    article_number = _normalize_number(_get(citation, "article_number"))
    law_name = _get(citation, "law_name") or _get(citation, "source") or "legal_source"

    if not law_number or not law_year:
        return None

    key = f"{law_name}:{law_number}:{law_year}"
    if article_number:
        key += f":article:{article_number}"
    return key


def _normalize_number(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = normalize_digits(str(value)).strip()
    return text or None


def extract_citations_from_answer(answer: str) -> Set[str]:
    citations: Set[str] = set()
    for match in CITATION_TEXT_RE.finditer(normalize_digits(answer or "")):
        instrument, number, year, article = match.groups()
        key = f"{instrument}:{number}:{year}"
        if article:
            key += f":article:{article}"
        citations.add(key)
    return citations


def citation_keys_from_objects(citations: Iterable[Any]) -> Set[str]:
    keys: Set[str] = set()
    for citation in citations or []:
        key = _normalize_citation(citation)
        if key:
            keys.add(key)
    return keys


def evidence_chunk_ids(retrieved_chunks: Iterable[Any]) -> Set[str]:
    ids: Set[str] = set()
    for chunk in retrieved_chunks or []:
        chunk_id = _get(chunk, "chunk_id") or _get(chunk, "id")
        if chunk_id:
            ids.add(str(chunk_id))
    return ids


def compute_citation_metrics(
    records: Iterable[Any],
    generation_outputs: Dict[str, Dict[str, Any]],
) -> Dict[str, CitationMetricResult]:
    records = list(records)

    precision_values: List[float] = []
    recall_values: List[float] = []
    completeness_values: List[float] = []
    grounding_values: List[float] = []

    skipped = 0

    for record in records:
        output = generation_outputs.get(record.id)
        if not output:
            skipped += 1
            continue

        answer = str(output.get("answer", "") or "")
        actual_citations = output.get("citations", []) or []
        retrieved_chunks = output.get("retrieved_chunks", []) or []

        actual_keys = citation_keys_from_objects(actual_citations) | extract_citations_from_answer(answer)
        expected_keys = citation_keys_from_objects(getattr(record, "expected_citations", []) or [])

        if expected_keys:
            correct = actual_keys & expected_keys
            if actual_keys:
                precision_values.append(len(correct) / len(actual_keys))
            recall_values.append(len(correct) / len(expected_keys))

        if actual_keys:
            complete_count = 0
            for key in actual_keys:
                parts = key.split(":")
                if len(parts) >= 3 and all(part.strip() for part in parts[:3]):
                    complete_count += 1
            completeness_values.append(complete_count / len(actual_keys))

        chunk_ids = evidence_chunk_ids(retrieved_chunks)
        citation_source_ids = {
            str(_get(c, "source_id") or _get(c, "chunk_id") or "")
            for c in actual_citations
        }
        citation_source_ids.discard("")

        if citation_source_ids:
            grounded = citation_source_ids & chunk_ids
            grounding_values.append(len(grounded) / len(citation_source_ids))

    def _mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    return {
        "citation_precision": CitationMetricResult(
            metric="citation_precision",
            value=_mean(precision_values),
            evaluated_count=len(precision_values),
            skipped_count=skipped,
        ),
        "citation_recall": CitationMetricResult(
            metric="citation_recall",
            value=_mean(recall_values),
            evaluated_count=len(recall_values),
            skipped_count=skipped,
        ),
        "citation_completeness": CitationMetricResult(
            metric="citation_completeness",
            value=_mean(completeness_values),
            evaluated_count=len(completeness_values),
            skipped_count=skipped,
        ),
        "citation_grounding": CitationMetricResult(
            metric="citation_grounding",
            value=_mean(grounding_values),
            evaluated_count=len(grounding_values),
            skipped_count=skipped,
        ),
    }