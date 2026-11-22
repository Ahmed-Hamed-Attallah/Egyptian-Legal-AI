# src/evaluation/generation_metrics.py
"""
Generation evaluation metrics.

These metrics are deterministic and evidence-grounded. They do not treat LLM
self-assessment as proof of correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN_DIGITS = "0123456789"
DIGIT_TRANS = str.maketrans(ARABIC_DIGITS, LATIN_DIGITS)

ABSTENTION_PHRASES = (
    "لا تتوفر معلومات كافية",
    "لا توجد معلومات كافية",
    "غير متاح في المصادر",
    "لا يمكن الإجابة",
    "المصادر المقدمة لا تحتوي",
)

LEGAL_REF_RE = re.compile(
    r"(قانون|مرسوم|قرار)\s+رقم\s+([\d٠-٩]+)\s+(?:لسنة|لعام)\s+([\d٠-٩]{4})(?:.*?المادة\s+([\d٠-٩]+))?",
    re.IGNORECASE | re.DOTALL,
)

ARTICLE_ONLY_RE = re.compile(r"المادة\s+([\d٠-٩]+)", re.IGNORECASE)


@dataclass
class GenerationMetricResult:
    metric: str
    value: Optional[float]
    evaluated_count: int
    skipped_count: int


def normalize_digits(text: str) -> str:
    return (text or "").translate(DIGIT_TRANS)


def extract_legal_refs(text: str) -> Set[str]:
    text = normalize_digits(text or "")
    refs: Set[str] = set()

    for match in LEGAL_REF_RE.finditer(text):
        instrument = match.group(1)
        number = match.group(2)
        year = match.group(3)
        article = match.group(4)

        ref = f"{instrument}:{number}:{year}"
        if article:
            ref += f":article:{article}"
        refs.add(ref)

    for match in ARTICLE_ONLY_RE.finditer(text):
        refs.add(f"article:{match.group(1)}")

    return refs


def evidence_text(record: Any, retrieved_chunks: Iterable[Any]) -> str:
    parts: List[str] = []

    expected_answer = getattr(record, "expected_answer", None)
    if expected_answer:
        parts.append(str(expected_answer))

    for chunk in retrieved_chunks or []:
        if isinstance(chunk, dict):
            parts.append(str(chunk.get("content", "")))
            metadata = chunk.get("metadata", {}) or {}
            parts.append(str(metadata))
        else:
            parts.append(str(getattr(chunk, "content", "")))
            parts.append(str(getattr(chunk, "metadata", {})))

    return "\n".join(parts)


def token_overlap(answer: str, reference: str) -> Optional[float]:
    answer = normalize_digits(answer or "").lower()
    reference = normalize_digits(reference or "").lower()

    answer_tokens = set(re.findall(r"[\w\u0600-\u06FF]+", answer))
    reference_tokens = set(re.findall(r"[\w\u0600-\u06FF]+", reference))

    if not answer_tokens or not reference_tokens:
        return None

    intersection = answer_tokens & reference_tokens
    return len(intersection) / len(reference_tokens)


def compute_generation_metrics(
    records: Iterable[Any],
    generation_outputs: Dict[str, Dict[str, Any]],
) -> Dict[str, GenerationMetricResult]:
    records = list(records)

    faithfulness_values: List[float] = []
    unsupported_values: List[float] = []
    relevance_values: List[float] = []
    context_utilization_values: List[float] = []
    abstention_values: List[float] = []

    skipped = 0

    for record in records:
        output = generation_outputs.get(record.id)
        if not output:
            skipped += 1
            continue

        answer = str(output.get("answer", "") or "")
        retrieved_chunks = output.get("retrieved_chunks", []) or []
        evidence = evidence_text(record, retrieved_chunks)

        answer_refs = extract_legal_refs(answer)
        evidence_refs = extract_legal_refs(evidence)

        if answer_refs:
            supported = answer_refs & evidence_refs
            faithfulness = len(supported) / len(answer_refs)
            unsupported = 1.0 - faithfulness

            faithfulness_values.append(faithfulness)
            unsupported_values.append(unsupported)

        if evidence_refs:
            used = answer_refs & evidence_refs
            context_utilization_values.append(len(used) / len(evidence_refs))

        overlap = token_overlap(answer, record.expected_answer or "")
        if overlap is not None:
            relevance_values.append(overlap)

        has_evidence = bool(retrieved_chunks) or bool(evidence_refs)
        says_insufficient = any(phrase in answer for phrase in ABSTENTION_PHRASES)

        if not has_evidence:
            abstention_values.append(1.0 if says_insufficient else 0.0)
        elif has_evidence and says_insufficient and not answer_refs:
            abstention_values.append(0.0)

    def _mean(values: List[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    return {
        "faithfulness": GenerationMetricResult(
            metric="faithfulness",
            value=_mean(faithfulness_values),
            evaluated_count=len(faithfulness_values),
            skipped_count=skipped,
        ),
        "unsupported_claim_rate": GenerationMetricResult(
            metric="unsupported_claim_rate",
            value=_mean(unsupported_values),
            evaluated_count=len(unsupported_values),
            skipped_count=skipped,
        ),
        "answer_relevance_lexical": GenerationMetricResult(
            metric="answer_relevance_lexical",
            value=_mean(relevance_values),
            evaluated_count=len(relevance_values),
            skipped_count=skipped,
        ),
        "context_utilization": GenerationMetricResult(
            metric="context_utilization",
            value=_mean(context_utilization_values),
            evaluated_count=len(context_utilization_values),
            skipped_count=skipped,
        ),
        "correct_abstention_rate": GenerationMetricResult(
            metric="correct_abstention_rate",
            value=_mean(abstention_values),
            evaluated_count=len(abstention_values),
            skipped_count=skipped,
        ),
    }