# src/evaluation/arabic_evaluation.py
"""
Arabic legal query evaluation helpers.

This module reuses the existing ArabicNormalizer where available. It does not
apply aggressive normalization to legal identifiers because law numbers, article
numbers, and dates carry legal identity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from src.evaluation.generation_metrics import normalize_digits

try:
    from src.preprocessing.arabic_normalizer import ArabicNormalizer

    _NORMALIZER = ArabicNormalizer()
except Exception:
    _NORMALIZER = None


QUERY_CATEGORY_PATTERNS = {
    "article_lookup": re.compile(r"المادة|ماده|مادة", re.IGNORECASE),
    "law_lookup": re.compile(r"قانون|مرسوم|قرار|لائحة", re.IGNORECASE),
    "amendment": re.compile(r"تعديل|عدل|ألغي|إلغاء|يستبدل|أضيف", re.IGNORECASE),
    "ruling": re.compile(r"طعن|حكم|محكمة|نقض|استئناف", re.IGNORECASE),
    "procedure": re.compile(r"إجراء|دعوى|خصومة|تسجيل|رفع", re.IGNORECASE),
    "conceptual": re.compile(r"ما هو|ما تعريف|شرح|يفسر|معنى", re.IGNORECASE),
}

MIXED_LANGUAGE_RE = re.compile(r"[A-Za-z]")


def normalize_query_text(text: str) -> str:
    text = normalize_digits(text or "")

    if _NORMALIZER is not None:
        try:
            result = _NORMALIZER.normalize(text)
            normalized = getattr(result, "text_normalized", None) or getattr(result, "text", None)
            if normalized:
                return normalize_digits(str(normalized))
        except Exception:
            pass

    return text


def classify_query(record: Any) -> str:
    question = normalize_query_text(getattr(record, "question", "") or "")

    for category, pattern in QUERY_CATEGORY_PATTERNS.items():
        if pattern.search(question):
            if MIXED_LANGUAGE_RE.search(question):
                return f"{category}_mixed_language"
            return category

    if MIXED_LANGUAGE_RE.search(question):
        return "mixed_language"

    return "general"


def group_records_by_category(records: Iterable[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for record in records:
        category = getattr(record, "query_category", None) or classify_query(record)
        grouped.setdefault(category, []).append(record)
    return grouped