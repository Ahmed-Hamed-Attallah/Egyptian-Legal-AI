# src/retrieval/legal_synonyms.py
"""
Arabic legal synonym expansion for BM25 recall improvement.
"""
import logging
from typing import List

logger = logging.getLogger(__name__)

# Curated Egyptian legal synonyms (query-side expansion only)
LEGAL_SYNONYMS = {
    "سرقة": ["اختلاس", "استيلاء", "نهب", "سطو"],
    "قتل": ["إزهاق", "اعتداء مميت", "جريمة قتل"],
    "عقوبة": ["جزاء", "حكم", "إدانة", "مؤاخذة"],
    "حبس": ["سجن", "اعتقال", "تقييد الحرية"],
    "غرامة": ["جزاء مالي", "عقوبة مالية"],
    "تعويض": ["رد", "جبر الضرر", "تعويض مدني"],
    "نصب": ["احتيال", "خداع", "استيلاء بالغش"],
}


def expand_query_with_synonyms(query: str) -> str:
    """
    Expands a query by appending synonyms for recognized legal terms.
    Only expands the query (never the index) to preserve verbatim legal text.
    """
    if not query:
        return query

    tokens = query.split()
    expanded_terms = list(tokens)

    for token in tokens:
        # Strip punctuation for matching
        clean_token = "".join(c for c in token if c.isalpha())
        if clean_token in LEGAL_SYNONYMS:
            expanded_terms.extend(LEGAL_SYNONYMS[clean_token])

    expanded = " ".join(expanded_terms)
    if expanded != query:
        logger.debug(f"Expanded query with synonyms: {expanded}")
    return expanded