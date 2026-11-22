
"""
CSV Column Semantic Classifier
================================

Purpose:
    Classify CSV columns for Egyptian legal document ingestion.

Architecture:
    Stage 1: Column name classification
             - English and Arabic aliases
             - Regex-based semantic detection

    Stage 2: Content analysis
             - Value lengths
             - Numeric and year patterns
             - Arabic legal article patterns
             - Repeated values
             - Content/name contradictions

    Stage 3: LLM classification
             - Only ambiguous or conflicting columns
             - Considers all columns together
             - Uses actual sample values

    Stage 4: Schema validation
             - Detect multiple text columns
             - Detect missing text columns
             - Detect classification conflicts
             - Require review when uncertain

Important:
    Never automatically select the longest column as legal text.
    Never trust a column name without checking its content.

Called once per CSV file.
Results can be cached using file_hash.
"""

import json
import logging
import re
import unicodedata

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple


logger = logging.getLogger(__name__)


# ============================================================
# 1. COLUMN ROLES
# ============================================================

class ColumnRole(str, Enum):

    TEXT_CONTENT = "text_content"

    LAW_NAME = "law_name"
    LAW_NUMBER = "law_number"
    LAW_YEAR = "law_year"

    ARTICLE_NUMBER = "article_number"

    CATEGORY = "category"
    TITLE = "title"

    SOURCE = "source"
    DATE = "date"
    URL = "url"

    ID = "id"
    INDEX = "index"

    METADATA_ONLY = "metadata_only"

    UNKNOWN = "unknown"


SKIP_ROLES = {
    ColumnRole.ID,
    ColumnRole.INDEX,
}

METADATA_ROLES = {
    ColumnRole.LAW_NAME,
    ColumnRole.LAW_NUMBER,
    ColumnRole.LAW_YEAR,
    ColumnRole.ARTICLE_NUMBER,
    ColumnRole.CATEGORY,
    ColumnRole.TITLE,
    ColumnRole.SOURCE,
    ColumnRole.DATE,
    ColumnRole.URL,
    ColumnRole.METADATA_ONLY,
}

CONTENT_ROLE = ColumnRole.TEXT_CONTENT


# ============================================================
# 2. DATA STRUCTURES
# ============================================================

@dataclass
class ColumnClassification:
    """
    Classification result for one column.

    confidence:
        Final classification confidence.

    name_confidence:
        Confidence derived from the column name.

    content_confidence:
        Confidence derived from actual values.

    method:
        rule_name, rule_content, llm, or combined.
    """

    name: str

    role: ColumnRole

    confidence: float = 0.0

    method: str = "rule"

    name_confidence: float = 0.0

    content_confidence: float = 0.0

    needs_review: bool = False

    reason: str = ""


@dataclass
class ColumnStats:
    """Statistical information about a CSV column."""

    name: str

    total_rows: int = 0

    non_empty_count: int = 0

    unique_count: int = 0

    unique_ratio: float = 0.0

    average_length: float = 0.0

    median_length: float = 0.0

    max_length: int = 0

    numeric_ratio: float = 0.0

    year_ratio: float = 0.0

    article_pattern_ratio: float = 0.0

    arabic_ratio: float = 0.0

    samples: List[str] = field(default_factory=list)


@dataclass
class CSVColumnSchema:
    """
    Final CSV schema.

    text_column:
        Selected legal text column.
        None means no reliable text column was identified.

    metadata_columns:
        Original column name -> semantic role.

    skip_columns:
        IDs and indexes.

    text_candidates:
        All columns classified as possible legal text.

    needs_review:
        True if the schema is uncertain or conflicting.

    issues:
        Human-readable validation problems.
    """

    text_column: Optional[str] = None

    metadata_columns: Dict[str, ColumnRole] = field(
        default_factory=dict
    )

    skip_columns: List[str] = field(
        default_factory=list
    )

    classifications: List[ColumnClassification] = field(
        default_factory=list
    )

    text_candidates: List[str] = field(
        default_factory=list
    )

    unclassified_columns: List[str] = field(
        default_factory=list
    )

    needs_review: bool = False

    issues: List[str] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Extract metadata from a row
    # --------------------------------------------------------

    def get_metadata_for_row(
        self,
        row: Dict[str, Any],
    ) -> Dict[str, Any]:

        metadata = {}

        role_counts = Counter(
            role.value
            for role in self.metadata_columns.values()
        )

        for col, role in self.metadata_columns.items():

            value = row.get(col)

            if _is_empty(value):
                continue

            # Preserve the old role-based key when unique.
            # If multiple columns share a role, use their
            # original column names to avoid overwriting data.

            if role_counts[role.value] == 1:
                key = role.value
            else:
                key = col

            metadata[key] = str(value).strip()

        return metadata

    # --------------------------------------------------------
    # Extract text from a row
    # --------------------------------------------------------

    def get_text_for_row(
        self,
        row: Dict[str, Any],
    ) -> str:

        if self.text_column is None:
            return ""

        value = row.get(self.text_column)

        if _is_empty(value):
            return ""

        return str(value).strip()

    # --------------------------------------------------------
    # Get classification for a column
    # --------------------------------------------------------

    def get_classification(
        self,
        column_name: str,
    ) -> Optional[ColumnClassification]:

        for classification in self.classifications:

            if classification.name == column_name:
                return classification

        return None


def _is_empty(value: Any) -> bool:
    """Check whether a CSV value is empty."""

    if value is None:
        return True

    text = str(value).strip()

    return text.lower() in {
        "",
        "nan",
        "none",
        "<na>",
        "nat",
    }


# ============================================================
# 3. COLUMN NAME NORMALIZATION
# ============================================================

def _normalize_column_name(name: str) -> str:
    """
    Normalize Arabic and English column names.

    Examples:
        Law Name       -> law_name
        اسم القانون   -> اسم_القانون
        النص القانوني -> النص_القانوني
    """

    name = str(name).strip().lower()

    # Remove BOM and Unicode formatting characters.
    name = name.replace("\ufeff", "")

    name = unicodedata.normalize("NFKC", name)

    # Remove Arabic diacritics.
    name = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        name,
    )

    # Normalize Arabic letter variants.
    name = name.replace("أ", "ا")
    name = name.replace("إ", "ا")
    name = name.replace("آ", "ا")
    name = name.replace("ى", "ي")

    # Replace spaces and punctuation with underscores.
    name = re.sub(
        r"[\s\-./\\]+",
        "_",
        name,
    )

    name = re.sub(
        r"[^\w]",
        "",
        name,
        flags=re.UNICODE,
    )

    name = re.sub(
        r"_+",
        "_",
        name,
    )

    return name.strip("_")


# ============================================================
# 4. STAGE 1: NAME-BASED CLASSIFICATION
# ============================================================

_NAME_PATTERNS: List[
    Tuple[re.Pattern, ColumnRole, float]
] = [

    # --------------------------------------------------------
    # Legal text
    # --------------------------------------------------------

    (
        re.compile(
            r"^(text|content|body|"
            r"article_text|law_text|"
            r"legal_text|document_text|"
            r"full_text|"
            r"النص|المحتوي|المحتوى|"
            r"نص|نص_القانون|"
            r"النص_القانوني|"
            r"متن_القانون|"
            r"نص_المادة|"
            r"النص_الكامل)$",
            re.I,
        ),
        ColumnRole.TEXT_CONTENT,
        0.95,
    ),

    # --------------------------------------------------------
    # Law name
    # --------------------------------------------------------

    (
        re.compile(
            r"^(law_name|document_name|"
            r"law_title|"
            r"اسم_القانون|"
            r"القانون|"
            r"اسم_التشريع|"
            r"التشريع|"
            r"اسم_الوثيقة)$",
            re.I,
        ),
        ColumnRole.LAW_NAME,
        0.95,
    ),

    # --------------------------------------------------------
    # Law number
    # --------------------------------------------------------

    (
        re.compile(
            r"^(law_number|law_no|"
            r"law_id|"
            r"رقم_القانون|"
            r"رقم_التشريع|"
            r"رقم_القانون_رقم|"
            r"رقم_القانون_او_التشريع)$",
            re.I,
        ),
        ColumnRole.LAW_NUMBER,
        0.95,
    ),

    # --------------------------------------------------------
    # Law year
    # --------------------------------------------------------

    (
        re.compile(
            r"^(law_year|year|"
            r"issue_year|"
            r"سنة|السنة|"
            r"سنة_الإصدار|"
            r"سنة_الاصدار|"
            r"عام_الإصدار|"
            r"عام_الاصدار)$",
            re.I,
        ),
        ColumnRole.LAW_YEAR,
        0.95,
    ),

    # --------------------------------------------------------
    # Article number
    # --------------------------------------------------------

    (
        re.compile(
            r"^(article_number|article_no|"
            r"article_id|"
            r"رقم_المادة|"
            r"المادة_رقم|"
            r"المادة|"
            r"رقم_البند)$",
            re.I,
        ),
        ColumnRole.ARTICLE_NUMBER,
        0.95,
    ),

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    (
        re.compile(
            r"^(category|categories|"
            r"legal_category|"
            r"type|"
            r"التصنيف|النوع|الفئة|"
            r"المجال|القسم|التصنيف_القانوني)$",
            re.I,
        ),
        ColumnRole.CATEGORY,
        0.90,
    ),

    # --------------------------------------------------------
    # Document title
    # --------------------------------------------------------

    (
        re.compile(
            r"^(title|document_title|"
            r"article_title|"
            r"العنوان|عنوان_المادة|"
            r"عنوان_الوثيقة)$",
            re.I,
        ),
        ColumnRole.TITLE,
        0.90,
    ),

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    (
        re.compile(
            r"^(source|reference|"
            r"مصدر|المصدر|"
            r"المرجع|المراجع|"
            r"جهة_النشر)$",
            re.I,
        ),
        ColumnRole.SOURCE,
        0.85,
    ),

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    (
        re.compile(
            r"^(date|issued_date|"
            r"publication_date|"
            r"تاريخ|التاريخ|"
            r"تاريخ_الإصدار|"
            r"تاريخ_الاصدار|"
            r"تاريخ_النشر)$",
            re.I,
        ),
        ColumnRole.DATE,
        0.90,
    ),

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    (
        re.compile(
            r"^(url|link|source_url|"
            r"الرابط|رابط_المصدر)$",
            re.I,
        ),
        ColumnRole.URL,
        0.95,
    ),

    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------

    (
        re.compile(
            r"^(id|row_id|"
            r"internal_id|"
            r"معرف|المعرف|"
            r"رقم_تسلسلي)$",
            re.I,
        ),
        ColumnRole.ID,
        0.95,
    ),

    # --------------------------------------------------------
    # Indexes
    # --------------------------------------------------------

    (
        re.compile(
            r"^(index|row_index|"
            r"source_row|"
            r"segment_index|"
            r"chunk_index|"
            r"row_number|"
            r"__index)$",
            re.I,
        ),
        ColumnRole.INDEX,
        0.95,
    ),
    
    # --------------------------------------------------------
    # IDs
    # --------------------------------------------------------
    (
        re.compile(
            r"^(id|row_id|document_id|doc_id|source_id|"  # <--- Added document_id, doc_id, source_id
            r"internal_id|"
            r"معرف|المعرف|"
            r"رقم_تسلسلي)$",
            re.I,
        ),
        ColumnRole.ID,
        0.95,
    ),

    # --------------------------------------------------------
    # Metadata / System Columns (Skip or Metadata Only)
    # --------------------------------------------------------
    (
        re.compile(
            r"^(char_count|word_count|token_count|length|"
            r"size|bytes|metadata|extra)$",
            re.I,
        ),
        ColumnRole.METADATA_ONLY, # Or ColumnRole.INDEX if you want to skip it entirely
        0.90,
    )
]


def _classify_by_name(
    col_name: str,
) -> Optional[ColumnClassification]:
    """Classify a column using its normalized name."""

    normalized = _normalize_column_name(col_name)

    for pattern, role, confidence in _NAME_PATTERNS:

        if pattern.fullmatch(normalized):

            return ColumnClassification(
                name=col_name,
                role=role,
                confidence=confidence,
                name_confidence=confidence,
                method="rule_name",
                reason="Matched column name pattern.",
            )

    return None


# ============================================================
# 5. STAGE 2: CONTENT ANALYSIS
# ============================================================

_ARABIC_LEGAL_PATTERNS = [
    r"المادة\s+[\(]?\s*\d+",
    r"مادة\s+[\(]?\s*\d+",
    r"الباب\s+(الأول|الثاني|الثالث|\d+)",
    r"الفصل\s+(الأول|الثاني|الثالث|\d+)",
    r"بموجب\s+القانون",
    r"يُعمل\s+بهذا\s+القانون",
    r"على\s+كل\s+شخص",
    r"لا\s+يجوز",
    r"يلتزم",
    r"يعاقب",
    r"تسري\s+أحكام",
]

_LEGAL_PATTERN_REGEX = re.compile(
    "|".join(_ARABIC_LEGAL_PATTERNS),
    re.UNICODE,
)


def _calculate_column_stats(
    col_name: str,
    values: List[str],
    total_rows: int,
    max_samples: int = 10,
) -> ColumnStats:
    """Calculate useful statistics for one column."""

    non_empty = [
        str(value).strip()
        for value in values
        if not _is_empty(value)
    ]

    lengths = [
        len(value)
        for value in non_empty
    ]

    if lengths:

        sorted_lengths = sorted(lengths)

        middle = len(sorted_lengths) // 2

        if len(sorted_lengths) % 2:

            median = sorted_lengths[middle]

        else:

            median = (
                sorted_lengths[middle - 1]
                + sorted_lengths[middle]
            ) / 2

        average = sum(lengths) / len(lengths)

        maximum = max(lengths)

    else:

        median = 0
        average = 0
        maximum = 0

    unique_count = len(set(non_empty))

    numeric_count = sum(
        bool(re.fullmatch(r"[\d٠-٩۰-۹]+", v))
        for v in non_empty
    )

    year_count = sum(
        bool(re.fullmatch(r"(19|20)\d{2}", v))
        for v in non_empty
    )

    article_count = sum(
        bool(
            re.search(
                r"(المادة|مادة)\s*[\(\[]?\s*[\d٠-٩۰-۹]+",
                v,
            )
        )
        for v in non_empty
    )

    arabic_count = sum(
        bool(re.search(r"[\u0600-\u06FF]", v))
        for v in non_empty
    )

    count = len(non_empty) or 1

    return ColumnStats(
        name=col_name,
        total_rows=total_rows,
        non_empty_count=len(non_empty),
        unique_count=unique_count,
        unique_ratio=unique_count / count,
        average_length=average,
        median_length=median,
        max_length=maximum,
        numeric_ratio=numeric_count / count,
        year_ratio=year_count / count,
        article_pattern_ratio=article_count / count,
        arabic_ratio=arabic_count / count,
        samples=non_empty[:max_samples],
    )


def _classify_by_content(
    col_name: str,
    sample_values: List[str],
) -> Optional[ColumnClassification]:
    """
    Classify using actual column values.

    This function intentionally avoids classifying
    every numeric column as a law number.

    Numeric columns can be IDs, article numbers,
    law numbers, or indexes.
    """

    if not sample_values:
        return None

    stats = _calculate_column_stats(
        col_name=col_name,
        values=sample_values,
        total_rows=len(sample_values),
    )

    if stats.non_empty_count == 0:
        return None

    # --------------------------------------------------------
    # Year detection
    # --------------------------------------------------------

    if stats.year_ratio >= 0.90:

        return ColumnClassification(
            name=col_name,
            role=ColumnRole.LAW_YEAR,
            confidence=0.85,
            content_confidence=0.85,
            method="rule_content",
            reason="Most values match four-digit years.",
        )

    # --------------------------------------------------------
    # Legal text detection
    # --------------------------------------------------------

    # Long text is strong evidence, but not absolute proof.

    if stats.average_length >= 250:

        return ColumnClassification(
            name=col_name,
            role=ColumnRole.TEXT_CONTENT,
            confidence=0.95,
            content_confidence=0.95,
            method="rule_content",
            reason="Values contain long text passages.",
        )

    if (
        stats.average_length >= 100
        and stats.max_length >= 150
    ):

        return ColumnClassification(
            name=col_name,
            role=ColumnRole.TEXT_CONTENT,
            confidence=0.85,
            content_confidence=0.85,
            method="rule_content",
            reason="Values contain substantial text passages.",
        )

    # --------------------------------------------------------
    # Legal article patterns
    # --------------------------------------------------------

    if (
        stats.article_pattern_ratio >= 0.50
        and stats.average_length >= 60
    ):

        return ColumnClassification(
            name=col_name,
            role=ColumnRole.TEXT_CONTENT,
            confidence=0.90,
            content_confidence=0.90,
            method="rule_content",
            reason="Values contain legal article patterns.",
        )

    # --------------------------------------------------------
    # Arabic legal language
    # --------------------------------------------------------

    legal_matches = sum(
        bool(_LEGAL_PATTERN_REGEX.search(value))
        for value in stats.samples
    )

    legal_ratio = (
        legal_matches / max(len(stats.samples), 1)
    )

    if (
        legal_ratio >= 0.60
        and stats.average_length >= 80
    ):

        return ColumnClassification(
            name=col_name,
            role=ColumnRole.TEXT_CONTENT,
            confidence=0.85,
            content_confidence=0.85,
            method="rule_content",
            reason="Values contain Egyptian legal language patterns.",
        )

    # --------------------------------------------------------
    # Numeric values
    # --------------------------------------------------------

    # Do not guess whether these are law numbers,
    # article numbers, IDs, or indexes.
    #
    # Leave them for the LLM or column name analysis.

    if stats.numeric_ratio >= 0.90:

        return None

    # --------------------------------------------------------
    # Short repeated values
    # --------------------------------------------------------

    # These could be categories, law names, or titles.
    # Do not force a classification.

    return None


# ============================================================
# 6. COLUMN NAME / CONTENT VALIDATION
# ============================================================

def _validate_name_against_content(
    name_result: Optional[ColumnClassification],
    content_result: Optional[ColumnClassification],
    stats: ColumnStats,
) -> Tuple[
    Optional[ColumnClassification],
    bool,
    str,
]:
    """
    Compare the name-based result against content evidence.

    Returns:
        selected classification
        conflict flag
        explanation
    """

    if name_result is None:

        if content_result is not None:
            return (
                content_result,
                False,
                "Classification based on content.",
            )

        return (
            None,
            False,
            "Insufficient classification evidence.",
        )

    # --------------------------------------------------------
    # Strong content contradicts column name
    # --------------------------------------------------------

    if (
        content_result is not None
        and content_result.role != name_result.role
    ):

        # Example:
        # Column name = text
        # Content = actual law names

        return (
            content_result,
            True,
            (
                f"Column name suggests {name_result.role.value}, "
                f"but content suggests {content_result.role.value}."
            ),
        )

    # --------------------------------------------------------
    # Special validation for text columns
    # --------------------------------------------------------

    if name_result.role == ColumnRole.TEXT_CONTENT:

        # Short content is suspicious for a legal text column.
        #
        # It might contain:
        # - Law names
        # - Article titles
        # - Categories
        # - Short descriptions

        if (
            stats.average_length < 40
            or stats.non_empty_count == 0
        ):

            return (
                None,
                True,
                (
                    "Column name suggests legal text, "
                    "but sample values are too short or empty."
                ),
            )

        # Moderate text length requires additional validation.

        if stats.average_length < 100:

            return (
                None,
                True,
                (
                    "Column name suggests legal text, "
                    "but content evidence is not strong enough."
                ),
            )

        # Long content supports the name classification.

        return (
            ColumnClassification(
                name=name_result.name,
                role=ColumnRole.TEXT_CONTENT,
                confidence=0.95,
                name_confidence=name_result.name_confidence,
                content_confidence=0.85,
                method="combined",
                reason="Column name and content agree.",
            ),
            False,
            "Name and content agree.",
        )

    # --------------------------------------------------------
    # Content contradicts a metadata column
    # --------------------------------------------------------

    if (
        content_result is not None
        and content_result.role == ColumnRole.TEXT_CONTENT
    ):

        return (
            content_result,
            True,
            (
                f"Column name suggests {name_result.role.value}, "
                "but actual values look like legal text."
            ),
        )

    # --------------------------------------------------------
    # No contradiction
    # --------------------------------------------------------

    return (
        name_result,
        False,
        "Column name classification accepted.",
    )


# ============================================================
# 7. STAGE 3: LLM CLASSIFICATION
# ============================================================

LLM_CLASSIFICATION_SYSTEM_PROMPT = """
You are an expert data engineer specializing in
Egyptian legal documents and Arabic legal terminology.

Your task is to classify CSV columns based on:

1. Column names in Arabic or English.
2. Actual sample values.
3. Value lengths and data characteristics.
4. Relationships between columns.
5. The semantic meaning of the data.

AVAILABLE ROLES:

- text_content
- law_name
- law_number
- law_year
- article_number
- category
- title
- source
- date
- url
- id
- index
- metadata_only
- unknown

IMPORTANT RULES:

1. NEVER trust a column name alone.

2. A column named "text", "content", "النص",
   or "نص القانون" is NOT necessarily legal text.

3. A column named "title" may contain a law name,
   an article title, or a document title.

4. Distinguish between:
   - Law name
   - Law number
   - Law year
   - Article number
   - Article title
   - Actual legal provisions

5. Short values are not automatically legal text.

6. Numeric values alone cannot distinguish:
   law numbers, article numbers, IDs, or indexes.

7. Consider relationships between all columns.

8. If evidence is insufficient, return "unknown".

9. Do not force classifications.

10. Prefer semantic correctness over column-name matching.

Return ONLY valid JSON using this structure:

{
    "columns": {
        "column_name": {
            "role": "text_content",
            "confidence": 0.95,
            "reason": "Explanation"
        }
    }
}
"""


LLM_CLASSIFICATION_USER_TEMPLATE = """
Analyze the following Egyptian legal CSV schema.

Classify the requested columns.

IMPORTANT:
Some columns may have already been classified by
rules, but those classifications may be incorrect.

Use the samples to detect contradictions.

CSV INFORMATION:

{columns_info}

COLUMNS REQUIRING CLASSIFICATION:

{target_columns}

Return JSON only.
"""


def _build_llm_prompt(
    columns: List[str],
    sample_data: Dict[str, List[str]],
    unresolved: List[str],
    stats_data: Dict[str, ColumnStats],
) -> str:
    """
    Build an LLM prompt.

    Includes information about the entire CSV,
    not just individual columns.
    """

    cols_info = []

    for col in columns:

        stats = stats_data[col]

        cols_info.append({

            "name": col,

            "samples": [
                str(value)[:300]
                for value in sample_data.get(col, [])[:5]
            ],

            "statistics": {

                "non_empty_count": stats.non_empty_count,

                "unique_count": stats.unique_count,

                "unique_ratio": round(
                    stats.unique_ratio, 3
                ),

                "average_length": round(
                    stats.average_length, 2
                ),

                "median_length": round(
                    stats.median_length, 2
                ),

                "numeric_ratio": round(
                    stats.numeric_ratio, 3
                ),

                "year_ratio": round(
                    stats.year_ratio, 3
                ),

                "article_pattern_ratio": round(
                    stats.article_pattern_ratio, 3
                ),

                "arabic_ratio": round(
                    stats.arabic_ratio, 3
                ),
            },
        })

    return LLM_CLASSIFICATION_USER_TEMPLATE.format(

        columns_info=json.dumps(
            cols_info,
            ensure_ascii=False,
            indent=2,
        ),

        target_columns=json.dumps(
            unresolved,
            ensure_ascii=False,
            indent=2,
        ),
    )


def _parse_llm_response(
    response_text: str,
    valid_columns: List[str],
) -> Dict[str, ColumnClassification]:
    """Safely parse LLM JSON output."""

    results = {}

    if not response_text:
        return results

    try:

        # Support JSON wrapped in Markdown code fences.

        cleaned = re.sub(
            r"```(?:json)?",
            "",
            response_text,
            flags=re.I,
        ).replace("```", "").strip()

        # Extract JSON without relying on a greedy regex.

        decoder = json.JSONDecoder()

        start = cleaned.find("{")

        if start == -1:
            return results

        parsed, _ = decoder.raw_decode(
            cleaned[start:]
        )

        # Support both response formats:
        #
        # {"column": {...}}
        #
        # and:
        #
        # {"columns": {"column": {...}}}

        if "columns" in parsed:
            parsed = parsed["columns"]

        if not isinstance(parsed, dict):
            return results

        for col_name, info in parsed.items():

            if col_name not in valid_columns:
                continue

            if not isinstance(info, dict):
                continue

            try:

                role = ColumnRole(
                    info.get("role", "unknown")
                )

            except ValueError:

                role = ColumnRole.UNKNOWN

            try:

                confidence = float(
                    info.get("confidence", 0.5)
                )

            except (ValueError, TypeError):

                confidence = 0.5

            confidence = max(
                0.0,
                min(1.0, confidence),
            )

            results[col_name] = ColumnClassification(

                name=col_name,

                role=role,

                confidence=confidence,

                method="llm",

                reason=str(
                    info.get("reason", "")
                ),
            )

    except (
        json.JSONDecodeError,
        TypeError,
        AttributeError,
    ) as error:

        logger.warning(
            "Failed to parse LLM classification: %s",
            error,
        )

    return results


# ============================================================
# 8. STAGE 4: FINAL SCHEMA VALIDATION
# ============================================================

def _build_final_schema(
    classifications: List[ColumnClassification],
) -> CSVColumnSchema:
    """
    Build and validate the final schema.

    Important:
        Multiple TEXT_CONTENT columns are not silently
        resolved by choosing the first one.
    """

    schema = CSVColumnSchema()

    schema.classifications = classifications

    text_columns = []

    for classification in classifications:

        role = classification.role

        col = classification.name

        if classification.needs_review:

            schema.needs_review = True

        if role == ColumnRole.UNKNOWN:

            schema.unclassified_columns.append(col)

            continue

        if role == ColumnRole.TEXT_CONTENT:

            text_columns.append(classification)

        elif role in SKIP_ROLES:

            schema.skip_columns.append(col)

        elif role in METADATA_ROLES:

            schema.metadata_columns[col] = role

    # --------------------------------------------------------
    # Multiple content columns
    # --------------------------------------------------------

    schema.text_candidates = [
        c.name
        for c in text_columns
    ]

    if len(text_columns) == 1:

        selected = text_columns[0]

        if (
            selected.confidence >= 0.70
            and not selected.needs_review
        ):

            schema.text_column = selected.name

        else:

            schema.needs_review = True

            schema.issues.append(
                "The detected text column has insufficient confidence."
            )

    elif len(text_columns) > 1:

        schema.needs_review = True

        schema.issues.append(
            "Multiple columns were classified as legal text: "
            + ", ".join(schema.text_candidates)
            + ". An explicit content-column selection is required."
        )

    else:

        schema.needs_review = True

        schema.issues.append(
            "No reliable legal text column was detected."
        )

    # --------------------------------------------------------
    # Unclassified columns
    # --------------------------------------------------------

    if schema.unclassified_columns:

        schema.needs_review = True

        schema.issues.append(
            "Some columns could not be classified: "
            + ", ".join(schema.unclassified_columns)
        )

    return schema


# ============================================================
# 9. MAIN ENTRY POINT
# ============================================================

_schema_cache: Dict[str, CSVColumnSchema] = {}


def classify_csv_columns(
    df,
    file_hash: Optional[str] = None,
    llm_provider=None,
    max_sample_rows: int = 10,
    use_llm: bool = True,
    llm_confidence_threshold: float = 0.70,
) -> CSVColumnSchema:
    """
    Classify columns in a pandas DataFrame.

    Args:
        df:
            Loaded pandas DataFrame.

        file_hash:
            Optional file hash for caching.

        llm_provider:
            Optional ProviderRouter or compatible LLM provider.

        max_sample_rows:
            Number of rows sampled from each column.

        use_llm:
            Whether to use LLM for ambiguous columns.

        llm_confidence_threshold:
            Minimum confidence accepted from LLM.

    Returns:
        CSVColumnSchema.

    IMPORTANT:
        If schema.needs_review is True, inspect the schema
        before generating chunks.
    """

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    if file_hash and file_hash in _schema_cache:

        logger.info(
            "Using cached column schema: %s",
            file_hash[:12],
        )

        return _schema_cache[file_hash]

    columns = list(df.columns)

    if not columns:

        schema = CSVColumnSchema(
            needs_review=True,
            issues=["CSV contains no columns."],
        )

        return schema

    # --------------------------------------------------------
    # Collect samples and statistics
    # --------------------------------------------------------

    sample_data: Dict[str, List[str]] = {}

    stats_data: Dict[str, ColumnStats] = {}

    total_rows = len(df)

    for col in columns:

        values = (
            df[col]
            .dropna()
            .astype(str)
            .head(max_sample_rows)
            .tolist()
        )

        sample_data[col] = values

        # Statistics from the sampled values.

        stats_data[col] = _calculate_column_stats(

            col_name=col,

            values=values,

            total_rows=total_rows,

        )

    # --------------------------------------------------------
    # Stage 1 + Stage 2
    # --------------------------------------------------------

    classifications = []

    unresolved = []

    for col in columns:

        name_result = _classify_by_name(col)

        content_result = _classify_by_content(
            col,
            sample_data.get(col, []),
        )

        stats = stats_data[col]

        selected, conflict, reason = (
            _validate_name_against_content(
                name_result,
                content_result,
                stats,
            )
        )

        # ----------------------------------------------------
        # Conflict detected
        # ----------------------------------------------------

        if conflict:

            logger.warning(
                "Column conflict detected: '%s' - %s",
                col,
                reason,
            )

            unresolved.append(col)

            # Preserve the best provisional evidence.
            # The LLM will review it.

            if selected is None:

                provisional = ColumnClassification(
                    name=col,
                    role=ColumnRole.UNKNOWN,
                    confidence=0.0,
                    method="rule_conflict",
                    needs_review=True,
                    reason=reason,
                )

            else:

                provisional = selected

                provisional.needs_review = True

                provisional.reason = reason

            classifications.append(provisional)

            continue

        # ----------------------------------------------------
        # No conflict
        # ----------------------------------------------------

        if selected is None:

            classification = ColumnClassification(

                name=col,

                role=ColumnRole.UNKNOWN,

                confidence=0.0,

                method="rule",

                reason=reason,
            )

            unresolved.append(col)

        else:

            classification = selected

            # A confident rule-based classification
            # does not automatically require LLM.

        classifications.append(classification)

    # --------------------------------------------------------
    # Stage 3: LLM
    # --------------------------------------------------------

    if use_llm and llm_provider is not None and unresolved:

        logger.info(
            "LLM reviewing columns: %s",
            unresolved,
        )

        try:

            from src.domain.responses import GenerationRequest

            prompt = _build_llm_prompt(

                columns=columns,

                sample_data=sample_data,

                unresolved=unresolved,

                stats_data=stats_data,

            )

            request = GenerationRequest(

                system_prompt=LLM_CLASSIFICATION_SYSTEM_PROMPT,

                user_prompt=prompt,

                temperature=0.0,

                max_tokens=1500,

            )

            response = llm_provider.generate(request)

            llm_results = _parse_llm_response(

                response_text=response.content,

                valid_columns=columns,

            )

            for col_name, llm_result in llm_results.items():

                if col_name not in unresolved:
                    continue

                if (
                    llm_result.confidence
                    < llm_confidence_threshold
                ):

                    logger.warning(
                        "LLM confidence too low for '%s': %.2f",
                        col_name,
                        llm_result.confidence,
                    )

                    continue

                # Replace the provisional classification.

                classifications = [

                    llm_result
                    if c.name == col_name
                    else c

                    for c in classifications

                ]

                logger.info(
                    "LLM classified '%s' -> %s (%.2f)",
                    col_name,
                    llm_result.role.value,
                    llm_result.confidence,
                )

        except Exception as error:

            logger.exception(
                "LLM column classification failed: %s",
                error,
            )

    # --------------------------------------------------------
    # Stage 4: Final validation
    # --------------------------------------------------------

    schema = _build_final_schema(
        classifications
    )

    # --------------------------------------------------------
    # Additional validation
    # --------------------------------------------------------

    if schema.text_column is None:

        logger.warning(
            "No reliable text column detected."
        )

    if schema.needs_review:

        logger.warning(
            "CSV schema requires review: %s",
            schema.issues,
        )

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    if file_hash:

        _schema_cache[file_hash] = schema

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger.info(
        """
        CSV classification completed.

        Text column: %s
        Text candidates: %s
        Metadata columns: %s
        Skip columns: %s
        Unclassified: %s
        Needs review: %s
        Issues: %s
        """,
        schema.text_column,
        schema.text_candidates,
        schema.metadata_columns,
        schema.skip_columns,
        schema.unclassified_columns,
        schema.needs_review,
        schema.issues,
    )

    return schema


# ============================================================
# 10. CACHE MANAGEMENT
# ============================================================

def clear_schema_cache():
    """Clear all cached CSV schemas."""

    _schema_cache.clear()

    logger.info(
        "CSV schema cache cleared."
    )
