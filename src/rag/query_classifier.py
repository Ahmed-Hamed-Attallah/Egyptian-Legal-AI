# src/rag/query_classifier.py
"""
Rule-based query classifier for Egyptian legal questions.

Classifies a user query into a retrieval intent:
- ARTICLE_LOOKUP  → direct request for a specific article number.
- CONVERSATIONAL  → about the conversation itself (memory recall, follow-up,
                    chitchat). These must NOT trigger legal retrieval
                    (ARCHITECTURE §9): the answer comes from conversation
                    history, not from a legal source, so no citations attach.
- MULTI_CONCEPT   → several legal concepts joined; decompose before retrieval.
- SINGLE_CONCEPT  → default legal question.

The classifier normalizes Arabic before matching (hamza folding, ta-marbuta
folding, diacritics removal) so patterns are written once in folded form.
"""
import re
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    SINGLE_CONCEPT = "single_concept"
    MULTI_CONCEPT = "multi_concept"
    ARTICLE_LOOKUP = "article_lookup"
    CONVERSATIONAL = "conversational"


class QueryClassifier:
    """Rule-based, zero-latency, deterministic, Arabic-aware classifier."""

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    _HAMZA_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"})
    _DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670]")
    _TATWEEL_RE = re.compile(r"\u0640")

    # ------------------------------------------------------------------
    # Patterns (written against NORMALIZED text: no hamza-on-alef, ة→ه)
    # ------------------------------------------------------------------
    # Article lookup: المادة 147 / مادة رقم ١٤٧ / مادة 5
    ARTICLE_PATTERN = re.compile(r"(?:الماده|ماده)\s*(?:رقم\s*)?(\d+|[٠-٩]+)")

    # Explicit references to the conversation / memory.
    _CONVERSATIONAL_REFERENCE_PATTERNS = [
        r"اخر\s+سوال",                                     # اخر سؤال / اخر سوال
        r"السوال\s+(السابق|اللي\s+فات|الاخير)",            # السؤال السابق/الأخير
        r"سوالي\s+(السابق|الاخير|اللي)",                   # سؤالي السابق
        r"(ماذا|ما|ايه)\s+سالتك",                          # ماذا سألتك
        r"هل?\s*تتذكر",                                    # هل تتذكر
        r"(تتذكر|تذكر)\s+(سوالي|المحادثه)",                # تذكر سؤالي/المحادثة
        r"في\s+المحادثه",                                  # في المحادثة السابقة
    ]

    # Continuation / follow-up prompts (anchored so they can't hit mid-sentence
    # legal text). Only treated as conversational when there is NO legal signal.
    _CONTINUATION_PATTERNS = [
        r"^\s*(تابع|استمر|اكمل|كمل)\b",
        r"^\s*(وضح|اشرح|فسر)\s+اكثر",
        r"^\s*(وبعدين|وماذا بعد|ثم ماذا)",
        r"^\s*(زيدني|اعطني المزيد|اعطني تفاصيل)",
    ]

    # Identity / chitchat with no legal content.
    _CHITCHAT_PATTERNS = [
        r"^\s*(مرحبا|اهلا|هلا|السلام عليكم|صباح الخير|مساء الخير|شكرا)\b",
        r"(من انت|مين انت|ما اسمك|عرفني بنفسك)",
        r"(ما هي قدراتك|ماذا تستطيع|ماذا تقدر|كيف تساعدني)",
    ]

    # Multi-concept conjunctions (from the original implementation).
    _MULTI_CONCEPT_CONJUNCTIONS = [
        r"\sو\s", r"\sمع\s", r"\sأو\s", r"\sاو\s", r"\sثم\s",
        r"\sأثناء\s", r"\sاثناء\s",
    ]

    # Strong legal signals that protect legal queries from being misclassified
    # as chitchat/continuation.
    _LEGAL_SIGNALS = [
        "قانون", "مادة", "مواد", "عقوبة", "عقوبات", "جريمة", "جرائم", "محكمة",
        "محاكم", "عقد", "عقود", "تعويض", "سرقة", "نصب", "دستور", "مرسوم",
        "قرار", "لائحة", "نيابة", "قاضي", "حكم", "استئناف", "نقض", "جنحة",
        "جناية", "حبس", "سجن", "غرامة", "عامل", "صاحب عمل", "إيجار", "ايجار",
        "ميراث", "طلاق", "نفقة", "شركة", "تجاري", "مدني", "جنائي", "إجراءات",
        "تشريع",
    ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _normalize(self, text: str) -> str:
        text = self._DIACRITICS_RE.sub("", text)
        text = self._TATWEEL_RE.sub("", text)
        text = text.translate(self._HAMZA_FOLD)
        return text.strip()

    def _has_legal_signal(self, normalized: str) -> bool:
        return any(self._normalize(sig) in normalized for sig in self._LEGAL_SIGNALS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, question: str) -> QueryIntent:
        if not question:
            return QueryIntent.SINGLE_CONCEPT

        normalized = self._normalize(question)

        # 1. Explicit reference to the conversation or memory.
        for pattern in self._CONVERSATIONAL_REFERENCE_PATTERNS:
            if re.search(pattern, normalized):
                logger.debug("Query classified as CONVERSATIONAL (reference)")
                return QueryIntent.CONVERSATIONAL

        # 2. Continuation follow-ups — conversational only when there is no
        #    legal signal (so "اشرح المادة 147" still retrieves).
        if not self._has_legal_signal(normalized):
            for pattern in self._CONTINUATION_PATTERNS:
                if re.search(pattern, normalized):
                    logger.debug("Query classified as CONVERSATIONAL (continuation)")
                    return QueryIntent.CONVERSATIONAL

            # 3. Pure chitchat / identity questions.
            for pattern in self._CHITCHAT_PATTERNS:
                if re.search(pattern, normalized):
                    logger.debug("Query classified as CONVERSATIONAL (chitchat)")
                    return QueryIntent.CONVERSATIONAL

        # 4. Direct article lookup.
        if self.ARTICLE_PATTERN.search(normalized):
            logger.debug("Query classified as ARTICLE_LOOKUP")
            return QueryIntent.ARTICLE_LOOKUP

        # 5. Multi-concept.
        for pattern in self._MULTI_CONCEPT_CONJUNCTIONS:
            if re.search(pattern, normalized):
                logger.debug("Query classified as MULTI_CONCEPT")
                return QueryIntent.MULTI_CONCEPT

        logger.debug("Query classified as SINGLE_CONCEPT")
        return QueryIntent.SINGLE_CONCEPT