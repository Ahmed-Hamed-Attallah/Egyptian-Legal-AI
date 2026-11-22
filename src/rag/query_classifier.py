# src/rag/query_classifier.py
"""
Rule-based query classifier for Egyptian legal questions.
"""
import re
import logging
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    SINGLE_CONCEPT = "single_concept"
    MULTI_CONCEPT = "multi_concept"
    ARTICLE_LOOKUP = "article_lookup"


class QueryClassifier:
    """
    Classifies queries to determine the optimal retrieval strategy.
    Rule-based for zero-latency classification.
    """

    # Patterns indicating multi-concept questions
    MULTI_CONCEPT_CONJUNCTIONS = [
        r"\sو\s",  # " و " (and)
        r"\sمع\s", # " مع " (with)
        r"\sأو\s", # " أو " (or)
        r"\sثم\s", # " ثم " (then)
        r"\sأثناء\s", # " أثناء " (during)
    ]

    # Pattern for direct article lookup
    ARTICLE_PATTERN = re.compile(r"(?:المادة|مادة)\s*(?:رقم\s*)?(\d+|[٠-٩]+)")

    def classify(self, question: str) -> QueryIntent:
        if not question:
            return QueryIntent.SINGLE_CONCEPT

        # 1. Check for direct article lookup
        if self.ARTICLE_PATTERN.search(question):
            logger.debug("Query classified as ARTICLE_LOOKUP")
            return QueryIntent.ARTICLE_LOOKUP

        # 2. Check for multi-concept (multiple legal concepts joined)
        for pattern in self.MULTI_CONCEPT_CONJUNCTIONS:
            if re.search(pattern, question):
                logger.debug("Query classified as MULTI_CONCEPT")
                return QueryIntent.MULTI_CONCEPT

        logger.debug("Query classified as SINGLE_CONCEPT")
        return QueryIntent.SINGLE_CONCEPT