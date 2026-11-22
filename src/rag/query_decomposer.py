# src/rag/query_decomposer.py
"""
LLM-based query decomposition for multi-concept legal questions.
"""
import json
import logging
from typing import List

from src.llm.provider_router import ProviderRouter
from src.domain.responses import GenerationRequest
from src.config.settings import settings

logger = logging.getLogger(__name__)


class QueryDecomposer:
    """
    Decomposes complex legal questions into focused retrieval sub-queries.
    """

    DECOMPOSE_PROMPT = """أنت خبير قانوني مصري. قسّم السؤال القانوني المعقد إلى أسئلة فرعية بسيطة.
كل سؤال فرعي يجب أن يستهدف مفهوماً قانونياً واحداً أو مادة محددة.
احتفظ بالتفاصيل المهمة (مثل: "سرقة المنازل" وليس فقط "سرقة").

السؤال: {question}

أعد النتيجة بصيغة JSON فقط (قائمة من النصوص):
["سؤال فرعي 1", "سؤال فرعي 2"]"""

    def __init__(self, provider_router: ProviderRouter):
        self.provider_router = provider_router

    def decompose(self, question: str) -> List[str]:
        """Returns a list of sub-queries. Always includes the original as a safety net."""
        sub_queries = self._extract_concepts(question)

        # Safety net: always include the original question
        if question not in sub_queries:
            sub_queries.append(question)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in sub_queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        logger.info(f"Decomposed query into {len(unique)} sub-queries.")
        return unique

    def _extract_concepts(self, question: str) -> List[str]:
        try:
            prompt = self.DECOMPOSE_PROMPT.format(question=question)
            request = GenerationRequest(
                system_prompt="أنت مساعد قانوني دقيق. أعد JSON فقط دون أي شرح أو علامات markdown.",
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=150,
            )
            response = self.provider_router.generate(request)
            return self._parse_json_list(response.content)
        except Exception as e:
            logger.warning(f"Query decomposition failed, using original query: {e}")
            return []

    def _parse_json_list(self, text: str) -> List[str]:
        """Robustly parse a JSON list from LLM output."""
        try:
            text = text.strip().strip("`").replace("json", "").strip()
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1:
                return []
            parsed = json.loads(text[start : end + 1])
            return [str(q).strip() for q in parsed if str(q).strip()]
        except Exception as e:
            logger.warning(f"Failed to parse decomposition JSON: {e}")
            return []