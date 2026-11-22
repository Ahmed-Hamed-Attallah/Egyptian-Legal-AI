# src/generation/context_builder.py
"""
Context builder for RAG generation.
"""
import logging
import math
from typing import List

from src.domain.queries import RetrievedChunk
from src.config.settings import settings

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds structured legal context from retrieved chunks.

    Token budgeting:
      - ``max_tokens`` defaults to ``settings.CONTEXT_MAX_TOKENS`` (1500),
        keeping the full request (context + question + answer) safely under
        Groq's free-tier 2000 tokens-per-minute limit. The previous default
        of 4000 guaranteed HTTP 413 errors.
      - Token estimation uses ~2.5 characters per token, a conservative ratio
        for Arabic text on Llama/Qwen-style tokenizers (the old len//4 rule
        underestimated Arabic tokens by ~40%).
    """

    CHARS_PER_TOKEN_AR: float = 2.5

    def __init__(self, max_tokens: int = None):
        self.max_tokens = max_tokens or settings.CONTEXT_MAX_TOKENS

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Conservative Arabic token estimate without loading a tokenizer."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / cls.CHARS_PER_TOKEN_AR))

    def build_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Build formatted context from retrieved chunks.
        Preserves legal metadata and prevents duplication.
        """
        if not chunks:
            return "لا تتوفر نصوص قانونية ذات صلة."

        context_parts = []
        seen_content = set()
        token_count = 0

        for idx, retrieved_chunk in enumerate(chunks, 1):
            chunk = retrieved_chunk.chunk

            if len(chunk.content.strip()) < 80:
                logger.warning(
                    "Skipping short chunk | document=%s | chunk=%s",
                    chunk.document_id,
                    chunk.chunk_index,
                )
                continue

            # Skip duplicates
            content_hash = hash(chunk.content)
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)

            # Build metadata header
            metadata_parts = []
            if chunk.metadata.get("law_number") and chunk.metadata.get("law_year"):
                metadata_parts.append(
                    f"القانون رقم {chunk.metadata['law_number']} لسنة {chunk.metadata['law_year']}"
                )
            if chunk.article_number:
                metadata_parts.append(f"المادة {chunk.article_number}")
            if chunk.section_title:
                metadata_parts.append(chunk.section_title)

            # Format chunk with metadata
            if metadata_parts:
                header = " | ".join(metadata_parts)
                chunk_text = f"[{header}]\n{chunk.content}"
            else:
                chunk_text = chunk.content

            estimated_tokens = self.estimate_tokens(chunk_text)
            if token_count + estimated_tokens > self.max_tokens:
                logger.info(f"Context limit reached at chunk {idx}")
                break

            context_parts.append(f"--- نص {idx} ---\n{chunk_text}")
            token_count += estimated_tokens

        if not context_parts:
            return "لا تتوفر نصوص قانونية ذات صلة."

        return "\n".join(context_parts)