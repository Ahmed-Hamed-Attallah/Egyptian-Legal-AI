"""
Answer generator using LLM providers.
"""
import logging
from typing import Dict, List, Optional

from src.llm.provider_router import ProviderRouter
from src.domain.responses import GenerationRequest, GenerationResponse
from src.domain.exceptions import LLMError
from src.generation.prompt_templates import LegalPromptTemplates
from src.config.settings import settings

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Generates legal answers using the provider router.
    
    Supports conversation memory:
    - Short-term: history (prior turns of current conversation)
    - Long-term: memory_digest (compact digest of other conversations)
    """

    def __init__(self, provider_router: ProviderRouter, language: str = "ar"):
        self.provider_router = provider_router
        self.language = language

    def generate(
        self,
        question: str,
        context: str,
        temperature: float = None,
        max_tokens: int = None,
        history: Optional[List[Dict[str, str]]] = None,
        memory_digest: Optional[List[Dict[str, object]]] = None,
    ) -> GenerationResponse:
        """
        Generate an answer for a legal question with conversation memory.
        
        Args:
            question: The user's question
            context: Retrieved legal context (chunks)
            temperature: LLM temperature (optional)
            max_tokens: Max tokens to generate (optional)
            history: Short-term memory (prior turns of current conversation)
            memory_digest: Long-term memory (compact digest of other conversations)
        """
        try:
            # Build system prompt with memory digest (if any)
            system_prompt = LegalPromptTemplates.build_system_prompt_with_memory(
                memory_digest or [],
                language=self.language,
            )

            # Build user prompt with question and context
            user_prompt = LegalPromptTemplates.format_user_prompt(
                question, context, self.language
            )

            # Construct the full messages list for the LLM
            # This is the key: history becomes real turns, not just context
            messages = [
                {"role": "system", "content": system_prompt},
            ]
            if history:
                messages.extend(history)  # Prior user/assistant turns
            messages.append(
                {"role": "user", "content": user_prompt}
            )

            request = GenerationRequest(
                system_prompt=system_prompt,  # Kept for backward compatibility
                user_prompt=user_prompt,
                messages=messages,  # New field: full conversation
                temperature=temperature or settings.LLM_TEMPERATURE,
                max_tokens=max_tokens or settings.LLM_MAX_TOKENS,
            )

            response = self.provider_router.generate(request)
            logger.info(f"Answer generated successfully with {response.provider}")
            return response

        except LLMError as e:
            logger.error(f"Answer generation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in answer generation: {e}")
            raise LLMError(f"Generation failed: {e}")