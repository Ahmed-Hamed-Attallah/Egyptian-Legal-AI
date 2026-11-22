"""
High-level RAG orchestrator.
"""
import logging
from typing import Dict, List, Optional

from src.rag.pipeline import RAGPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranking_pipeline import RerankingPipeline
from src.generation.answer_generator import AnswerGenerator
from src.llm.provider_router import ProviderRouter
from src.config.settings import settings

logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    High-level orchestrator for RAG operations.
    Provides a unified interface for question answering.
    """

    def __init__(self, rag_pipeline: RAGPipeline = None):
        if rag_pipeline:
            self.pipeline = rag_pipeline
        else:
            # Initialize with default components
            # In production, these would be injected
            raise ValueError("RAGPipeline must be provided. Use dependency injection.")

    def ask(
        self,
        question: str,
        top_k: int = None,
        history: Optional[List[Dict[str, str]]] = None,
        memory_digest: Optional[List[Dict[str, object]]] = None,
    ) -> dict:
        """
        Answer a legal question.
        
        Args:
            question: The user's question
            top_k: Number of chunks to retrieve
            history: Short-term memory (prior turns of current conversation)
            memory_digest: Long-term memory (compact digest of other conversations)
        """
        logger.info(f"Processing question: {question[:100]}...")
        try:
            result = self.pipeline.process(
                question,
                top_k=top_k,
                history=history,
                memory_digest=memory_digest,
            )
            logger.info(
                f"Question answered successfully. "
                f"Provider: {result.get('provider')}, "
                f"Tokens: {result.get('tokens_used')}"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise

    def ask_batch(self, questions: list, top_k: int = None) -> list:
        """
        Answer multiple questions.
        """
        logger.info(f"Processing batch of {len(questions)} questions")
        return self.pipeline.process_batch(questions, top_k)