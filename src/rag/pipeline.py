"""
Complete RAG pipeline implementation.
"""
import logging
from typing import Dict, List, Optional

from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranking_pipeline import RerankingPipeline
from src.generation.context_builder import ContextBuilder
from src.generation.answer_generator import AnswerGenerator
from src.generation.response_validator import ResponseValidator
from src.rag.citations import CitationResolver
from src.rag.query_decomposer import QueryDecomposer
from src.rag.query_classifier import QueryClassifier, QueryIntent
from src.rag.confidence_scorer import ConfidenceScorer
from src.domain.queries import RetrievedChunk
from src.domain.responses import GenerationResponse
from src.domain.citations import Citation
from src.domain.exceptions import RetrievalError, LLMError
from src.config.settings import settings
from src.monitoring.pipeline_monitor import stage_timer

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / 2.5)


class RAGPipeline:
    """
    Complete RAG pipeline: Classification → Retrieval → Reranking → Generation → Validation.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranking_pipeline: RerankingPipeline,
        answer_generator: AnswerGenerator,
        query_decomposer: Optional[QueryDecomposer] = None,
        context_builder: ContextBuilder = None,
        response_validator: ResponseValidator = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranking_pipeline = reranking_pipeline
        self.answer_generator = answer_generator
        self.query_decomposer = query_decomposer
        self.context_builder = context_builder or ContextBuilder()
        self.response_validator = response_validator or ResponseValidator()
        self.citation_resolver = CitationResolver()
        self.query_classifier = QueryClassifier()
        self.confidence_scorer = ConfidenceScorer()

    @staticmethod
    def _no_evidence_response() -> dict:
        return {
            "answer": "لا تتوفر معلومات كافية في المصادر القانونية المتاحة للإجابة على هذا السؤال بدقة.",
            "citations": [],
            "retrieved_chunks": [],
            "confidence": 0.0,
            "confidence_label": "low",
            "validation": {"status": "no_evidence", "valid": True},
        }

    def process(
        self,
        question: str,
        top_k: int = None,
        history: Optional[List[Dict[str, str]]] = None,
        memory_digest: Optional[List[Dict[str, object]]] = None,
    ) -> dict:
        try:
            logger.info(f"Starting retrieval for question: {question[:50]}...")

            # Step 0: Query Classification
            intent = self.query_classifier.classify(question)

            # Step 1: Hybrid Retrieval (with optional decomposition)
            with stage_timer("1. HYBRID RETRIEVAL"):
                if intent == QueryIntent.MULTI_CONCEPT and self.query_decomposer:
                    sub_queries = self.query_decomposer.decompose(question)
                    all_chunks = []
                    for sq in sub_queries:
                        chunks = self.hybrid_retriever.retrieve(query=sq)
                        all_chunks.extend(chunks)
                    retrieved_chunks = self._deduplicate_chunks(all_chunks)
                    logger.info(f"Multi-concept retrieval: {len(retrieved_chunks)} unique chunks from {len(sub_queries)} sub-queries.")
                else:
                    retrieved_chunks = self.hybrid_retriever.retrieve(query=question)

            if not retrieved_chunks:
                logger.warning("No chunks retrieved")
                return self._no_evidence_response()

            # Step 2: Reranking (enforces MIN_RERANK_SCORE internally)
            with stage_timer("2. RERANKING"):
                reranked_chunks = self.reranking_pipeline.process(
                    question,
                    retrieved_chunks,
                    top_k=top_k,
                )

            if not reranked_chunks:
                logger.warning("Reranking rejected all retrieved chunks.")
                return self._no_evidence_response()

            # Step 2.5: Context truncation
            max_prompt_tokens = settings.CONTEXT_MAX_TOKENS
            reserved_tokens = 200
            truncated_chunks: List[RetrievedChunk] = []
            current_tokens = estimate_tokens(question) + reserved_tokens

            for chunk in reranked_chunks:
                chunk_tokens = estimate_tokens(chunk.chunk.content)
                if current_tokens + chunk_tokens <= max_prompt_tokens:
                    truncated_chunks.append(chunk)
                    current_tokens += chunk_tokens
                else:
                    break

            if not truncated_chunks and reranked_chunks:
                truncated_chunks = [reranked_chunks[0]]
            reranked_chunks = truncated_chunks

            # Step 3: Context Building
            with stage_timer("3. Building Context"):
                context = self.context_builder.build_context(reranked_chunks)

            # Step 4: Generation
            with stage_timer("4. Generating Answer"):
                response = self.answer_generator.generate(
                    question, context, history=history, memory_digest=memory_digest
                )

            # Step 5: Citation Resolution
            with stage_timer("5. Resolving Citations"):
                citations = self.citation_resolver.resolve_citations(reranked_chunks)

            # Step 6: Validation & Confidence
            with stage_timer("6. RESPONSE VALIDATION"):
                validation = self.response_validator.validate_response(
                    response, citations, has_context=True
                )
                confidence = self.confidence_scorer.calculate(reranked_chunks)
                confidence_label = self.confidence_scorer.get_label(confidence)

            return {
                "answer": response.content,
                "citations": citations,
                "retrieved_chunks": reranked_chunks,
                "confidence": confidence,
                "confidence_label": confidence_label,
                "validation": validation,
                "provider": response.provider,
                "model": response.model,
                "tokens_used": response.tokens_used,
                "latency_ms": response.latency_ms,
            }

        except RetrievalError as e:
            logger.exception(f"Retrieval failed: {e}")
            raise
        except LLMError as e:
            logger.exception(f"Generation failed: {e}")
            raise
        except Exception as e:
            logger.exception(f"RAG pipeline failed: {e}")
            raise

    def _deduplicate_chunks(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        seen = set()
        unique = []
        for c in chunks:
            key = f"{c.chunk.document_id}_{c.chunk.chunk_index}"
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def process_batch(self, questions: List[str], top_k: int = None) -> List[dict]:
        results = []
        for question in questions:
            try:
                result = self.process(question, top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process question: {e}")
                results.append({"error": str(e), "question": question})
        return results