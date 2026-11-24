"""
Complete RAG pipeline implementation.

Retrieval ownership
-------------------
When a `RetrievalPipeline` is injected it becomes the retrieval entry point, so
its `ensure_fresh()` runs before every query and newly ingested chunks become
visible to BM25 without a restart. `rerank=False` is passed because this class
owns the reranking stage (and its stage timers) itself.

Falling back to `hybrid_retriever` keeps older callers working, but loses
automatic index freshness — the factory should always inject RetrievalPipeline.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional

from src.config.settings import settings
from src.domain.exceptions import LLMError, RetrievalError
from src.domain.queries import RetrievedChunk
from src.generation.answer_generator import AnswerGenerator
from src.generation.context_builder import ContextBuilder
from src.generation.response_validator import ResponseValidator
from src.monitoring.pipeline_monitor import stage_timer
from src.rag.citations import CitationResolver
from src.rag.confidence_scorer import ConfidenceScorer
from src.rag.query_classifier import QueryClassifier, QueryIntent
from src.rag.query_decomposer import QueryDecomposer
from src.reranking.reranking_pipeline import RerankingPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.metadata_filters import MetadataFilter
from src.retrieval.retrieval_pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / 2.5)


# Refusal phrases the generator uses when context is insufficient. If the
# answer contains one of these, citations must be suppressed — a refusal is
# not source-grounded, and showing sources would mislead (ARCHITECTURE §7).
_REFUSAL_MARKERS = (
    "لا تتوفر معلومات كافية",
    "لا توجد معلومات كافية",
    "غير متوفرة في المصادر",
    "لا تتوفر معلومات",
    "لا يمكنني الإجابة",
    "لا استطيع الإجابة",
    "insufficient information",
)


def _is_refusal(answer: str) -> bool:
    if not answer:
        return False
    return any(marker in answer for marker in _REFUSAL_MARKERS)


class RAGPipeline:
    """Classification → Retrieval → Reranking → Generation → Validation."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranking_pipeline: RerankingPipeline,
        answer_generator: AnswerGenerator,
        query_decomposer: Optional[QueryDecomposer] = None,
        context_builder: Optional[ContextBuilder] = None,
        response_validator: Optional[ResponseValidator] = None,
        retrieval_pipeline: Optional[RetrievalPipeline] = None,
    ) -> None:
        self.hybrid_retriever = hybrid_retriever
        self.reranking_pipeline = reranking_pipeline
        self.answer_generator = answer_generator
        self.query_decomposer = query_decomposer
        self.retrieval_pipeline = retrieval_pipeline

        self.context_builder = context_builder or ContextBuilder()
        self.response_validator = response_validator or ResponseValidator()
        self.citation_resolver = CitationResolver()
        self.query_classifier = QueryClassifier()
        self.confidence_scorer = ConfidenceScorer()

        # Detect once whether the generator supports the conversational mode,
        # so a signature mismatch degrades gracefully instead of raising.
        self._generator_supports_conversational = self._detect_conversational_support()

        if retrieval_pipeline is None:
            logger.warning(
                "RAGPipeline built without RetrievalPipeline: BM25 will NOT "
                "auto-refresh after ingestion. Inject it from the service factory."
            )

    @staticmethod
    def _detect_conversational_support() -> bool:
        try:
            params = inspect.signature(AnswerGenerator.generate).parameters
        except (TypeError, ValueError):
            return False
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return True
        return "conversational" in params

    # ------------------------------------------------------------------ helpers
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

    def _retrieve(
        self,
        query: str,
        filters: Optional[MetadataFilter],
        top_k: Optional[int],
    ) -> List[RetrievedChunk]:
        """Single retrieval entry point.

        Prefers RetrievalPipeline so `ensure_fresh()` runs; reranking is done by
        this class, so `rerank=False` avoids scoring twice.
        """
        if self.retrieval_pipeline is not None:
            return self.retrieval_pipeline.retrieve(
                query=query, filters=filters, top_k=top_k, rerank=False
            )
        return self.hybrid_retriever.retrieve(
            query=query, filters=filters, top_k=top_k
        )

    def _process_conversational(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]],
        memory_digest: Optional[List[Dict[str, object]]],
    ) -> dict:
        """Answer from conversation history/memory only — no retrieval, no citations."""
        logger.info("Conversational query detected; skipping legal retrieval.")

        with stage_timer("1. CONVERSATIONAL (NO RETRIEVAL)"):
            kwargs: Dict[str, Any] = {
                "context": "",
                "history": history,
                "memory_digest": memory_digest,
            }
            if self._generator_supports_conversational:
                kwargs["conversational"] = True
            else:
                logger.warning(
                    "AnswerGenerator.generate() has no `conversational` parameter; "
                    "answering with empty context instead."
                )
            response = self.answer_generator.generate(question, **kwargs)

        return {
            "answer": response.content,
            "citations": [],
            "retrieved_chunks": [],
            "confidence": 0.0,
            "confidence_label": self.confidence_scorer.get_label(0.0),
            "validation": {"status": "conversational", "valid": True},
            "provider": response.provider,
            "model": response.model,
            "tokens_used": response.tokens_used,
            "latency_ms": response.latency_ms,
        }

    # -------------------------------------------------------------------- main
    def process(
        self,
        question: str,
        top_k: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
        memory_digest: Optional[List[Dict[str, object]]] = None,
        filters: Optional[MetadataFilter] = None,
    ) -> dict:
        try:
            logger.info("Starting retrieval for question: %s...", question[:50])

            # Step 0: Query Classification
            intent = self.query_classifier.classify(question)

            # Conversational / meta questions are answered from history and
            # memory only — never from legal retrieval (ARCHITECTURE §9).
            if intent == QueryIntent.CONVERSATIONAL:
                return self._process_conversational(question, history, memory_digest)

            # Step 1: Hybrid Retrieval (with optional decomposition)
            with stage_timer("1. HYBRID RETRIEVAL"):
                if intent == QueryIntent.MULTI_CONCEPT and self.query_decomposer:
                    sub_queries = self.query_decomposer.decompose(question)
                    all_chunks: List[RetrievedChunk] = []
                    for sq in sub_queries:
                        all_chunks.extend(self._retrieve(sq, filters, top_k))
                    retrieved_chunks = self._deduplicate_chunks(all_chunks)
                    logger.info(
                        "Multi-concept retrieval: %d unique chunks from %d sub-queries.",
                        len(retrieved_chunks), len(sub_queries),
                    )
                else:
                    retrieved_chunks = self._retrieve(question, filters, top_k)

            if not retrieved_chunks:
                logger.warning("No chunks retrieved")
                return self._no_evidence_response()

            # Step 2: Reranking (enforces MIN_RERANK_SCORE internally)
            with stage_timer("2. RERANKING"):
                reranked_chunks = self.reranking_pipeline.process(
                    question, retrieved_chunks, top_k=top_k
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

            # If the model refused for lack of supporting context, suppress
            # citations so a refusal is never shown as if it were sourced.
            if _is_refusal(response.content):
                logger.info("Refusal detected; suppressing %d citations.", len(citations))
                citations = []
                reranked_chunks = []
                confidence = 0.0
                confidence_label = self.confidence_scorer.get_label(0.0)

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

        except RetrievalError as exc:
            logger.exception("Retrieval failed: %s", exc)
            raise
        except LLMError as exc:
            logger.exception("Generation failed: %s", exc)
            raise
        except Exception as exc:
            logger.exception("RAG pipeline failed: %s", exc)
            raise

    def _deduplicate_chunks(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        seen = set()
        unique: List[RetrievedChunk] = []
        for c in chunks:
            key = f"{c.chunk.document_id}_{c.chunk.chunk_index}"
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique

    def process_batch(self, questions: List[str], top_k: Optional[int] = None) -> List[dict]:
        results: List[dict] = []
        for question in questions:
            try:
                results.append(self.process(question, top_k))
            except Exception as exc:
                logger.error("Failed to process question: %s", exc)
                results.append({"error": str(exc), "question": question})
        return results