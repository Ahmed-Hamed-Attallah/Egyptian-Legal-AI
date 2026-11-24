# src/services/factory.py
"""
Central application service factory.

Single place where core application services are constructed. Used by:
- api/lifespan.py
- workers/ingestion_worker.py
- src/evaluation/evaluation_runner.py

Retrieval index lifecycle is delegated to RetrievalPipeline so the factory no
longer inlines Qdrant scrolling / BM25 construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ApplicationServices:
    conversation_service: Optional[Any] = None
    chat_service: Optional[Any] = None
    search_service: Optional[Any] = None
    document_service: Optional[Any] = None
    admin_service: Optional[Any] = None
    job_manager: Optional[Any] = None
    provider_router: Optional[Any] = None
    retrieval_pipeline: Optional[Any] = None
    startup_errors: List[str] = field(default_factory=list)


def _build_conversation_service() -> Optional[Any]:
    try:
        from src.services.conversation_service import ConversationService
        from src.storage.database.database import get_database

        return ConversationService(database=get_database())
    except Exception as exc:
        logger.error("Conversation service initialization failed: %s", exc)
        return None


def _build_rag_orchestrator() -> tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Construct the RAG orchestrator with full dependency injection.

    Returns ``(orchestrator, provider_router, retrieval_pipeline)``. If Qdrant or
    any core component is unreachable, all are None so dependent services stay
    unavailable and the failure is recorded.
    """
    try:
        from src.rag.orchestrator import RAGOrchestrator
        from src.rag.pipeline import RAGPipeline
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.sparse_retriever import SparseRetriever
        from src.retrieval.score_fusion import ReciprocalRankFusion
        from src.retrieval.retrieval_pipeline import RetrievalPipeline
        from src.reranking.reranking_pipeline import RerankingPipeline
        from src.reranking.cross_encoder_reranker import CrossEncoderReranker
        from src.generation.answer_generator import AnswerGenerator
        from src.llm.provider_router import ProviderRouter
        from src.vector_store.qdrant_store import QdrantVectorStore
        from src.embeddings.model_registry import get_default_embedding_model

        embedding_model = get_default_embedding_model()
        vector_store = QdrantVectorStore()

        dense_retriever = DenseRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
        )
        sparse_retriever = SparseRetriever()
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=ReciprocalRankFusion(),
        )

        reranking_pipeline = None
        if getattr(settings, "RERANKER_ENABLED", True):
            reranking_pipeline = RerankingPipeline(reranker=CrossEncoderReranker())
        else:
            logger.info("Reranking disabled by configuration (RERANKER_ENABLED=false).")

        # Owns BM25 build/refresh. Replaces the old inline scroll_all()+build_index().
        retrieval_pipeline = RetrievalPipeline(
            vector_store=vector_store,
            sparse_retriever=sparse_retriever,
            hybrid_retriever=hybrid_retriever,
            reranking_pipeline=reranking_pipeline,
        )
        if getattr(settings, "SPARSE_RETRIEVAL_ENABLED", True):
            try:
                retrieval_pipeline.build_index()
            except Exception as exc:
                logger.warning(
                    "Could not build BM25 index from Qdrant: %s. "
                    "Sparse retrieval will be skipped until the next refresh.",
                    exc,
                )

        provider_router = ProviderRouter()

        from src.rag.query_decomposer import QueryDecomposer

        answer_generator = AnswerGenerator(provider_router=provider_router)
        query_decomposer = QueryDecomposer(provider_router=provider_router)

        pipeline = RAGPipeline(
            hybrid_retriever=hybrid_retriever,
            reranking_pipeline=reranking_pipeline or RerankingPipeline(
                reranker=CrossEncoderReranker()
            ),
            answer_generator=answer_generator,
            query_decomposer=query_decomposer,
            retrieval_pipeline=retrieval_pipeline,
        )

        return RAGOrchestrator(rag_pipeline=pipeline), provider_router, retrieval_pipeline

    except Exception as exc:
        logger.error("RAG orchestrator initialization failed: %s", exc)
        return None, None, None


def _build_chat_service(
    orchestrator: Optional[Any], conversation_service: Optional[Any]
) -> Optional[Any]:
    if orchestrator is None:
        return None
    try:
        from src.services.chat_service import ChatService

        return ChatService(
            orchestrator=orchestrator,
            conversation_service=conversation_service,
        )
    except Exception as exc:
        logger.error("Chat service initialization failed: %s", exc)
        return None


def _build_search_service(orchestrator: Optional[Any]) -> Optional[Any]:
    if orchestrator is None:
        return None
    try:
        from src.services.search_service import SearchService

        pipeline = getattr(orchestrator, "pipeline", None) or getattr(
            orchestrator, "rag_pipeline", None
        )
        hybrid_retriever = getattr(pipeline, "hybrid_retriever", None) if pipeline else None
        reranker = getattr(pipeline, "reranking_pipeline", None) if pipeline else None

        if hybrid_retriever is None:
            logger.warning("Search service could not extract hybrid retriever from orchestrator.")
            return None

        return SearchService(hybrid_retriever=hybrid_retriever, reranker=reranker)
    except Exception as exc:
        logger.error("Search service initialization failed: %s", exc)
        return None


def _build_document_service() -> Optional[Any]:
    try:
        from src.services.document_service import DocumentService
        from src.storage.database.database import get_database

        return DocumentService(database=get_database(), vector_store=None)
    except Exception as exc:
        logger.error("Document service initialization failed: %s", exc)
        return None


def _build_admin_service(
    chat_service: Optional[Any],
    search_service: Optional[Any],
    document_service: Optional[Any],
    provider_router: Optional[Any],
    database: Optional[Any],
) -> Optional[Any]:
    try:
        from src.services.admin_service import AdminService

        return AdminService(
            chat_service=chat_service,
            search_service=search_service,
            document_service=document_service,
            provider_router=provider_router,
            database=database,
        )
    except Exception as exc:
        logger.error("Admin service initialization failed: %s", exc)
        return None


def _build_job_manager(document_service: Optional[Any]) -> Optional[Any]:
    if not settings.WORKER_ENABLED:
        logger.info("Background worker support disabled by settings.")
        return None
    try:
        from src.workers.job_manager import JobManager
        from src.workers.job_store import create_job_store

        store = create_job_store(settings.WORKER_JOB_STORE)
        return JobManager(store=store, document_service=document_service)
    except Exception as exc:
        logger.error("Job manager initialization failed: %s", exc)
        return None


def build_application_services() -> ApplicationServices:
    services = ApplicationServices()

    services.conversation_service = _build_conversation_service()
    if services.conversation_service is None:
        services.startup_errors.append("conversation_service")

    orchestrator, provider_router, retrieval_pipeline = _build_rag_orchestrator()
    services.provider_router = provider_router
    services.retrieval_pipeline = retrieval_pipeline
    if orchestrator is None:
        services.startup_errors.append("rag_orchestrator")

    services.chat_service = _build_chat_service(orchestrator, services.conversation_service)
    if services.chat_service is None:
        services.startup_errors.append("chat_service")

    services.search_service = _build_search_service(orchestrator)
    if services.search_service is None:
        services.startup_errors.append("search_service")

    services.document_service = _build_document_service()
    if services.document_service is None:
        services.startup_errors.append("document_service")

    services.job_manager = _build_job_manager(services.document_service)
    if settings.WORKER_ENABLED and services.job_manager is None:
        services.startup_errors.append("job_manager")

    db = None
    try:
        from src.storage.database.database import get_database

        db = get_database()
    except Exception as exc:
        logger.error("Database initialization failed: %s", exc)
        services.startup_errors.append("database")

    services.admin_service = _build_admin_service(
        chat_service=services.chat_service,
        search_service=services.search_service,
        document_service=services.document_service,
        provider_router=services.provider_router,
        database=db,
    )
    if services.admin_service is None:
        services.startup_errors.append("admin_service")

    return services