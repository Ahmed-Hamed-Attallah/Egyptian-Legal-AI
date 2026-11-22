"""
Ingestion service — coordinates the full document processing pipeline.

Orchestrates:
    File Validation → Document Store → DB Registration → Ingestion →
    Preprocessing → Chunking → Embedding → Vector Store → Status Update
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional

from src.chunking.pipeline import ChunkingPipeline
from src.domain.exceptions import (
    DocumentChunkingError,
    EmbeddingError,
    IngestionError,
    VectorDatabaseError,
)
from src.embeddings.model_registry import get_default_embedding_model
from src.ingestion.pipeline import IngestionPipeline
from src.preprocessing.pipeline import PreprocessingPipeline
from src.security.authentication import AuthContext
from src.security.authorization import Permission, require_permission
from src.services.document_service import DocumentService
from src.storage.database.database import Database
from src.storage.database.models.document import DocumentStatus
from src.storage.database.repositories.document_repository import DocumentRepository
from src.vector_store.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    """Coordinates end-to-end document ingestion."""

    def __init__(
        self,
        document_service: DocumentService,
        database: Database,
        ingestion_pipeline: Optional[IngestionPipeline] = None,
        preprocessing_pipeline: Optional[PreprocessingPipeline] = None,
        chunking_pipeline: Optional[ChunkingPipeline] = None,
        vector_store: Optional[QdrantVectorStore] = None,
    ):
        self._doc_service = document_service
        self._database = database
        self._ingestion = ingestion_pipeline or IngestionPipeline()
        self._preprocessing = preprocessing_pipeline or PreprocessingPipeline()
        self._chunking = chunking_pipeline or ChunkingPipeline()
        self._vector_store = vector_store

    def _get_vector_store(self) -> QdrantVectorStore:
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore()
        return self._vector_store

    def ingest_upload(
        self,
        file_content: BinaryIO,
        filename: str,
        size_bytes: int,
        auth: AuthContext,
    ) -> str:
        """Full ingestion flow for a new upload. Returns the document_id."""
        require_permission(auth, Permission.DOCUMENT_WRITE)

        document = self._doc_service.register_upload(
            file_content=file_content,
            filename=filename,
            size_bytes=size_bytes,
            auth=auth,
        )
        document_id = document.id

        try:
            self._process_document(document_id, auth)
        except Exception as exc:
            logger.exception("Ingestion failed for document %s", document_id)
            self._doc_service.update_status(
                document_id, DocumentStatus.FAILED, auth,
                error_message=str(exc),
            )
            raise

        return document_id

    def reprocess_document(self, document_id: str, auth: AuthContext) -> None:
        """Reprocess an existing document."""
        require_permission(auth, Permission.DOCUMENT_WRITE)
        doc = self._doc_service.get_document(document_id, auth)
        if doc is None:
            raise IngestionError(f"Document not found: {document_id}")
        self._process_document(document_id, auth)

    def _process_document(self, document_id: str, auth: AuthContext) -> None:
        """Run the full processing pipeline for a registered document."""
        self._doc_service.update_status(document_id, DocumentStatus.PROCESSING, auth)

        doc = self._doc_service.get_document(document_id, auth)
        if doc is None:
            raise IngestionError(f"Document disappeared during processing: {document_id}")

        # Load the file bytes
        file_stream = self._doc_service.load_file(document_id, auth)
        try:
            file_bytes = file_stream.read()
        finally:
            file_stream.close()

        # Phase 3: Ingestion — write bytes to a temp file so the existing
        # IngestionPipeline can handle format detection via extension.
        suffix = Path(doc.original_filename).suffix or ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            parsed_results = self._ingestion.ingest_file(tmp_path)
        except Exception as exc:
            raise IngestionError(f"Ingestion failed: {exc}") from exc
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        if not parsed_results:
            raise IngestionError("Ingestion produced no content")

        all_chunks = []
        embedding_model = None
        last_preprocessed = None

        for idx, parsed in enumerate(parsed_results):
            doc_id_for_processing = (
                f"{document_id}:{idx}" if len(parsed_results) > 1 else document_id
            )

            # Phase 4: Preprocessing
            preprocessed = self._preprocessing.process(
                text=parsed.content,
                doc_id=doc_id_for_processing,
                source_path=parsed.source_path or "",
            )
            last_preprocessed = preprocessed

            # Phase 5: Chunking
            chunks = self._chunking.process(preprocessed)
            if not chunks:
                logger.warning("No chunks produced for %s", doc_id_for_processing)
                continue

            # Phase 6: Embeddings (lazy init)
            if embedding_model is None:
                embedding_model = get_default_embedding_model()

            texts = [c.content for c in chunks]
            try:
                embeddings = embedding_model.encode_documents(texts)
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(f"Embedding failed: {exc}") from exc

            # Phase 7: Vector storage
            try:
                self._get_vector_store().insert(chunks, embeddings)
            except VectorDatabaseError:
                raise
            except Exception as exc:
                raise VectorDatabaseError(f"Vector store insert failed: {exc}") from exc

            all_chunks.extend(chunks)

        # Update DB with extracted metadata and final status
        with self._database.session() as session:
            repo = DocumentRepository(session)
            if last_preprocessed is not None:
                repo.update_metadata(
                    document_id, auth.tenant_id,
                    law_number=last_preprocessed.metadata.law_number,
                    law_year=last_preprocessed.metadata.law_year,
                    title=last_preprocessed.metadata.title,
                )
            repo.update_status(
                document_id, auth.tenant_id, DocumentStatus.COMPLETED,
                chunk_count=len(all_chunks),
            )
            session.commit()

        logger.info("Ingestion complete: document=%s chunks=%d", document_id, len(all_chunks))
