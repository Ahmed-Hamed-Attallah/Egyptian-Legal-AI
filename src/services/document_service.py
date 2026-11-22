"""
Document ingestion service.

Coordinates file validation, storage, metadata persistence, parsing,
preprocessing, chunking, chunk persistence, embedding and Qdrant indexing.
It orchestrates existing components and does not reimplement their internals.

Status contract
---------------
The public status is derived from the persisted DocumentStatus, never from
chunk_count alone. PENDING/PROCESSING must stay visible so the client keeps
polling; only a terminal state may report completed/partial/failed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.chunking.pipeline import ChunkingPipeline
from src.config.settings import settings
from src.domain.exceptions import IngestionError, UnsupportedFormatError
from src.embeddings.model_registry import get_default_embedding_model
from src.ingestion.pipeline import IngestionPipeline
from src.preprocessing.pipeline import PreprocessingPipeline
from src.security.authentication import AuthContext
from src.storage.database.database import Database
from src.storage.database.models.chunk import DocumentChunkRecord
from src.storage.database.models.document import DocumentRecord, DocumentStatus
from src.storage.database.repositories.chunk_repository import ChunkRepository
from src.storage.database.repositories.document_repository import DocumentRepository

logger = logging.getLogger(__name__)

# Default human-readable messages per public status.
_STATUS_MESSAGES: Dict[str, str] = {
    "uploaded": "Document queued for processing.",
    "processing": "Document is being parsed, chunked and indexed.",
    "completed": "Document ingested and indexed.",
    "partial": (
        "Document parsed and chunked, but vector indexing was unavailable. "
        "Chunks are stored in PostgreSQL and can be inspected."
    ),
    "failed": "Document processing failed.",
    "unknown": "Document state is unknown.",
}


class DocumentService:
    """Document ingestion and chunk inspection service."""

    def __init__(
        self,
        database: Optional[Database] = None,
        vector_store: Optional[Any] = None,
    ) -> None:
        self._database = database
        self._vector_store = vector_store
        self._vector_store_checked = vector_store is not None
        self._ingestion: Optional[IngestionPipeline] = None
        self._preprocessing: Optional[PreprocessingPipeline] = None
        self._chunking: Optional[ChunkingPipeline] = None
        self._embedding_model: Optional[Any] = None
        # Fallback only when database is unavailable.
        self._records: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def ingest_upload(
        self,
        file_bytes: bytes,
        filename: str,
        tenant_id: Optional[str] = None,
        auth: Optional[AuthContext] = None,
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest an uploaded document.

        Supports both call styles:
            ingest_upload(file_bytes, filename, tenant_id=...)
            ingest_upload(file_bytes, filename, auth=auth_context)
        """
        if auth is not None:
            tenant_id = auth.tenant_id
            owner_id = auth.user_id
        else:
            tenant_id = tenant_id or getattr(settings, "DEFAULT_TENANT_ID", "default")
            owner_id = None

        if not file_bytes:
            raise IngestionError("Uploaded file is empty.")

        if len(file_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise IngestionError(
                f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES} bytes."
            )

        suffix = Path(filename).suffix.lower()
        allowed = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in settings.ALLOWED_FILE_TYPES
        }
        if suffix and suffix not in allowed:
            raise UnsupportedFormatError(
                f"Unsupported file type: {suffix}. Allowed types: {sorted(allowed)}"
            )

        document_id = document_id or str(uuid.uuid4())

        db_chunks: List[DocumentChunkRecord] = []
        domain_chunks: List[Any] = []
        chunk_count = 0
        indexed = False
        document_metadata: Dict[str, Any] = {}
        storage_key = ""

        try:
            # Storage happens INSIDE the guarded block: if it fails we still
            # persist a FAILED record, so a job's document_id never dangles.
            stored_path, storage_key = self._store_original(
                file_bytes=file_bytes,
                tenant_id=tenant_id,
                document_id=document_id,
                suffix=suffix,
            )

            self._create_or_update_document_record(
                document_id=document_id,
                tenant_id=tenant_id,
                owner_id=owner_id,
                filename=filename,
                storage_key=storage_key,
                file_size_bytes=len(file_bytes),
                status=DocumentStatus.PROCESSING,
                error_message=None,
                chunk_count=0,
                indexed=False,
            )

            self._ensure_components()
            parsed_results = self._ingestion.ingest_file(stored_path)

            if not parsed_results:
                message = "Ingestion produced no content."
                self._finalize_document(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    status=DocumentStatus.FAILED,
                    chunk_count=0,
                    indexed=False,
                    error_message=message,
                    document_metadata={},
                )
                return self._memory_or_db_record(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    filename=filename,
                    status="failed",
                    chunk_count=0,
                    message=message,
                    indexed=False,
                )

            for idx, parsed in enumerate(parsed_results):
                doc_id_for_processing = (
                    f"{document_id}:{idx}" if len(parsed_results) > 1 else document_id
                )
                preprocessed = self._preprocessing.process(
                    text=parsed.content,
                    doc_id=doc_id_for_processing,
                    source_path=str(stored_path),
                )

                metadata = getattr(preprocessed, "metadata", None)
                if metadata is not None:
                    document_metadata.update(
                        {
                            "law_number": getattr(metadata, "law_number", None),
                            "law_year": getattr(metadata, "law_year", None),
                            "title": getattr(metadata, "title", None),
                        }
                    )
                    document_metadata = {
                        k: v for k, v in document_metadata.items() if v is not None
                    }

                chunks = self._chunking.process(preprocessed)
                if not chunks:
                    logger.warning("No chunks produced for %s", doc_id_for_processing)
                    continue

                for chunk in chunks:
                    domain_chunks.append(chunk)
                    if self._database is not None:
                        db_chunks.append(
                            self._build_chunk_record(
                                chunk=chunk,
                                document_id=document_id,
                                tenant_id=tenant_id,
                                chunk_index=len(db_chunks),
                                document_metadata=document_metadata,
                                storage_key=storage_key,
                            )
                        )

            chunk_count = len(domain_chunks)

            if self._database is not None and db_chunks:
                with self._database.session() as session:
                    chunk_repo = ChunkRepository(session)
                    chunk_repo.delete_by_document(document_id, tenant_id)
                    chunk_repo.create_many(db_chunks)

            # Attach metadata needed for Qdrant payload and deterministic point IDs.
            for i, chunk in enumerate(domain_chunks):
                db_chunk = db_chunks[i] if i < len(db_chunks) else None
                self._attach_chunk_metadata(
                    chunk=chunk,
                    db_chunk=db_chunk,
                    document_id=document_id,
                    tenant_id=tenant_id,
                    storage_key=storage_key,
                    document_metadata=document_metadata,
                    chunk_index=i,
                )

            indexed = self._embed_and_index_chunks(domain_chunks)

            if self._database is not None and db_chunks:
                self._update_chunk_embedding_results(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    db_chunks=db_chunks,
                    indexed=indexed,
                )

            if chunk_count == 0:
                status = DocumentStatus.FAILED
                public_status = "failed"
                message = "No chunks produced."
            elif indexed:
                status = DocumentStatus.COMPLETED
                public_status = "completed"
                message = _STATUS_MESSAGES["completed"]
            else:
                status = DocumentStatus.COMPLETED
                public_status = "partial"
                message = _STATUS_MESSAGES["partial"]

            self._finalize_document(
                document_id=document_id,
                tenant_id=tenant_id,
                status=status,
                chunk_count=chunk_count,
                indexed=indexed,
                error_message=None,
                document_metadata=document_metadata,
            )

            return self._memory_or_db_record(
                document_id=document_id,
                tenant_id=tenant_id,
                filename=filename,
                status=public_status,
                chunk_count=chunk_count,
                message=message,
                indexed=indexed,
            )

        except Exception as exc:
            logger.exception("Document ingestion failed for %s", document_id)
            self._ensure_failed_record(
                document_id=document_id,
                tenant_id=tenant_id,
                owner_id=owner_id,
                filename=filename,
                storage_key=storage_key,
                file_size_bytes=len(file_bytes),
                chunk_count=chunk_count,
                error_message=str(exc),
                document_metadata=document_metadata,
            )
            raise IngestionError(f"Document ingestion failed: {exc}") from exc

    def get_document(self, document_id: str, tenant_id: str) -> Dict[str, Any]:
        """Return document metadata."""
        if self._database is not None:
            with self._database.read_only_session() as session:
                repo = DocumentRepository(session)
                record = repo.get_by_id(document_id, tenant_id)
                if record is None:
                    raise LookupError("Document not found")
                return self._document_record_to_public(record)

        record = self._records.get(document_id)
        if record is None or record.get("tenant_id") != tenant_id:
            raise LookupError("Document not found")
        return dict(record)

    def list_documents(
        self,
        tenant_id: str,
        offset: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List documents for a tenant."""
        if self._database is not None:
            with self._database.read_only_session() as session:
                repo = DocumentRepository(session)
                records = repo.list_by_tenant(
                    tenant_id=tenant_id, offset=offset, limit=limit
                )
                return [self._document_record_to_public(r) for r in records]

        return [
            dict(record)
            for record in self._records.values()
            if record.get("tenant_id") == tenant_id
        ]

    def list_document_chunks(
        self,
        document_id: str,
        tenant_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Return persisted chunks for a document as (items, total)."""
        if self._database is None:
            return [], 0

        with self._database.read_only_session() as session:
            doc_repo = DocumentRepository(session)
            document = doc_repo.get_by_id(document_id, tenant_id)
            if document is None:
                raise LookupError("Document not found")

            chunk_repo = ChunkRepository(session)
            total = chunk_repo.count_by_document(document_id, tenant_id)
            records = chunk_repo.list_by_document(
                document_id=document_id,
                tenant_id=tenant_id,
                offset=offset,
                limit=limit,
            )
            return [self._chunk_record_to_public(r) for r in records], total

    # -------------------------------------------------------------------------
    # Component initialization
    # -------------------------------------------------------------------------
    def _ensure_components(self) -> None:
        if self._ingestion is None:
            self._ingestion = IngestionPipeline()
        if self._preprocessing is None:
            self._preprocessing = PreprocessingPipeline()
        if self._chunking is None:
            self._chunking = ChunkingPipeline()

    def _get_vector_store(self) -> Optional[Any]:
        """Lazily create the Qdrant vector store."""
        if self._vector_store_checked:
            return self._vector_store
        self._vector_store_checked = True
        try:
            from src.vector_store.qdrant_store import QdrantVectorStore

            self._vector_store = QdrantVectorStore()
            logger.info("Qdrant vector store initialized for document indexing.")
        except Exception as exc:
            logger.warning("Vector store unavailable: %s", exc)
            self._vector_store = None
        return self._vector_store

    # -------------------------------------------------------------------------
    # Storage helpers
    # -------------------------------------------------------------------------
    def _store_original(
        self,
        file_bytes: bytes,
        tenant_id: str,
        document_id: str,
        suffix: str,
    ) -> Tuple[Path, str]:
        safe_tenant = self._safe_component(tenant_id, default="default")
        base_dir = Path(settings.DOCUMENT_STORAGE_DIR) / safe_tenant / document_id
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"source{suffix or ''}"
        path.write_bytes(file_bytes)
        try:
            storage_key = str(path.relative_to(Path(settings.DOCUMENT_STORAGE_DIR)))
        except Exception:
            storage_key = str(path)
        return path, storage_key

    @staticmethod
    def _safe_component(value: str, default: str = "default") -> str:
        value = (value or default).strip()
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
        return safe[:128] or default

    # -------------------------------------------------------------------------
    # Database helpers
    # -------------------------------------------------------------------------
    def _create_or_update_document_record(
        self,
        document_id: str,
        tenant_id: str,
        owner_id: Optional[str],
        filename: str,
        storage_key: str,
        file_size_bytes: int,
        status: DocumentStatus,
        error_message: Optional[str],
        chunk_count: int,
        indexed: bool,
    ) -> None:
        if self._database is None:
            self._records[document_id] = {
                "document_id": document_id,
                "id": document_id,
                "tenant_id": tenant_id,
                "filename": filename,
                "status": self._public_status(status.value, chunk_count, indexed),
                "chunk_count": chunk_count,
                "message": error_message,
                "indexed": indexed,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            return

        with self._database.session() as session:
            repo = DocumentRepository(session)
            existing = repo.get_by_id(document_id, tenant_id, include_deleted=True)
            if existing is not None:
                repo.update_status(
                    document_id=document_id,
                    tenant_id=tenant_id,
                    status=status,
                    error_message=error_message,
                    chunk_count=chunk_count,
                    indexed=indexed,
                )
                return

            repo.create(
                DocumentRecord(
                    id=document_id,
                    tenant_id=tenant_id,
                    owner_id=owner_id,
                    original_filename=filename,
                    storage_key=storage_key,
                    file_size_bytes=file_size_bytes,
                    status=status,
                    error_message=error_message,
                    chunk_count=chunk_count,
                    indexed=indexed,
                )
            )

    def _finalize_document(
        self,
        document_id: str,
        tenant_id: str,
        status: DocumentStatus,
        chunk_count: int,
        indexed: bool,
        error_message: Optional[str],
        document_metadata: Dict[str, Any],
    ) -> None:
        if self._database is None:
            record = self._records.get(document_id, {})
            record.update(
                {
                    "status": self._public_status(status.value, chunk_count, indexed),
                    "chunk_count": chunk_count,
                    "indexed": indexed,
                    "message": error_message,
                }
            )
            self._records[document_id] = record
            return

        with self._database.session() as session:
            repo = DocumentRepository(session)
            repo.update_metadata(
                document_id=document_id,
                tenant_id=tenant_id,
                law_number=document_metadata.get("law_number"),
                law_year=document_metadata.get("law_year"),
                title=document_metadata.get("title"),
                legal_metadata=document_metadata or None,
            )
            repo.update_status(
                document_id=document_id,
                tenant_id=tenant_id,
                status=status,
                error_message=error_message,
                chunk_count=chunk_count,
                indexed=indexed,
            )

    def _ensure_failed_record(
        self,
        document_id: str,
        tenant_id: str,
        owner_id: Optional[str],
        filename: str,
        storage_key: str,
        file_size_bytes: int,
        chunk_count: int,
        error_message: str,
        document_metadata: Dict[str, Any],
    ) -> None:
        """Guarantee a FAILED record exists so job.document_id never dangles.

        A failure can occur before the first record was written (e.g. disk
        error in _store_original). Without this, the job keeps a document_id
        that resolves to 404 forever.
        """
        if self._database is None:
            record = self._records.get(document_id, {})
            record.update(
                {
                    "document_id": document_id,
                    "id": document_id,
                    "tenant_id": tenant_id,
                    "filename": filename,
                    "status": "failed",
                    "chunk_count": chunk_count,
                    "indexed": False,
                    "message": error_message,
                    "created_at": record.get("created_at")
                    or datetime.now(timezone.utc).isoformat(),
                }
            )
            self._records[document_id] = record
            return

        try:
            with self._database.session() as session:
                repo = DocumentRepository(session)
                existing = repo.get_by_id(document_id, tenant_id, include_deleted=True)
                if existing is None:
                    repo.create(
                        DocumentRecord(
                            id=document_id,
                            tenant_id=tenant_id,
                            owner_id=owner_id,
                            original_filename=filename,
                            # Deterministic fallback keeps the tenant/storage_key
                            # unique constraint satisfied.
                            storage_key=storage_key
                            or f"{self._safe_component(tenant_id)}/{document_id}/source",
                            file_size_bytes=file_size_bytes,
                            status=DocumentStatus.FAILED,
                            error_message=error_message,
                            chunk_count=chunk_count,
                            indexed=False,
                        )
                    )
                    return

            self._finalize_document(
                document_id=document_id,
                tenant_id=tenant_id,
                status=DocumentStatus.FAILED,
                chunk_count=chunk_count,
                indexed=False,
                error_message=error_message,
                document_metadata=document_metadata,
            )
        except Exception:
            logger.exception(
                "Could not persist FAILED state for document %s", document_id
            )

    def _build_chunk_record(
        self,
        chunk: Any,
        document_id: str,
        tenant_id: str,
        chunk_index: int,
        document_metadata: Dict[str, Any],
        storage_key: str,
    ) -> DocumentChunkRecord:
        metadata = getattr(chunk, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}

        content = str(getattr(chunk, "content", "") or "")

        chunk_type_value = getattr(chunk, "chunk_type", None)
        if chunk_type_value is not None:
            chunk_type = str(chunk_type_value)
        else:
            chunk_type = str(metadata.get("chunk_type") or "other")

        page_number = metadata.get("page_number")
        try:
            page_number = int(page_number) if page_number is not None else None
        except Exception:
            page_number = None

        law_name = (
            metadata.get("law_name")
            or document_metadata.get("title")
            or document_metadata.get("law_name")
        )
        law_number = metadata.get("law_number") or document_metadata.get("law_number")
        law_year = metadata.get("law_year") or document_metadata.get("law_year")
        article_number = getattr(chunk, "article_number", None) or metadata.get(
            "article_number"
        )
        section_title = getattr(chunk, "section_title", None) or metadata.get(
            "section_title"
        )

        safe_metadata = dict(metadata)
        safe_metadata.setdefault("storage_key", storage_key)

        return DocumentChunkRecord(
            id=str(uuid.uuid4()),
            document_id=document_id,
            tenant_id=tenant_id,
            chunk_index=chunk_index,
            content=content,
            chunk_type=chunk_type,
            page_number=page_number,
            section_title=section_title,
            law_name=law_name,
            law_number=law_number,
            law_year=law_year,
            article_number=article_number,
            embedding_status="pending",
            vector_id=None,
            metadata_json=safe_metadata,
        )

    def _attach_chunk_metadata(
        self,
        chunk: Any,
        db_chunk: Optional[DocumentChunkRecord],
        document_id: str,
        tenant_id: str,
        storage_key: str,
        document_metadata: Dict[str, Any],
        chunk_index: int,
    ) -> None:
        """Attach retrieval/filter/citation metadata to the domain chunk."""
        metadata = getattr(chunk, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        try:
            chunk.metadata = metadata
        except Exception:
            pass

        vector_id = self._generate_vector_id(document_id, chunk_index)
        chunk_id = db_chunk.id if db_chunk is not None else str(uuid.uuid4())

        metadata.update(
            {
                "tenant_id": tenant_id,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "vector_id": vector_id,
                "storage_key": storage_key,
            }
        )

        if db_chunk is not None:
            metadata.update(
                {
                    "law_name": db_chunk.law_name,
                    "law_number": db_chunk.law_number,
                    "law_year": db_chunk.law_year,
                    "article_number": db_chunk.article_number,
                    "section_title": db_chunk.section_title,
                    "page_number": db_chunk.page_number,
                    "chunk_type": db_chunk.chunk_type,
                }
            )
        else:
            metadata.update(
                {
                    "law_name": document_metadata.get("title"),
                    "law_number": document_metadata.get("law_number"),
                    "law_year": document_metadata.get("law_year"),
                }
            )

        chunk.metadata = {k: v for k, v in metadata.items() if v is not None}

        try:
            chunk.document_id = uuid.UUID(document_id)
        except Exception:
            pass

    def _generate_vector_id(self, document_id: str, chunk_index: int) -> str:
        """Deterministic Qdrant point ID — must match QdrantVectorStore."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{chunk_index}"))

    # -------------------------------------------------------------------------
    # Embedding / indexing helpers
    # -------------------------------------------------------------------------
    def _embed_and_index_chunks(self, chunks: List[Any]) -> bool:
        if not chunks:
            return False

        store = self._get_vector_store()
        if store is None:
            logger.warning("Cannot index chunks: vector store unavailable.")
            return False

        try:
            if self._embedding_model is None:
                self._embedding_model = get_default_embedding_model()

            texts = [str(getattr(chunk, "content", "") or "") for chunk in chunks]
            embeddings = self._embedding_model.encode_documents(texts)

            normalized_embeddings: List[List[float]] = [
                [float(x) for x in emb] for emb in embeddings
            ]

            result = store.insert(chunks, normalized_embeddings)
            if isinstance(result, list):
                logger.info("Indexed %d chunks into Qdrant.", len(result))
                return True
            logger.info("Indexed %d chunks into Qdrant.", len(chunks))
            return True
        except Exception as exc:
            logger.warning("Chunk embedding/indexing failed: %s", exc)
            return False

    def _update_chunk_embedding_results(
        self,
        document_id: str,
        tenant_id: str,
        db_chunks: List[DocumentChunkRecord],
        indexed: bool,
    ) -> None:
        if self._database is None or not db_chunks:
            return

        with self._database.session() as session:
            chunk_repo = ChunkRepository(session)
            for db_chunk in db_chunks:
                if indexed:
                    chunk_repo.update_embedding_result(
                        chunk_id=db_chunk.id,
                        tenant_id=tenant_id,
                        embedding_status="done",
                        vector_id=db_chunk.vector_id
                        or self._generate_vector_id(document_id, db_chunk.chunk_index),
                    )
                else:
                    chunk_repo.update_embedding_result(
                        chunk_id=db_chunk.id,
                        tenant_id=tenant_id,
                        embedding_status="failed",
                        vector_id=None,
                    )

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------
    def _memory_or_db_record(
        self,
        document_id: str,
        tenant_id: str,
        filename: str,
        status: str,
        chunk_count: int,
        message: str,
        indexed: bool,
    ) -> Dict[str, Any]:
        if self._database is not None:
            try:
                return self.get_document(document_id, tenant_id)
            except Exception:
                pass

        record = {
            "document_id": document_id,
            "id": document_id,
            "tenant_id": tenant_id,
            "filename": filename,
            "status": status,
            "chunk_count": chunk_count,
            "message": message,
            "indexed": indexed,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._records[document_id] = record
        return dict(record)

    @staticmethod
    def _public_status(db_status: str, chunk_count: int, indexed: bool) -> str:
        """Map persisted DocumentStatus to the public API status.

        Non-terminal states MUST stay non-terminal, otherwise the client stops
        polling and reports a failure while ingestion is still running.
        """
        state = str(db_status or "").lower()

        if state == DocumentStatus.FAILED.value:
            return "failed"
        if state == DocumentStatus.PENDING.value:
            return "uploaded"
        if state == DocumentStatus.PROCESSING.value:
            return "processing"
        if state == DocumentStatus.COMPLETED.value:
            if chunk_count == 0:
                return "failed"
            return "completed" if indexed else "partial"
        return "unknown"

    def _document_record_to_public(self, record: DocumentRecord) -> Dict[str, Any]:
        raw_status = (
            record.status.value if hasattr(record.status, "value") else str(record.status)
        )
        indexed = bool(getattr(record, "indexed", False))
        chunk_count = int(record.chunk_count or 0)

        public_status = self._public_status(raw_status, chunk_count, indexed)
        message = record.error_message or _STATUS_MESSAGES.get(
            public_status, _STATUS_MESSAGES["unknown"]
        )

        return {
            "document_id": record.id,
            "id": record.id,
            "tenant_id": record.tenant_id,
            "filename": record.original_filename,
            "status": public_status,
            "chunk_count": chunk_count,
            "indexed": indexed,
            "message": message,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            "updated_at": record.updated_at.isoformat() if record.updated_at else None,
            "storage_key": record.storage_key,
            "law_number": record.law_number,
            "law_year": record.law_year,
            "title": record.title,
        }

    def _chunk_record_to_public(self, chunk: DocumentChunkRecord) -> Dict[str, Any]:
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "page_number": chunk.page_number,
            "section_title": chunk.section_title,
            "law_name": chunk.law_name,
            "law_number": chunk.law_number,
            "law_year": chunk.law_year,
            "article_number": chunk.article_number,
            "chunk_type": chunk.chunk_type,
            "embedding_status": chunk.embedding_status,
            "vector_id": chunk.vector_id,
            "metadata": chunk.metadata_json or {},
            "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
        }