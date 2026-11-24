#!/usr/bin/env python3
"""
Bulk Ingestion Script for Egyptian Legal AI.

Processes legal documents through:
  Parsing -> Preprocessing -> Legal-Aware Chunking -> Quality Gate
  -> Embedding -> Qdrant

PostgreSQL stores document records and processing status.
Qdrant stores the resulting chunk embeddings.

Quality gate: low-quality chunks (mojibake, PDF UI garbage, repetitive loops,
glued quotes, empty/short) are dropped BEFORE embedding, so only clean legal
provisions reach the vector store.

Usage:
    python -m scripts.bulk_ingest
"""

import hashlib
import logging
import mimetypes
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None
    logging_redirect_tqdm = None
    print("WARNING: tqdm is not installed. Install with: pip install tqdm")

from src.config.logging_config import setup_logging
from src.config.settings import settings
from src.ingestion.loader_factory import LoaderFactory
from src.ingestion.ocr.engine import TesseractOCREngine
from src.preprocessing.pipeline import PreprocessingPipeline
from src.chunking.pipeline import ChunkingPipeline
from src.embeddings.model_registry import get_default_embedding_model
from src.vector_store.qdrant_store import QdrantVectorStore
from src.storage.database.database import get_database
from src.storage.database.repositories.document_repository import DocumentRepository
from src.storage.database.models.document import DocumentRecord, DocumentStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file using streaming reads."""
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256_hash.update(block)
    return sha256_hash.hexdigest()


def get_supported_files(directory: Path) -> List[Path]:
    """
    Recursively discover supported files.
    Supports settings.ALLOWED_FILE_TYPES values such as:
        ["pdf", "docx", ".csv"]
    """
    supported_extensions = {
        f".{str(ext).lower().lstrip('.')}"
        for ext in settings.ALLOWED_FILE_TYPES
    }
    files = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in supported_extensions:
            files.append(path)
    return files


def discover_input_files() -> List[Path]:
    """
    Discover files from configured ingestion directories.
    Resolve paths and remove duplicates in case directories overlap.
    """
    raw_dirs = [
        settings.DATA_DIR / "raw" / "official_laws",
    ]
    discovered = {}
    for directory in raw_dirs:
        if not directory.exists():
            logger.warning("Input directory does not exist: %s", directory)
            continue
        for file_path in get_supported_files(directory):
            resolved = file_path.resolve()
            discovered[str(resolved)] = resolved
    return sorted(discovered.values(), key=lambda path: str(path).lower())


# ---------------------------------------------------------------------
# Metadata utilities
# ---------------------------------------------------------------------

def _get_metadata_value(metadata: Any, key: str) -> Optional[Any]:
    """
    Read a metadata value from either a dictionary or an object.
    This supports ParsedResult metadata dictionaries and preprocessing
    metadata objects without assuming a specific implementation.
    """
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        value = metadata.get(key)
    else:
        value = getattr(metadata, key, None)
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def collect_document_metadata(
    parsed_results: List[Any],
    fallback_metadata: Any = None,
) -> Dict[str, Any]:
    """
    Collect useful document-level metadata across all parsed results.

    CSV files may contain different legal metadata on individual rows.
    We therefore do not rely exclusively on the final parsed row.
    The first non-empty value is retained for each document-level field.
    """
    collected = {
        "law_number": None,
        "law_year": None,
        "title": None,
    }
    metadata_sources = []
    for parsed in parsed_results:
        metadata_sources.append(getattr(parsed, "metadata", None))
    if fallback_metadata is not None:
        metadata_sources.append(fallback_metadata)

    for metadata in metadata_sources:
        for key in collected:
            if collected[key] is not None:
                continue
            value = _get_metadata_value(metadata, key)
            # Some CSV classifiers expose the legal name as law_name.
            if key == "title" and value is None:
                value = _get_metadata_value(metadata, "law_name")
            if value is not None:
                collected[key] = value
    return collected


def log_csv_schema_warnings(parsed_results, file_path):
    """Log unique CSV schema warnings once per file."""
    issues = set()
    for parsed in parsed_results:
        metadata = getattr(parsed, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        if metadata.get("csv_schema_needs_review"):
            for issue in metadata.get("csv_schema_issues", []):
                issues.add(str(issue))
    if issues:
        logger.warning(
            "CSV schema needs review for %s. Issues: %s",
            file_path.name,
            sorted(issues),
        )


# ---------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------

def process_document(
    file_path: Path,
    file_hash: str,
    doc_id: str,
    tenant_id: str,
    file_size: int,
    mime_type: str,
    session: Any,
    preprocessing_pipeline: PreprocessingPipeline,
    chunking_pipeline: ChunkingPipeline,
    embedding_model: Any,
    vector_store: QdrantVectorStore,
) -> int:
    """
    Process a document section by section.

    For EACH parsed section:
        1. Preprocess
        2. Chunk
        3. Quality gate — drop mojibake / PDF UI garbage / repetitive loops
        4. Generate embeddings
        5. Insert that section's chunks into Qdrant
        6. Update PostgreSQL metadata and cumulative chunk count
        7. Commit PostgreSQL

    The document is marked COMPLETED only after every section succeeds.

    Returns:
        Total number of chunks inserted into Qdrant.
    """
    repo = DocumentRepository(session)

    # -------------------------------------------------------------
    # Phase 1: Parse entire file
    # -------------------------------------------------------------
    logger.info("Parsing document: %s", file_path.name)
    loader = LoaderFactory.get_loader(file_path)
    if hasattr(loader, "ocr_engine") and loader.ocr_engine is None:
        loader.ocr_engine = TesseractOCREngine()
    parsed_results = loader.load(file_path)
    if not parsed_results:
        raise ValueError("Parsing produced no content.")
    log_csv_schema_warnings(parsed_results, file_path)

    total_sections = len(parsed_results)
    total_chunks = 0
    processed_sections = 0
    quality_dropped = 0

    # Accumulate document metadata as sections are processed.
    document_metadata = {
        "law_number": None,
        "law_year": None,
        "title": None,
    }

    logger.info(
        "Document parsed: %s sections found in %s",
        total_sections,
        file_path.name,
    )

    # -------------------------------------------------------------
    # LOOP: Process every section independently
    # -------------------------------------------------------------
    section_iterator = enumerate(parsed_results, start=1)
    if HAS_TQDM and total_sections > 1:
        section_iterator = tqdm(
            section_iterator,
            total=total_sections,
            desc=f"Sections: {file_path.name[:25]}",
            unit="section",
            leave=False,
            colour="cyan",
            ncols=100,
        )

    for section_number, parsed in section_iterator:
        logger.info(
            "[%s] Processing section %d/%d",
            file_path.name,
            section_number,
            total_sections,
        )
        section_chunks_count = 0

        # ---------------------------------------------------------
        # Phase 2: Validate section content
        # ---------------------------------------------------------
        content = getattr(parsed, "content", None)
        if not isinstance(content, str) or not content.strip():
            logger.warning(
                "Skipping empty section %d of %s",
                section_number,
                file_path.name,
            )
            continue

        # ---------------------------------------------------------
        # Phase 3: Preprocessing
        # ---------------------------------------------------------
        preprocessed = preprocessing_pipeline.process(
            text=content,
            doc_id=f"{doc_id}:{section_number}",
            source_path=str(file_path),
        )
        preprocessed_metadata = getattr(
            preprocessed,
            "metadata",
            None,
        )

        # ---------------------------------------------------------
        # Phase 4: Chunking + Quality Gate
        #
        # process_with_report() runs the chunker AND the quality
        # filter in one call. Low-quality chunks (mojibake, PDF UI
        # garbage, repetitive TOC loops, glued quotes, empty/short)
        # are dropped before they ever reach the embedding model.
        # ---------------------------------------------------------
        chunks, quality_report = chunking_pipeline.process_with_report(preprocessed)

        if not chunks:
            if quality_report.total_rejected > 0:
                logger.warning(
                    "No chunks generated for section %d of %s "
                    "(quality gate rejected all %d raw chunks: %s)",
                    section_number,
                    file_path.name,
                    quality_report.total_rejected,
                    dict(quality_report.by_reason),
                )
            else:
                logger.warning(
                    "No chunks generated for section %d of %s",
                    section_number,
                    file_path.name,
                )
            quality_dropped += quality_report.total_rejected
            continue

        if quality_report.total_rejected > 0:
            quality_dropped += quality_report.total_rejected
            logger.info(
                "[Quality Gate] %s section %d: accepted=%d dropped=%d reasons=%s",
                file_path.name,
                section_number,
                quality_report.total_accepted,
                quality_report.total_rejected,
                dict(quality_report.by_reason),
            )
            for sample in quality_report.rejected_samples[:2]:
                logger.debug(
                    "Dropped chunk %s/%s reason=%s sample=%r",
                    sample.get("document_id"),
                    sample.get("chunk_index"),
                    sample.get("reason"),
                    sample.get("sample"),
                )

        # -----------------------------------------------------
        # Phase 5: Generate embeddings
        # -----------------------------------------------------
        texts = [chunk.content for chunk in chunks]
        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                f"Invalid chunk content in section {section_number}."
            )

        embeddings = embedding_model.encode_documents(texts)
        if embeddings is None:
            raise ValueError(
                f"Embedding model returned None for section "
                f"{section_number}."
            )
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding mismatch in section {section_number}: "
                f"{len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        # -----------------------------------------------------
        # Phase 6: Store THIS section in Qdrant
        # -----------------------------------------------------
        logger.info(
            "Qdrant insert started: %s | section %d/%d | %d chunks",
            file_path.name,
            section_number,
            total_sections,
            len(chunks),
        )
        vector_store.insert(chunks, embeddings)

        # Count only after insert returns successfully.
        section_chunks_count = len(chunks)
        total_chunks += section_chunks_count

        logger.info(
            "Qdrant insert returned successfully: %s | "
            "section %d | %d chunks | cumulative %d",
            file_path.name,
            section_number,
            section_chunks_count,
            total_chunks,
        )

        # ---------------------------------------------------------
        # Phase 7: Extract and accumulate metadata
        # ---------------------------------------------------------
        section_metadata = collect_document_metadata(
            parsed_results=[parsed],
            fallback_metadata=preprocessed_metadata,
        )
        for key, value in section_metadata.items():
            if value is not None and document_metadata[key] is None:
                document_metadata[key] = value

        # ---------------------------------------------------------
        # Phase 8: Update PostgreSQL AFTER THIS SECTION
        # ---------------------------------------------------------
        metadata_updates = {
            key: value
            for key, value in document_metadata.items()
            if value is not None
        }
        # Set a fallback title if no legal title has been extracted yet.
        if document_metadata["title"] is None:
            metadata_updates["title"] = file_path.name

        if metadata_updates:
            repo.update_metadata(
                doc_id,
                tenant_id,
                **metadata_updates,
            )

        # Save cumulative progress after each section.
        repo.update_status(
            doc_id,
            tenant_id,
            DocumentStatus.PROCESSING,
            chunk_count=total_chunks,
        )

        # Commit the current section's PostgreSQL updates.
        session.commit()
        processed_sections += 1

        logger.info(
            "PostgreSQL progress committed: %s | "
            "section %d/%d | section chunks=%d | total chunks=%d",
            file_path.name,
            section_number,
            total_sections,
            section_chunks_count,
            total_chunks,
        )

    # -------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------
    if processed_sections == 0:
        raise ValueError(
            f"No valid sections were processed in {file_path.name}."
        )
    if total_chunks == 0:
        raise ValueError(
            f"No chunks were generated for {file_path.name}."
        )

    # -------------------------------------------------------------
    # Final document status
    # -------------------------------------------------------------
    repo.update_status(
        doc_id,
        tenant_id,
        DocumentStatus.COMPLETED,
        chunk_count=total_chunks,
    )
    session.commit()

    logger.info(
        "DOCUMENT COMPLETED: %s | sections=%d/%d | "
        "total_chunks=%d | dropped_by_quality=%d",
        file_path.name,
        processed_sections,
        total_sections,
        total_chunks,
        quality_dropped,
    )

    return total_chunks


# ---------------------------------------------------------------------
# Main ingestion process
# ---------------------------------------------------------------------

def main():
    setup_logging()
    logger.info("Starting Egyptian Legal AI bulk ingestion.")

    # -------------------------------------------------------------
    # Initialize database
    # -------------------------------------------------------------
    db = get_database()
    db.create_all()
    logger.info("Database initialized.")

    # -------------------------------------------------------------
    # Initialize pipelines and models
    # -------------------------------------------------------------
    preprocessing_pipeline = PreprocessingPipeline()
    chunking_pipeline = ChunkingPipeline()

    logger.info("Loading embedding model...")
    embedding_model = get_default_embedding_model()

    logger.info("Connecting to Qdrant...")
    vector_store = QdrantVectorStore()

    # -------------------------------------------------------------
    # Discover input files
    # -------------------------------------------------------------
    all_files = discover_input_files()
    logger.info("Found %s supported files.", len(all_files))
    if not all_files:
        logger.warning("No supported files found. Nothing to ingest.")
        return

    tenant_id = settings.DEFAULT_TENANT_ID
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    total_chunks = 0
    total_quality_dropped = 0

    log_context = (
        logging_redirect_tqdm()
        if HAS_TQDM
        else nullcontext()
    )

    with log_context:
        if HAS_TQDM:
            pbar = tqdm(
                all_files,
                desc="Ingesting documents",
                unit="file",
                bar_format="{l_bar}{bar:40}{r_bar}",
                ncols=120,
                colour="green",
            )
        else:
            pbar = all_files

        for file_path in pbar:
            if HAS_TQDM:
                pbar.set_description(
                    f"Ingesting: {file_path.name[:35]}"
                )

            try:
                # -------------------------------------------------
                # File information
                # -------------------------------------------------
                if not file_path.exists() or not file_path.is_file():
                    raise FileNotFoundError(
                        f"Input file no longer exists: {file_path}"
                    )

                file_hash = compute_file_hash(file_path)
                file_size = file_path.stat().st_size
                mime_type = (
                    mimetypes.guess_type(file_path.name)[0]
                    or "application/octet-stream"
                )

                # Stable document ID across repeated ingestion runs.
                doc_id = str(
                    uuid.uuid5(uuid.NAMESPACE_DNS, file_hash)
                )

                # -------------------------------------------------
                # Database record
                # -------------------------------------------------
                with db.session() as session:
                    repo = DocumentRepository(session)
                    existing = repo.get_by_id(
                        doc_id,
                        tenant_id,
                        include_deleted=True,
                    )

                    if (
                        existing
                        and existing.status == DocumentStatus.COMPLETED
                    ):
                        skipped_count += 1
                        logger.info(
                            "Skipping %s: already completed.",
                            file_path.name,
                        )
                        if HAS_TQDM:
                            pbar.set_postfix(
                                ok=processed_count,
                                fail=failed_count,
                                skip=skipped_count,
                                chunks=total_chunks,
                                dropped=total_quality_dropped,
                                refresh=True,
                            )
                        continue

                    if not existing:
                        doc_record = DocumentRecord(
                            id=doc_id,
                            tenant_id=tenant_id,
                            original_filename=file_path.name,
                            storage_key=str(file_path.resolve()),
                            mime_type=mime_type,
                            file_size_bytes=file_size,
                            status=DocumentStatus.PROCESSING,
                        )
                        repo.create(doc_record)
                    else:
                        repo.update_status(
                            doc_id,
                            tenant_id,
                            DocumentStatus.PROCESSING,
                        )

                    # Commit PROCESSING status before the expensive work.
                    session.commit()

                    try:
                        if file_size > settings.MAX_UPLOAD_SIZE_BYTES:
                            logger.warning(
                                "%s is %.1f MB, exceeding the configured "
                                "maximum upload size of %.1f MB. "
                                "Processing will continue.",
                                file_path.name,
                                file_size / (1024 * 1024),
                                settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024),
                            )

                        chunk_count = process_document(
                            file_path=file_path,
                            file_hash=file_hash,
                            doc_id=doc_id,
                            tenant_id=tenant_id,
                            file_size=file_size,
                            mime_type=mime_type,
                            session=session,
                            preprocessing_pipeline=preprocessing_pipeline,
                            chunking_pipeline=chunking_pipeline,
                            embedding_model=embedding_model,
                            vector_store=vector_store,
                        )

                        processed_count += 1
                        total_chunks += chunk_count
                        logger.info(
                            "Successfully processed %s: %s chunks.",
                            file_path.name,
                            chunk_count,
                        )

                    except Exception as processing_error:
                        # Discard any uncommitted PostgreSQL changes.
                        try:
                            session.rollback()
                        except Exception:
                            logger.exception(
                                "Database rollback failed for %s.",
                                file_path.name,
                            )

                        logger.error(
                            "Failed to process %s: %s",
                            file_path.name,
                            processing_error,
                            exc_info=True,
                        )

                        # Record failure in a fresh transaction.
                        try:
                            repo.update_status(
                                doc_id,
                                tenant_id,
                                DocumentStatus.FAILED,
                                error_message=str(processing_error)[:4000],
                            )
                            session.commit()
                        except Exception:
                            session.rollback()
                            logger.exception(
                                "Could not record FAILED status for %s.",
                                file_path.name,
                            )
                        failed_count += 1

            except Exception as file_error:
                failed_count += 1
                logger.error(
                    "Could not ingest %s: %s",
                    file_path,
                    file_error,
                    exc_info=True,
                )

            finally:
                if HAS_TQDM:
                    pbar.set_postfix(
                        ok=processed_count,
                        fail=failed_count,
                        skip=skipped_count,
                        chunks=total_chunks,
                        dropped=total_quality_dropped,
                        refresh=True,
                    )

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("Bulk ingestion complete.")
    logger.info("Total discovered: %s", len(all_files))
    logger.info("Successfully processed: %s", processed_count)
    logger.info("Skipped: %s", skipped_count)
    logger.info("Failed: %s", failed_count)
    logger.info("Total chunks generated: %s", total_chunks)
    logger.info("Total chunks dropped by quality gate: %s", total_quality_dropped)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()