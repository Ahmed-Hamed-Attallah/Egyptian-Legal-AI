# src/ingestion/pipeline.py
"""
Ingestion Pipeline.
Orchestrates the document loading process, coordinating the factory,
loaders, and registry.
"""
import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import List, Optional

from src.domain.documents import Document, DocumentType, SourceAuthority, DocumentStatus
from src.ingestion.loader_factory import LoaderFactory
from src.ingestion.document_registry import DocumentRegistry
from src.ingestion.models import ParsedResult
from src.ingestion.ocr.engine import TesseractOCREngine

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Main pipeline for ingesting documents from files.
    Coordinates loading, parsing, and registration.
    """
    def __init__(self, registry: Optional[DocumentRegistry] = None):
        """
        Args:
            registry: Document registry instance. If None, creates a new one.
        """
        self.registry = registry or DocumentRegistry()
        self.ocr_engine = TesseractOCREngine()
        logger.info("Initialized Ingestion Pipeline")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Computes SHA256 hash of a file to detect duplicates."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def ingest_file(self, file_path: Path) -> List[ParsedResult]:
        """
        Ingests a single file.
        Returns List[ParsedResult] for downstream processing (e.g., Preprocessing Pipeline).
        """
        logger.info(f"Starting ingestion for: {file_path.name}")
        
        # Check for duplicates
        if self.registry.is_duplicate(file_path):
            logger.warning(f"Duplicate file detected: {file_path.name}")
            raise ValueError(f"File already processed: {file_path.name}")
        
        # Register as pending
        self.registry.register_pending(file_path)
        
        try:
            # Get the appropriate loader
            loader = LoaderFactory.get_loader(file_path)
            
            # Inject OCR engine if needed (for PDFs)
            if hasattr(loader, 'ocr_engine') and loader.ocr_engine is None:
                loader.ocr_engine = self.ocr_engine
            
            # Load the documents (returns ParsedResult objects)
            parsed_results: List[ParsedResult] = loader.load(file_path)
            
            # Compute file-level metadata
            file_hash = self._compute_file_hash(file_path)
            file_size = file_path.stat().st_size
            mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            
            # Heuristic mapping to Domain Enums
            name_lower = file_path.name.lower()
            # Use LEGISLATION and OTHER instead of LAW and USER_UPLOAD
            doc_type = DocumentType.LEGISLATION if any(k in name_lower for k in ["law", "قانون", "مرسوم"]) else DocumentType.OTHER
            # Use PARLIAMENT and OTHER instead of OFFICIAL and USER_UPLOADED
            source_auth = SourceAuthority.PARLIAMENT if "official" in str(file_path).lower() else SourceAuthority.OTHER
            # Create strict Domain Documents for tracking/registry
            for idx, result in enumerate(parsed_results):
                merged_metadata = {
                    **result.metadata,
                    "extracted_content_length": len(result.content),
                    "chunk_index": idx
                }
                
                # We create the Document object to satisfy the Domain Model requirements
                # In production, this would be saved to the database here.
                domain_doc = Document(
                    title=file_path.name,
                    document_type=doc_type,
                    source_authority=source_auth,
                    file_path=str(file_path.absolute()),
                    file_hash=file_hash,
                    file_size_bytes=file_size,
                    mime_type=mime_type,
                    metadata=merged_metadata
                )
                # Note: In a full implementation, we would save domain_doc to DB here.
            
            # Update registry
            self.registry.update_status(
                file_path,
                status=DocumentStatus.LOADED,
                document_count=len(parsed_results)
            )
            
            logger.info(f"Successfully ingested {len(parsed_results)} results from {file_path.name}")
            return parsed_results
            
        except Exception as e:
            # Update registry with failure
            self.registry.update_status(
                file_path,
                status=DocumentStatus.FAILED,
                error_message=str(e)
            )
            logger.error(f"Failed to ingest {file_path.name}: {e}")
            raise

    def ingest_directory(self, directory_path: Path, recursive: bool = False) -> List[ParsedResult]:
        """
        Ingests all supported files in a directory.
        Args:
            directory_path: Path to the directory.
            recursive: Whether to search subdirectories.
        Returns:
            List of all ParsedResult objects extracted.
        """
        logger.info(f"Starting directory ingestion: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        all_results = []
        
        # Get all files
        pattern = "**/*" if recursive else "*"
        for file_path in directory_path.glob(pattern):
            if not file_path.is_file():
                continue
            
            # Skip unsupported files
            try:
                LoaderFactory.get_loader(file_path)
            except ValueError:
                logger.debug(f"Skipping unsupported file: {file_path.name}")
                continue
            
            # Ingest the file
            try:
                results = self.ingest_file(file_path)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path.name}: {e}")
                continue  # Continue with other files
        
        logger.info(f"Successfully ingested {len(all_results)} results from directory")
        return all_results

    def get_registry_summary(self) -> dict:
        """Returns a summary of the registry status."""
        records = self.registry.get_all_records()
        summary = {
            "total_files": len(records),
            "by_status": {}
        }
        for record in records.values():
            status = record.status.value
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
        return summary