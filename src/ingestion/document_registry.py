"""
Document Registry.
Tracks the status of ingested documents to prevent duplicate processing
and maintain an audit trail.
"""
import hashlib
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from src.domain.documents import Document

logger = logging.getLogger(__name__)

class DocumentStatus(str, Enum):
    PENDING = "pending"
    LOADED = "loaded"
    PROCESSED = "processed"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentRecord:
    """Represents a document record in the registry."""
    
    def __init__(
        self,
        file_path: str,
        file_hash: str,
        status: DocumentStatus,
        document_count: int = 0,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        error_message: Optional[str] = None
    ):
        self.file_path = file_path
        self.file_hash = file_hash
        self.status = status
        self.document_count = document_count
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.error_message = error_message
        self._documents: dict[str, "Document"] = {}
    
    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "status": self.status.value,
            "document_count": self.document_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "error_message": self.error_message
        }

class DocumentRegistry:
    """
    In-memory registry for tracking document ingestion status.
    In production, this should be backed by PostgreSQL.
    """
    
    def __init__(self):
        self._records: Dict[str, DocumentRecord] = {}
        logger.info("Initialized Document Registry")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Computes SHA256 hash of a file to detect duplicates."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def is_duplicate(self, file_path: Path) -> bool:
        """Checks if a file has already been processed (by hash)."""
        file_hash = self._compute_file_hash(file_path)
        return file_hash in [record.file_hash for record in self._records.values()]
    
    def register_pending(self, file_path: Path) -> DocumentRecord:
        """Registers a file as pending ingestion."""
        file_hash = self._compute_file_hash(file_path)
        record = DocumentRecord(
            file_path=str(file_path.absolute()),
            file_hash=file_hash,
            status=DocumentStatus.PENDING
        )
        self._records[record.file_path] = record
        logger.info(f"Registered pending document: {file_path.name}")
        return record
    
    def update_status(
        self,
        file_path: Path,
        status: DocumentStatus,
        document_count: int = 0,
        error_message: Optional[str] = None
    ):
        """Updates the status of a document record."""
        record = self._records.get(str(file_path.absolute()))
        if not record:
            logger.warning(f"Document not found in registry: {file_path}")
            return
        
        record.status = status
        record.document_count = document_count
        record.updated_at = datetime.utcnow()
        record.error_message = error_message
        
        logger.info(f"Updated document status: {file_path.name} -> {status.value}")
    
    def get_record(self, file_path: Path) -> Optional[DocumentRecord]:
        """Retrieves a document record by file path."""
        return self._records.get(str(file_path.absolute()))
    
    def get_all_records(self) -> Dict[str, DocumentRecord]:
        """Returns all document records."""
        return self._records.copy()