"""
Custom exception classes for the Legal AI domain layer.
This module defines typed exceptions that carry context about errors,
support proper error handling at different layers (API, services, workers),
and can be serialized into meaningful API error responses.
"""
from typing import Any, Dict, Optional

# -----------------------------------------------------------------------------
# Base Exception
# -----------------------------------------------------------------------------
class LegalAIError(Exception):
    """
    Base exception for all Legal AI domain errors.
    All custom exceptions inherit from this class, allowing the application
    to catch all domain-specific errors with a single except clause if needed.
    """
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ):
        """
        Initialize the exception.
        Args:
            query: The query that caused the error.
            message: Human-readable error description.
            details: Structured context about the error (e.g., IDs, parameters).
            error_code: Machine-readable code for API responses.
        """
        super().__init__(message)
        self.message = message
        self.query = query
        self.details = details or {}
        self.error_code = error_code or self.__class__.__name__.upper()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the exception to a dictionary for API error responses.
        Returns:
            Dictionary with error_code, message, and details.
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }

# -----------------------------------------------------------------------------
# Document & Ingestion Errors
# -----------------------------------------------------------------------------
class DocumentProcessingError(LegalAIError):
    """Base exception for document-related errors."""
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if document_id:
            details["document_id"] = document_id
        super().__init__(message, query=None, details=details, error_code="DOCUMENT_PROCESSING_ERROR")

class DocumentNotFoundError(DocumentProcessingError):
    """Raised when a document ID doesn't exist."""
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document with ID '{document_id}' not found.",
            document_id=document_id,
        )
        self.error_code = "DOCUMENT_NOT_FOUND"

class DocumentParsingError(DocumentProcessingError):
    """Raised when file parsing fails (PDF, DOCX, etc.)."""
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        file_path: Optional[str] = None,
        parse_error: Optional[str] = None,
    ):
        details = {}
        if file_path:
            details["file_path"] = file_path
        if parse_error:
            details["parse_error"] = parse_error
        super().__init__(message, document_id, details)
        self.error_code = "DOCUMENT_PARSING_ERROR"

class DocumentChunkingError(DocumentProcessingError):
    """Raised when document chunking/splitting fails."""
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
    ):
        details = {}
        if chunking_strategy:
            details["chunking_strategy"] = chunking_strategy
        super().__init__(message, document_id, details)
        self.error_code = "DOCUMENT_CHUNKING_ERROR"

class DocumentValidationError(DocumentProcessingError):
    """Raised when document metadata is invalid."""
    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        validation_errors: Optional[Dict[str, str]] = None,
    ):
        details = {}
        if validation_errors:
            details["validation_errors"] = validation_errors
        super().__init__(message, document_id, details)
        self.error_code = "DOCUMENT_VALIDATION_ERROR"

class DuplicateDocumentError(DocumentProcessingError):
    """Raised when a file hash already exists (deduplication)."""
    def __init__(self, file_hash: str, existing_document_id: str):
        super().__init__(
            message=f"Document with hash '{file_hash}' already exists.",
            details={
                "file_hash": file_hash,
                "existing_document_id": existing_document_id,
            },
        )
        self.error_code = "DUPLICATE_DOCUMENT"

# -----------------------------------------------------------------------------
# Legal Source Errors
# -----------------------------------------------------------------------------
class LegalSourceError(LegalAIError):
    """Base exception for legal source errors."""
    def __init__(
        self,
        message: str,
        legal_source_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if legal_source_id:
            details["legal_source_id"] = legal_source_id
        super().__init__(message, details, "LEGAL_SOURCE_ERROR")

class LegalSourceNotFoundError(LegalSourceError):
    """Raised when a legal source ID doesn't exist."""
    def __init__(self, legal_source_id: str):
        super().__init__(
            message=f"Legal source with ID '{legal_source_id}' not found.",
            legal_source_id=legal_source_id,
        )
        self.error_code = "LEGAL_SOURCE_NOT_FOUND"

class LegalSourceValidationError(LegalSourceError):
    """Raised when legal source metadata is invalid."""
    def __init__(
        self,
        message: str,
        legal_source_id: Optional[str] = None,
        validation_errors: Optional[Dict[str, str]] = None,
    ):
        details = {}
        if validation_errors:
            details["validation_errors"] = validation_errors
        super().__init__(message, legal_source_id, details)
        self.error_code = "LEGAL_SOURCE_VALIDATION_ERROR"

class LegalVersionConflictError(LegalSourceError):
    """Raised when there's a version conflict (e.g., overlapping dates)."""
    def __init__(
        self,
        message: str,
        legal_source_id: Optional[str] = None,
        version_id: Optional[str] = None,
        conflicting_version_id: Optional[str] = None,
    ):
        details = {}
        if version_id:
            details["version_id"] = version_id
        if conflicting_version_id:
            details["conflicting_version_id"] = conflicting_version_id
        super().__init__(message, legal_source_id, details)
        self.error_code = "LEGAL_VERSION_CONFLICT"

class InvalidVersionTransitionError(LegalSourceError):
    """Raised when an illegal version status change is attempted."""
    def __init__(
        self,
        from_status: str,
        to_status: str,
        version_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Cannot transition from '{from_status}' to '{to_status}'.",
            details={
                "from_status": from_status,
                "to_status": to_status,
                "version_id": version_id,
            },
        )
        self.error_code = "INVALID_VERSION_TRANSITION"

# -----------------------------------------------------------------------------
# Query & Retrieval Errors
# -----------------------------------------------------------------------------
class QueryError(LegalAIError):
    """Base exception for query-related errors."""
    def __init__(
        self,
        message: str,
        query_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if query_id:
            details["query_id"] = query_id
        super().__init__(message, details, "QUERY_ERROR")

class QueryValidationError(QueryError):
    """Raised when query parameters are invalid."""
    def __init__(
        self,
        message: str,
        query_id: Optional[str] = None,
        validation_errors: Optional[Dict[str, str]] = None,
    ):
        details = {}
        if validation_errors:
            details["validation_errors"] = validation_errors
        super().__init__(message, query_id, details)
        self.error_code = "QUERY_VALIDATION_ERROR"

class QueryTimeoutError(QueryError):
    """Raised when a query exceeds the time limit."""
    def __init__(self, query_id: str, timeout_seconds: float):
        super().__init__(
            message=f"Query exceeded timeout of {timeout_seconds} seconds.",
            query_id=query_id,
            details={"timeout_seconds": timeout_seconds},
        )
        self.error_code = "QUERY_TIMEOUT"

class RetrievalError(LegalAIError):
    """Raised when retrieval fails."""
    pass


class DocumentLoadError(LegalAIError):
    """Raised when a document file cannot be loaded or read."""
    pass


class UnsupportedFormatError(LegalAIError):
    """Raised when a file format is not supported for ingestion."""

    def __init__(self, file_format: str, supported_formats: list[str] | None = None):
        self.file_format = file_format
        self.supported_formats = supported_formats or []
        if supported_formats:
            msg = (
                f"Unsupported file format: '{file_format}'. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        else:
            msg = f"Unsupported file format: '{file_format}'"
        super().__init__(msg)


class NoRelevantChunksFoundError(RetrievalError):
    """Raised when vector search returns no results."""
    def __init__(self, query_text: str, top_k: int):
        super().__init__(
            message=f"No relevant chunks found for query (top_k={top_k}).",
            details={
                "query_text": query_text[:100],  # Truncate for safety
                "top_k": top_k,
            },
        )
        self.error_code = "NO_RELEVANT_CHUNKS_FOUND"

class EmbeddingError(RetrievalError):
    """Raised when embedding generation fails."""
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        text_length: Optional[int] = None,
    ):
        details = {}
        if model_name:
            details["model_name"] = model_name
        if text_length:
            details["text_length"] = text_length
        super().__init__(message, details)
        self.error_code = "EMBEDDING_ERROR"

class VectorDatabaseError(RetrievalError):
    """Raised when Qdrant connection or query fails."""
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        details = {}
        if operation:
            details["operation"] = operation
        if collection_name:
            details["collection_name"] = collection_name
        super().__init__(message, details)
        self.error_code = "VECTOR_DATABASE_ERROR"

# -----------------------------------------------------------------------------
# Response Generation Errors
# -----------------------------------------------------------------------------
class ResponseGenerationError(LegalAIError):
    """Base exception for LLM generation errors."""
    def __init__(
        self,
        message: str,
        response_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if response_id:
            details["response_id"] = response_id
        super().__init__(message, details, "RESPONSE_GENERATION_ERROR")

class LLMError(ResponseGenerationError):
    """Base exception for LLM provider errors."""
    pass

class ProviderError(LLMError):
    """Raised when an LLM provider encounters an error."""
    def __init__(self, message: str, provider: Optional[str] = None, **context):
        details = {"provider": provider} if provider else {}
        details.update(context)
        super().__init__(message, details=details)
        self.error_code = "PROVIDER_ERROR"

class ProviderUnavailableError(ProviderError):
    """Raised when an LLM provider is unavailable."""
    def __init__(self, message: str, provider: Optional[str] = None, **context):
        super().__init__(message, provider=provider, **context)
        self.error_code = "PROVIDER_UNAVAILABLE"

class LLMServiceError(ResponseGenerationError):
    """Raised when LLM API call fails."""
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        response_id: Optional[str] = None,
        api_error: Optional[str] = None,
    ):
        details = {}
        if model_name:
            details["model_name"] = model_name
        if api_error:
            details["api_error"] = api_error
        super().__init__(message, response_id, details)
        self.error_code = "LLM_SERVICE_ERROR"

class LLMTimeoutError(ResponseGenerationError):
    """Raised when LLM response takes too long."""
    def __init__(self, response_id: str, model_name: str, timeout_seconds: float):
        super().__init__(
            message=f"LLM response exceeded timeout of {timeout_seconds} seconds.",
            response_id=response_id,
            details={
                "model_name": model_name,
                "timeout_seconds": timeout_seconds,
            },
        )
        self.error_code = "LLM_TIMEOUT"

class ContextWindowExceededError(ResponseGenerationError):
    """Raised when prompt exceeds model's context window."""
    def __init__(
        self,
        response_id: str,
        model_name: str,
        token_count: int,
        max_tokens: int,
    ):
        super().__init__(
            message=f"Prompt token count ({token_count}) exceeds model limit ({max_tokens}).",
            response_id=response_id,
            details={
                "model_name": model_name,
                "token_count": token_count,
                "max_tokens": max_tokens,
            },
        )
        self.error_code = "CONTEXT_WINDOW_EXCEEDED"

class CitationValidationError(ResponseGenerationError):
    """Raised when generated citations don't match retrieved chunks."""
    def __init__(
        self,
        response_id: str,
        message: str = "Generated citations do not match retrieved chunks.",
    ):
        super().__init__(message, response_id)
        self.error_code = "CITATION_VALIDATION_ERROR"

# -----------------------------------------------------------------------------
# Conversation Errors
# -----------------------------------------------------------------------------
class ConversationError(LegalAIError):
    """Base exception for conversation errors."""
    def __init__(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if conversation_id:
            details["conversation_id"] = conversation_id
        super().__init__(message, details, "CONVERSATION_ERROR")

class ConversationNotFoundError(ConversationError):
    """Raised when a conversation ID doesn't exist."""
    def __init__(self, conversation_id: str):
        super().__init__(
            message=f"Conversation with ID '{conversation_id}' not found.",
            conversation_id=conversation_id,
        )
        self.error_code = "CONVERSATION_NOT_FOUND"

class ConversationArchivedError(ConversationError):
    """Raised when attempting to query an archived conversation."""
    def __init__(self, conversation_id: str):
        super().__init__(
            message=f"Conversation '{conversation_id}' is archived and read-only.",
            conversation_id=conversation_id,
        )
        self.error_code = "CONVERSATION_ARCHIVED"

# -----------------------------------------------------------------------------
# Authentication & Authorization Errors
# -----------------------------------------------------------------------------
class SecurityError(LegalAIError):
    """Base exception for security-related errors."""
    pass

class AuthenticationError(SecurityError):
    """Raised when user is not authenticated."""
    def __init__(self, message: str = "Authentication required.", **context):
        super().__init__(message, **context)
        self.error_code = "AUTHENTICATION_ERROR"

class AuthorizationError(SecurityError):
    """Raised when user lacks permission for an action."""
    def __init__(
        self,
        message: str = "Insufficient permissions.",
        user_id: Optional[str] = None,
        required_permission: Optional[str] = None,
        **context
    ):
        details = {}
        if user_id:
            details["user_id"] = user_id
        if required_permission:
            details["required_permission"] = required_permission
        details.update(context)
        super().__init__(message, details, "AUTHORIZATION_ERROR")

class FileValidationError(SecurityError):
    """File validation failed."""
    def __init__(self, message: str, filename: Optional[str] = None, reason: Optional[str] = None, **context):
        details = {}
        if filename:
            details["filename"] = filename
        if reason:
            details["reason"] = reason
        details.update(context)
        super().__init__(message, details, "FILE_VALIDATION_ERROR")

class TenantIsolationError(SecurityError):
    """Tenant boundary violation."""
    def __init__(self, message: str = "Tenant isolation violation", tenant_id: Optional[str] = None, **context):
        details = {}
        if tenant_id:
            details["tenant_id"] = tenant_id
        details.update(context)
        super().__init__(message, details, "TENANT_ISOLATION_ERROR")

# -----------------------------------------------------------------------------
# Configuration & System Errors
# -----------------------------------------------------------------------------
class ConfigurationError(LegalAIError):
    """Raised when configuration is missing or invalid."""
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
    ):
        details = {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, details, "CONFIGURATION_ERROR")

# =============================================================================
# Phase 11 — Storage Exceptions
# =============================================================================
class StorageError(LegalAIError):
    """Base exception for storage-related errors."""
    def __init__(
        self,
        message: str,
        storage_backend: Optional[str] = None,
        file_path: Optional[str] = None,
        **context
    ):
        details = {}
        if storage_backend:
            details["storage_backend"] = storage_backend
        if file_path:
            details["file_path"] = file_path
        details.update(context)
        super().__init__(message, details, "STORAGE_ERROR")

class DatabaseError(StorageError):
    """Database operation failed."""
    def __init__(self, message: str, operation: Optional[str] = None, **context):
        details = {}
        if operation:
            details["operation"] = operation
        details.update(context)
        super().__init__(message, **details)

class DocumentStorageError(StorageError):
    """Document file storage operation failed."""
    def __init__(self, message: str, storage_key: Optional[str] = None, **context):
        details = {}
        if storage_key:
            details["storage_key"] = storage_key
        details.update(context)
        super().__init__(message, **details)

class CacheError(StorageError):
    """Cache operation failed."""
    def __init__(self, message: str, cache_key: Optional[str] = None, **context):
        details = {}
        if cache_key:
            details["cache_key"] = cache_key
        details.update(context)
        super().__init__(message, **details)

class IngestionError(LegalAIError):
    """Document ingestion failed."""
    pass

class RAGError(LegalAIError):
    """RAG pipeline failure."""
    pass

class RerankingError(RetrievalError):
    """Raised when cross-encoder reranking fails."""
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        chunk_count: Optional[int] = None,
    ):
        details = {}
        if model_name:
            details["model_name"] = model_name
        if chunk_count:
            details["chunk_count"] = chunk_count
        super().__init__(message, details)
        self.error_code = "RERANKING_ERROR"