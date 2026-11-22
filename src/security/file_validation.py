"""
File upload validation.

Validates uploads on multiple axes:
1. Filename safety (no path traversal, no control characters).
2. Extension allow-list.
3. File size limits.
4. MIME type via libmagic (not trusting extension alone).
5. Magic-byte signature verification for common formats.

This module does NOT execute or parse the file contents — that's the job of
the ingestion pipeline, which runs AFTER validation.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, List, Optional

from src.config.settings import settings
from src.domain.exceptions import FileValidationError

logger = logging.getLogger(__name__)


# Magic byte signatures for common formats
_SIGNATURES = {
    "pdf": [b"%PDF"],
    "docx": [b"PK\x03\x04"],  # ZIP-based
}

# Extension -> expected MIME types
_EXTENSION_MIME = {
    "pdf": {"application/pdf"},
    "docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    "txt": {"text/plain"},
    "md": {"text/plain", "text/markdown"},
    "csv": {"text/csv", "text/plain"},
    "tsv": {"text/tab-separated-values", "text/plain"},
}


@dataclass
class ValidationResult:
    """Result of file validation."""
    is_valid: bool
    filename: str
    extension: str
    size_bytes: int
    mime_type: Optional[str] = None
    errors: List[str] = field(default_factory=list)


class FileValidator:
    """Validates uploaded files against security and format rules."""

    def __init__(
        self,
        allowed_extensions: Optional[List[str]] = None,
        max_size_bytes: Optional[int] = None,
    ):
        self._allowed = {ext.lower().lstrip(".") for ext in (allowed_extensions or settings.ALLOWED_FILE_TYPES)}
        self._max_size = max_size_bytes if max_size_bytes is not None else settings.MAX_UPLOAD_SIZE_BYTES

    def validate_filename(self, filename: str) -> str:
        """
        Validate and sanitize a filename.

        Returns the sanitized basename. Raises FileValidationError if the
        filename is fundamentally invalid.
        """
        if not filename:
            raise FileValidationError("Filename is required", reason="empty_filename")

        # Strip directory components — never trust the client's path
        basename = os.path.basename(filename)
        if not basename or basename in {".", ".."}:
            raise FileValidationError(
                f"Invalid filename: {filename!r}",
                filename=filename,
                reason="invalid_basename",
            )

        # Reject control characters and path separators
        for ch in basename:
            if ord(ch) < 32 or ch in {"/", "\\", ":", "*", "?", "\"", "<", ">", "|"}:
                raise FileValidationError(
                    f"Filename contains illegal character: {ch!r}",
                    filename=filename,
                    reason="illegal_character",
                )

        # Reject hidden files (starting with .)
        if basename.startswith("."):
            raise FileValidationError(
                "Hidden files are not allowed",
                filename=filename,
                reason="hidden_file",
            )

        return basename

    def validate_extension(self, filename: str) -> str:
        """Validate the file extension. Returns the lowercase extension."""
        ext = Path(filename).suffix.lower().lstrip(".")
        if not ext:
            raise FileValidationError(
                f"File has no extension: {filename!r}",
                filename=filename,
                reason="no_extension",
            )
        if ext not in self._allowed:
            raise FileValidationError(
                f"File type .{ext} is not allowed. Allowed: {sorted(self._allowed)}",
                filename=filename,
                reason="unsupported_extension",
            )
        return ext

    def validate_size(self, size_bytes: int, filename: str) -> None:
        if size_bytes <= 0:
            raise FileValidationError(
                "File is empty",
                filename=filename,
                reason="empty_file",
            )
        if size_bytes > self._max_size:
            raise FileValidationError(
                f"File size {size_bytes} exceeds maximum {self._max_size} bytes",
                filename=filename,
                reason="file_too_large",
            )

    def detect_mime_type(self, file_content: BinaryIO, filename: str) -> Optional[str]:
        """Detect MIME type using libmagic if available, else fall back to extension."""
        try:
            import magic  # python-magic
            # Read a small sample
            pos = file_content.tell()
            sample = file_content.read(8192)
            file_content.seek(pos)
            if not sample:
                return None
            mime = magic.from_buffer(sample, mime=True)
            return mime
        except ImportError:
            logger.debug("python-magic not installed — falling back to extension-based MIME")
            ext = Path(filename).suffix.lower().lstrip(".")
            mimes = _EXTENSION_MIME.get(ext)
            return next(iter(mimes)) if mimes else None
        except Exception as exc:
            logger.warning("MIME detection failed: %s", exc)
            return None

    def validate_signature(self, file_content: BinaryIO, extension: str, filename: str) -> None:
        """Verify magic-byte signatures for formats that have them."""
        expected_sigs = _SIGNATURES.get(extension)
        if not expected_sigs:
            return  # No signature to check for this extension

        pos = file_content.tell()
        header = file_content.read(16)
        file_content.seek(pos)

        if not header:
            raise FileValidationError(
                "Cannot read file header for signature verification",
                filename=filename,
                reason="unreadable_header",
            )

        if not any(header.startswith(sig) for sig in expected_sigs):
            raise FileValidationError(
                f"File signature does not match .{extension} format — "
                "the file may be corrupted or misnamed",
                filename=filename,
                reason="signature_mismatch",
            )

    def validate(
        self,
        file_content: BinaryIO,
        filename: str,
        size_bytes: int,
    ) -> ValidationResult:
        """Run all validation checks and return a ValidationResult."""
        errors: List[str] = []
        sanitized = filename
        ext = ""
        mime: Optional[str] = None

        try:
            sanitized = self.validate_filename(filename)
        except FileValidationError as exc:
            errors.append(str(exc))
            return ValidationResult(
                is_valid=False, filename=filename, extension="", size_bytes=size_bytes, errors=errors
            )

        try:
            ext = self.validate_extension(sanitized)
        except FileValidationError as exc:
            errors.append(str(exc))

        try:
            self.validate_size(size_bytes, sanitized)
        except FileValidationError as exc:
            errors.append(str(exc))

        if not errors:
            try:
                self.validate_signature(file_content, ext, sanitized)
            except FileValidationError as exc:
                errors.append(str(exc))

            mime = self.detect_mime_type(file_content, sanitized)
            # Cross-check MIME against extension expectations
            if mime and ext in _EXTENSION_MIME:
                expected = _EXTENSION_MIME[ext]
                if mime not in expected and not mime.startswith("text/"):
                    logger.warning(
                        "MIME mismatch: extension=%s detected_mime=%s filename=%s",
                        ext, mime, sanitized,
                    )

        return ValidationResult(
            is_valid=len(errors) == 0,
            filename=sanitized,
            extension=ext,
            size_bytes=size_bytes,
            mime_type=mime,
            errors=errors,
        )
