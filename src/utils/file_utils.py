"""
Utility functions for safe and standardized file operations.
Critical for handling user uploads and document storage securely.
"""

import hashlib, re
from pathlib import Path

from src.config.settings import settings


def ensure_directory(path: Path) -> Path:
    """
    Create directory and parents if they don't exist.
    Returns the path for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_file_extension(filename:str) -> str:
    """
    Get the lowercase file extension (e.g., '.pdf').
    Returns empty string if no extension.
    """
    return Path(filename).suffix.lower()

def calculate_file_hash(file_path: Path, algorithem: str = "sha256") -> str:
    """
    Calculate the hash of a file.
    Uses Python 3.11's file_digest for efficient, buffered reading.
    Crucial for document deduplication.
    """
    with open(file_path, "rb") as f:
        return hashlib.file_digest(f, algorithem).hexdigest()


def senitize_filename(filename:str) -> str:
    """
    Sanitize a filename to prevent path traversal and remove invalid characters.
    Keeps alphanumeric, dots, hyphens, and underscores.
    """
    # Remove path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    # Remove invalid characters (keep word chars, dots, hyphens)
    filename = re.sub(r"[^\w\-.]", "_", filename)
    # Collapse multiple underscores
    filename = re.sub(r"_+", "_", filename)
    return filename.strip("_")

def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """
    Check if target_path is strictly within base_dir.
    Prevents path traversal attacks (e.g., user uploads '../../etc/passwd').
    """
    try:
        # Resolve to absolute paths to handle symlinks and '..'
        base_resolved = base_dir.resolve()
        target_resolved = target_path.resolve()
        # is_relative_to is available in Python 3.9+
        return target_resolved.is_relative_to(base_resolved)
    except ValueError:
        # Raised on Windows if paths are on different drives
        return False


def get_max_upload_size_bytes() -> int:
    return settings.MAX_UPLOAD_SIZE_BYTES


def get_file_size(file_path: Path) -> int:
    """
    Get the size of a file in bytes.
    Raises FileNotFoundError if the file does not exist.
    """
    return file_path.stat().st_size