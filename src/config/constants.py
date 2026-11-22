"""
Project-wide immutable constants.
These values are domain-specific (Egyptian Legal Context) and do not change based on environment.
"""
import re
from typing import Final

# ==========================================
# File Handling
# ==========================================
SUPPORTED_EXTENSIONS: Final[set[str]] = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE_BYTES: Final[int] = 50 * 1024 * 1024  # 50MB default limit

# ==========================================
# Legal Document Structure (Critical for Chunking)
# ==========================================
# Regex to match Egyptian legal article headers.
# Matches: "مادة 1", "المادة الأولى", "مادة رقم 5", etc.
# Used by the 'legal' chunking strategy to preserve article boundaries.
LEGAL_ARTICLE_PATTERN: Final[re.Pattern] = re.compile(
    r"(?:^|\n)\s*(?:مادة|المادة)\s*(?:رقم\s*)?[\u0660-\u0669\u0621-\u064A0-9]+",
    re.IGNORECASE | re.UNICODE
)

# ==========================================
# Arabic Text Normalization
# ==========================================
# Mapping to unify character variations for better retrieval recall.
ARABIC_NORMALIZATION_MAP: Final[dict[str, str]] = {
    # Alef forms -> Standard Alef (ا)
    "\u0622": "\u0627",  # آ -> ا
    "\u0623": "\u0627",  # أ -> ا
    "\u0625": "\u0627",  # إ -> ا
    "\u0671": "\u0627",  # ٱ -> ا
    
    # Ya forms -> Standard Ya (ي)
    "\u0649": "\u064a",  # ى -> ي
    
    # Ta Marbuta -> Ha (ه)
    "\u0629": "\u0647",  # ة -> ه
}

# Regex to remove Tashkeel (diacritics) which are often inconsistent in legal texts.
ARABIC_DIACRITICS_PATTERN: Final[re.Pattern] = re.compile(r"[\u064B-\u0652]")

# ==========================================
# Date Formats
# ==========================================
# Common formats found in Egyptian legal documents and headers.
LEGAL_DATE_FORMATS: Final[list[str]] = [
    "%Y-%m-%d",      # 2023-10-31
    "%d/%m/%Y",      # 31/10/2023
    "%d-%m-%Y",      # 31-10-2023
    "%Y/%m/%d",      # 2023/10/31
    "%d %B %Y",      # 31 October 2023
    "%d %b %Y",      # 31 Oct 2023
]

# ==========================================
# Document Processing Statuses
# ==========================================
class DocumentStatus:
    """Standard states for the document ingestion pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

# ==========================================
# Chunking Defaults
# ==========================================
# Fallback values if settings are not loaded or for testing.
DEFAULT_CHUNK_SIZE: Final[int] = 800
DEFAULT_CHUNK_OVERLAP: Final[int] = 120