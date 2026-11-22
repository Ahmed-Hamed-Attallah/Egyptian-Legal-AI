"""
Chunk quality filter for Egyptian legal documents.

Detects and drops low-quality chunks BEFORE they are embedded and indexed:
- PDF UI garbage (e.g., "رجوع للفهرس" repeated)
- Mojibake (e.g., مؠخالفةخ — broken Arabic glyph encoding)
- Repetitive text (PDF bookmarks / TOC loops)
- Glued quotes (e.g., 'لكتاب instead of الكتاب)
- Empty or too-short chunks

The filter is pure (no side effects), stateless, and returns a quality
report so the caller can log / persist the drop reason per chunk.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.domain.chunks import Chunk

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Signatures of low-quality chunks
# -----------------------------------------------------------------------------

# PDF navigation / UI labels (Arabic)
_PDF_UI_PATTERNS = [
    r"رجوع\s*للفهرس",
    r"اضغط\s*للخروج",
    r"اضغط\s*للرجوع",
    r"الصفحة\s*(التالية|السابقة|الأولى|الأخيرة)",
    r"قائمة\s*(البحث|المحتويات)",
    r"اذهب\s*الى\s*صفحة",
    r"بحث\s*متقدم",
    r"الفهرس\s*الرئيسي",
    r"محتويات\s*الكتاب",
    r"اضغط\s*على\s*الرابط",
    r"للرجوع\s*للقائمة",
]
PDF_UI_REGEX = re.compile("|".join(_PDF_UI_PATTERNS), re.IGNORECASE)

# Mojibake characters: impossible in modern Egyptian legal text.
# Their presence is a signature of a broken PDF ToUnicode CMap.
SUSPECT_CHARS = set("؀؁؂؃؄ؠؽؾؿ۝۞﷽")

# Apostrophe / quote glued between Arabic letters (e.g., 'لكتاب → الكتاب)
GLUED_QUOTE_REGEX = re.compile(r"[\u0621-\u064A]['\"][\u0621-\u064A]")

# Whitespace normalization for uniqueness analysis
_WS_RE = re.compile(r"\s+")

# Rejection thresholds
MIN_CHUNK_CHARS = 60           # Too short to be a legal provision
MAX_PDF_UI_SHARE = 0.10        # >10% UI keywords → PDF garbage
MIN_UNIQUE_WORD_RATIO = 0.15   # <15% unique words → repetitive loop
MAX_MOJIBAKE_RATIO = 0.005     # >0.5% impossible chars → mojibake
MAX_GLUED_RATIO = 0.01         # >1% glued quotes → broken extraction


# -----------------------------------------------------------------------------
# Quality report
# -----------------------------------------------------------------------------

class DropReason:
    EMPTY = "empty"
    TOO_SHORT = "too_short"
    PDF_UI_GARBAGE = "pdf_ui_garbage"
    REPETITIVE_LOOP = "repetitive_loop"
    MOJIBAKE = "mojibake"
    GLUED_QUOTES = "glued_quotes"


@dataclass
class QualityReport:
    """Aggregate statistics for a quality-filter pass."""

    total_input: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    by_reason: dict = field(default_factory=dict)
    rejected_samples: List[dict] = field(default_factory=list)

    def add_rejected(self, chunk: Chunk, reason: str, sample: str) -> None:
        self.total_rejected += 1
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1
        if len(self.rejected_samples) < 5:
            self.rejected_samples.append({
                "reason": reason,
                "document_id": str(chunk.document_id),
                "chunk_index": chunk.chunk_index,
                "sample": sample[:200],
            })

    def finalize(self, accepted: int) -> "QualityReport":
        self.total_accepted = accepted
        self.total_input = self.total_accepted + self.total_rejected
        if self.total_rejected > 0:
            logger.info(
                "[ChunkQualityFilter] accepted=%d rejected=%d reasons=%s",
                self.total_accepted,
                self.total_rejected,
                dict(self.by_reason),
            )
        return self


# -----------------------------------------------------------------------------
# Core detector
# -----------------------------------------------------------------------------

def _detect(text: str) -> Optional[str]:
    """Return a DropReason if the chunk is low quality, else None."""
    if not text:
        return DropReason.EMPTY

    stripped = text.strip()
    if len(stripped) < MIN_CHUNK_CHARS:
        return DropReason.TOO_SHORT

    total = len(stripped)

    # 1. Mojibake (impossible Arabic chars)
    suspect_count = sum(1 for ch in stripped if ch in SUSPECT_CHARS)
    if suspect_count / total > MAX_MOJIBAKE_RATIO:
        return DropReason.MOJIBAKE

    # 2. Glued quotes ('لكتاب instead of الكتاب)
    glued_count = len(GLUED_QUOTE_REGEX.findall(stripped))
    if glued_count / total > MAX_GLUED_RATIO:
        return DropReason.GLUED_QUOTES

    # 3. PDF UI garbage (short text dominated by navigation labels)
    if PDF_UI_REGEX.search(stripped) and len(stripped) < 250:
        return DropReason.PDF_UI_GARBAGE

    # 4. Repetitive loops ("رجوع للفهرس رجوع للفهرس ...")
    words = _WS_RE.split(stripped)
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < MIN_UNIQUE_WORD_RATIO:
            return DropReason.REPETITIVE_LOOP

    return None


def is_garbage_chunk(text: str) -> bool:
    """Public helper used by tests and retrieval-time filters."""
    return _detect(text or "") is not None


# -----------------------------------------------------------------------------
# Filter entry point
# -----------------------------------------------------------------------------

def filter_chunks(chunks: List[Chunk]) -> tuple[List[Chunk], QualityReport]:
    """
    Drop low-quality chunks and return (clean_chunks, report).

    The original list is never mutated.
    """
    report = QualityReport()
    clean: List[Chunk] = []

    for chunk in chunks:
        content = chunk.content or ""
        reason = _detect(content)
        if reason is None:
            clean.append(chunk)
        else:
            report.add_rejected(chunk, reason, content)

    return clean, report.finalize(len(clean))