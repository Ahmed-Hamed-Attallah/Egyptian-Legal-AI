"""Domain models for Egyptian legal document structure.

Models the hierarchical structure of Egyptian legal documents:
Laws → Books → Chapters → Sections → Articles → Paragraphs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LegalUnitType(str, Enum):
    """Types of structural units in Egyptian legal documents."""

    LAW = "law"
    CONSTITUTION = "constitution"
    BOOK = "book"
    CHAPTER = "chapter"
    SECTION = "section"
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    CLAUSE = "clause"
    ANNEX = "annex"
    PREAMBLE = "preamble"


# Arabic numeral mapping
_ARABIC_DIGITS = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
}

_TO_ARABIC_NUMERALS = str.maketrans(_ARABIC_DIGITS)


def _to_int(value: str) -> Optional[int]:
    """Convert Arabic-Indic or Western digits to int."""
    try:
        translated = value.translate(_TO_ARABIC_NUMERALS)
        return int(translated)
    except (ValueError, TypeError):
        return None


@dataclass
class LegalUnit:
    """A single structural unit within a legal document."""

    unit_type: LegalUnitType
    number: Optional[int] = None
    title: Optional[str] = None
    content: str = ""
    children: list["LegalUnit"] = field(default_factory=list)
    parent: Optional["LegalUnit"] = None
    start_offset: int = 0
    end_offset: int = 0

    def add_child(self, child: "LegalUnit") -> None:
        """Add a child unit to this unit."""
        child.parent = self
        self.children.append(child)

    def get_articles(self) -> list["LegalUnit"]:
        """Recursively collect all articles under this unit."""
        results: list[LegalUnit] = []
        if self.unit_type == LegalUnitType.ARTICLE:
            results.append(self)
        for child in self.children:
            results.extend(child.get_articles())
        return results


@dataclass
class LegalStructure:
    """Parsed hierarchical structure of a legal document."""

    root: Optional[LegalUnit] = None
    units: list[LegalUnit] = field(default_factory=list)
    law_number: Optional[int] = None
    law_year: Optional[int] = None
    law_title: Optional[str] = None

    def get_articles(self) -> list[LegalUnit]:
        """Get all article-level units."""
        if self.root is None:
            return []
        return self.root.get_articles()

    def get_units_by_type(self, unit_type: LegalUnitType) -> list[LegalUnit]:
        """Get all units of a specific type."""
        return [u for u in self.units if u.unit_type == unit_type]

    def get_books(self) -> list[LegalUnit]:
        """Get all book-level units."""
        return self.get_units_by_type(LegalUnitType.BOOK)

    def get_chapters(self) -> list[LegalUnit]:
        """Get all chapter-level units."""
        return self.get_units_by_type(LegalUnitType.CHAPTER)

    def get_sections(self) -> list[LegalUnit]:
        """Get all section-level units."""
        return self.get_units_by_type(LegalUnitType.SECTION)


class LegalStructureParser:
    """Parses Egyptian legal document text into a structured hierarchy."""

    # Patterns for law identification
    _LAW_PATTERN = re.compile(
        r"قانون\s+رقم\s+(\d+|[٠-٩]+)\s+لسنة\s+(\d+|[٠-٩]+)",
        re.IGNORECASE,
    )

    # Patterns for structural units
    _BOOK_PATTERN = re.compile(
        r"^(?:الكتاب|كتاب)\s+(?:ال)?(?:أول|ثان|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر|\d+|[٠-٩]+)\s*[:\-–]?\s*(.*)",
        re.MULTILINE,
    )
    _CHAPTER_PATTERN = re.compile(
        r"^(?:الباب|باب)\s+(?:ال)?(?:أول|ثان|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر|\d+|[٠-٩]+)\s*[:\-–]?\s*(.*)",
        re.MULTILINE,
    )
    _SECTION_PATTERN = re.compile(
        r"^(?:الفصل|فصل)\s+(?:ال)?(?:أول|ثان|ثالث|رابع|خامس|سادس|سابع|ثامن|تاسع|عاشر|\d+|[٠-٩]+)\s*[:\-–]?\s*(.*)",
        re.MULTILINE,
    )
    _ARTICLE_PATTERN = re.compile(
        r"^(?:المادة|مادة)\s+(\d+|[٠-٩]+)\s*[:\-–]?\s*(.*)",
        re.MULTILINE,
    )

    def parse(self, text: str) -> LegalStructure:
        """Parse legal text into a LegalStructure."""
        structure = LegalStructure()

        if not text or not text.strip():
            return structure

        # Extract law number and year
        self._extract_law_metadata(text, structure)

        # Build the structural hierarchy
        root = LegalUnit(unit_type=LegalUnitType.LAW, content=text)
        structure.root = root
        structure.units.append(root)

        self._parse_structure(text, root, structure)

        return structure

    def _extract_law_metadata(self, text: str, structure: LegalStructure) -> None:
        """Extract law number and year from text."""
        match = self._LAW_PATTERN.search(text)
        if match:
            num = _to_int(match.group(1))
            year = _to_int(match.group(2))
            structure.law_number = num
            structure.law_year = year

    def _parse_structure(
        self, text: str, root: LegalUnit, structure: LegalStructure
    ) -> None:
        """Parse structural units from text and attach to root."""
        # Find all structural markers with positions
        markers: list[tuple[int, int, LegalUnitType, Optional[int], str]] = []

        for match in self._BOOK_PATTERN.finditer(text):
            num = self._extract_ordinal_number(match.group(0))
            markers.append((match.start(), match.end(), LegalUnitType.BOOK, num, match.group(1).strip()))

        for match in self._CHAPTER_PATTERN.finditer(text):
            num = self._extract_ordinal_number(match.group(0))
            markers.append((match.start(), match.end(), LegalUnitType.CHAPTER, num, match.group(1).strip()))

        for match in self._SECTION_PATTERN.finditer(text):
            num = self._extract_ordinal_number(match.group(0))
            markers.append((match.start(), match.end(), LegalUnitType.SECTION, num, match.group(1).strip()))

        for match in self._ARTICLE_PATTERN.finditer(text):
            num = _to_int(match.group(1))
            markers.append((match.start(), match.end(), LegalUnitType.ARTICLE, num, match.group(2).strip()))

        # Sort by position
        markers.sort(key=lambda m: m[0])

        # Build hierarchy
        current_parent = root
        for start, end, unit_type, num, title in markers:
            unit = LegalUnit(
                unit_type=unit_type,
                number=num,
                title=title if title else None,
                start_offset=start,
                end_offset=end,
            )
            structure.units.append(unit)

            # Determine parent based on hierarchy level
            if unit_type == LegalUnitType.BOOK:
                current_parent = root
            elif unit_type == LegalUnitType.CHAPTER:
                # Chapter belongs to current book or root
                pass
            elif unit_type == LegalUnitType.SECTION:
                # Section belongs to current chapter or book
                pass

            current_parent.add_child(unit)

    def _extract_ordinal_number(self, text: str) -> Optional[int]:
        """Extract a numeric value from ordinal text like 'الأول' or '1'."""
        ordinals = {
            "أول": 1, "ثان": 2, "ثالث": 3, "رابع": 4,
            "خامس": 5, "سادس": 6, "سابع": 7, "ثامن": 8,
            "تاسع": 9, "عاشر": 10,
        }
        for word, num in ordinals.items():
            if word in text:
                return num
        # Try numeric extraction
        match = re.search(r"(\d+|[٠-٩]+)", text)
        if match:
            return _to_int(match.group(1))
        return None