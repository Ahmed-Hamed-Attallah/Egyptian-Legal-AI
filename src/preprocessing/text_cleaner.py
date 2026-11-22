"""
Dynamic text cleaning module.
Handles Arabic, English, and mixed-language documents with configurable artifact removal.
"""
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from src.preprocessing.patterns import (
    BIDI_AND_ZERO_WIDTH_RE,
    TATWEEL_RE,
    PDF_STRAY_DIGIT_ARABIC_RE,
    PDF_STRAY_DIGIT_ENGLISH_RE,
    STRAY_QUOTES_RE,
    DANGLING_PUNCTUATION_RE,
    BROKEN_HYPHENATION_RE,
    PAGE_HEADER_FOOTER_RE,
    WHITESPACE_RE,
    SMART_QUOTE_MAP,
)


@dataclass(frozen=True, slots=True)
class CleanedText:
    """Result of text cleaning with provenance tracking."""
    text: str
    offset_map: List[int]


@dataclass
class CleaningConfig:
    """
    Configuration for the text cleaner. 
    Allows dynamic adjustment based on document type (e.g., PDF vs OCR vs Word).
    """
    remove_bidi_and_zero_width: bool = True
    remove_tatweel: bool = True
    fix_pdf_stray_digits: bool = True
    fix_stray_quotes: bool = True
    fix_dangling_punctuation: bool = True
    fix_broken_hyphenation: bool = True
    remove_page_headers_footers: bool = False  # Set to True for raw PDF dumps
    normalize_smart_quotes: bool = True


class TextCleaner:
    """Stateless, configurable text cleaner for diverse legal documents."""

    def __init__(self, config: CleaningConfig | None = None):
        self.config = config or CleaningConfig()

    def clean(self, text: str) -> CleanedText:
        """Cleans the input text and generates an offset map."""
        if not text:
            return CleanedText(text="", offset_map=[])

        offset_map = list(range(len(text)))

        # 1. Invisible & Control Characters
        if self.config.remove_bidi_and_zero_width:
            text, offset_map = self._apply_removal(text, offset_map, BIDI_AND_ZERO_WIDTH_RE)
        if self.config.remove_tatweel:
            text, offset_map = self._apply_removal(text, offset_map, TATWEEL_RE)

        # 2. PDF/OCR Artifacts
        if self.config.fix_pdf_stray_digits:
            text, offset_map = self._apply_substitution(text, offset_map, PDF_STRAY_DIGIT_ARABIC_RE, " ")
            text, offset_map = self._apply_substitution(text, offset_map, PDF_STRAY_DIGIT_ENGLISH_RE, " ")
        
        if self.config.fix_stray_quotes:
            text, offset_map = self._apply_substitution(text, offset_map, STRAY_QUOTES_RE, '"')
            
        if self.config.fix_dangling_punctuation:
            text, offset_map = self._apply_substitution(text, offset_map, DANGLING_PUNCTUATION_RE, r" \1 ")

        if self.config.fix_broken_hyphenation:
            text, offset_map = self._apply_substitution(text, offset_map, BROKEN_HYPHENATION_RE, r"\1\2")

        if self.config.remove_page_headers_footers:
            text, offset_map = self._apply_removal(text, offset_map, PAGE_HEADER_FOOTER_RE)

        # 3. Smart Quotes Normalization (Map-based, requires special handling)
        if self.config.normalize_smart_quotes:
            text, offset_map = self._apply_char_map(text, offset_map, SMART_QUOTE_MAP)

        # 4. Whitespace Normalization (Always runs last)
        text, offset_map = self._normalize_whitespace(text, offset_map)

        return CleanedText(text=text, offset_map=offset_map)

    # --- Internal Helper Methods ---

    def _apply_removal(self, text: str, offset_map: List[int], pattern: re.Pattern) -> Tuple[str, List[int]]:
        new_text, new_offset_map = [], []
        for i, char in enumerate(text):
            if not pattern.match(char):
                new_text.append(char)
                new_offset_map.append(offset_map[i])
        return "".join(new_text), new_offset_map

    def _apply_substitution(self, text: str, offset_map: List[int], pattern: re.Pattern, replacement: str) -> Tuple[str, List[int]]:
        new_text, new_offset_map = [], []
        last_end = 0
        
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            new_text.append(text[last_end:start])
            new_offset_map.extend(offset_map[last_end:start])
            
            # Handle regex backreferences in replacement (e.g., \1\2)
            resolved_replacement = match.expand(replacement) if "\\" in replacement else replacement
            
            new_text.append(resolved_replacement)
            new_offset_map.extend([offset_map[start]] * len(resolved_replacement))
            last_end = end
            
        new_text.append(text[last_end:])
        new_offset_map.extend(offset_map[last_end:])
        return "".join(new_text), new_offset_map

    def _apply_char_map(self, text: str, offset_map: List[int], char_map: dict[str, str]) -> Tuple[str, List[int]]:
        """Applies a 1-to-1 or 1-to-N character map while maintaining offsets."""
        new_text, new_offset_map = [], []
        for i, char in enumerate(text):
            replacement = char_map.get(char, char)
            new_text.append(replacement)
            new_offset_map.extend([offset_map[i]] * len(replacement))
        return "".join(new_text), new_offset_map

    def _normalize_whitespace(self, text: str, offset_map: List[int]) -> Tuple[str, List[int]]:
        new_text, new_offset_map = [], []
        prev_was_space = False
        
        for i, char in enumerate(text):
            if char.isspace():
                if not prev_was_space:
                    new_text.append(" ")
                    new_offset_map.append(offset_map[i])
                prev_was_space = True
            else:
                new_text.append(char)
                new_offset_map.append(offset_map[i])
                prev_was_space = False
                
        return "".join(new_text).strip(), new_offset_map