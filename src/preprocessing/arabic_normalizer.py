"""
Arabic text normalization module.
Produces a dual-text representation: 
1. text_original: Verbatim legal fidelity (for LLM context and UI citations).
2. text_normalized: Aggressively folded text (for BM25 and dense retrieval indexing).

Includes an offset map to resolve normalized search hits back to the original text.
"""
from dataclasses import dataclass
from typing import Tuple

from src.preprocessing.patterns import (
    HAMZA_MAP, 
    TA_MARBUTA_MAP, 
    DIGIT_MAP, 
    DIACRITICS_RE
)


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """
    Result of Arabic normalization.
    
    Attributes:
        text_original: The cleaned text, preserving legal verbatim fidelity.
        text_normalized: The folded text, optimized for search and retrieval.
        offset_map: Tuple where offset_map[i] is the index in text_original 
                    corresponding to index i in text_normalized.
    """
    text_original: str
    text_normalized: str
    offset_map: Tuple[int, ...]


@dataclass
class NormalizationConfig:
    """
    Configuration for normalization aggressiveness.
    Tailored for legal text where semantic precision is critical.
    """
    fold_hamza: bool = True          # أ, إ, آ, ؤ, ئ -> ا, و, ي
    fold_ta_marbuta: bool = False    # ة -> ه (Default OFF for legal precision)
    fold_digits: bool = True         # ٠-٩, ۰-۹ -> 0-9
    strip_diacritics: bool = True    # Remove Tashkeel/Harakat


class ArabicNormalizer:
    """
    Stateless Arabic normalizer with offset tracking.
    Ensures perfect symmetry between document indexing and query processing.
    """

    def __init__(self, config: NormalizationConfig | None = None):
        self.config = config or NormalizationConfig()

    def normalize(self, text: str) -> NormalizationResult:
        """
        Normalizes the input text and generates an offset map.
        
        Args:
            text: The cleaned text from TextCleaner.
            
        Returns:
            NormalizationResult containing original, normalized, and offset map.
        """
        if not text:
            return NormalizationResult(text_original="", text_normalized="", offset_map=())

        normalized_chars = []
        offset_map = []

        # Single-pass O(N) normalization and offset mapping
        for i, char in enumerate(text):
            # 1. Complete removal (Diacritics)
            if self.config.strip_diacritics and DIACRITICS_RE.match(char):
                continue  # Skip character, do not advance normalized index

            # 2. Character replacement/folding
            replacement = char
            if self.config.fold_hamza and char in HAMZA_MAP:
                replacement = HAMZA_MAP[char]
            elif self.config.fold_ta_marbuta and char in TA_MARBUTA_MAP:
                replacement = TA_MARBUTA_MAP[char]
            elif self.config.fold_digits and char in DIGIT_MAP:
                replacement = DIGIT_MAP[char]

            # 3. Append replacement and map offset
            # Note: replacement is always length 1 in our maps, but logic supports expansion
            normalized_chars.append(replacement)
            offset_map.extend([i] * len(replacement))

        return NormalizationResult(
            text_original=text,
            text_normalized="".join(normalized_chars),
            offset_map=tuple(offset_map)
        )

    def normalize_query(self, query: str) -> str:
        """
        Normalizes a search query using the EXACT same rules as document indexing.
        Crucial for search symmetry.
        
        Args:
            query: Raw user search query.
            
        Returns:
            Normalized query string ready for BM25/Dense retrieval.
        """
        result = self.normalize(query)
        return result.text_normalized