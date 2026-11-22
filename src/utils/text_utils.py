"""
Utility functions for Arabic text normalization and cleaning.
Critical for improving retrieval recall in the Legal RAG pipeline.
"""
import re
import unicodedata

from src.config.constants import (
    ARABIC_DIACRITICS_PATTERN,
    ARABIC_NORMALIZATION_MAP,
)


def remove_diacritics(text: str) -> str:
    """
    Remove Arabic diacritics (Tashkeel: Fatha, Damma, Kasra, Sukun, etc.).
    Legal texts often have inconsistent diacritics; removing them improves matching.
    """
    return ARABIC_DIACRITICS_PATTERN.sub("", text)


def normalize_arabic_characters(text: str) -> str:
    """
    Unify variations of Arabic letters to their standard forms.
    - Alef forms (آ, أ, إ, ٱ) -> ا
    - Ya forms (ى) -> ي
    - Ta Marbuta (ة) -> ه
    """
    # Using str.translate is significantly faster than multiple replace() calls
    return text.translate(str.maketrans(ARABIC_NORMALIZATION_MAP))


def clean_whitespace(text: str) -> str:
    """
    Clean whitespace and remove invisible/zero-width characters.
    - Removes Zero Width Non-Joiner (ZWNJ), Zero Width Space (ZWSP), etc.
    - Collapses multiple spaces/newlines into a single space.
    - Strips leading/trailing whitespace.
    """
    # Remove zero-width and invisible characters
    # \u200b: Zero Width Space
    # \u200c: Zero Width Non-Joiner
    # \u200d: Zero Width Joiner
    # \u200e, \u200f: Left-to-Right / Right-to-Left Marks
    # \u061c: Arabic Letter Mark
    invisible_chars_pattern = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u061c]")
    text = invisible_chars_pattern.sub("", text)
    
    # Normalize all whitespace (spaces, tabs, newlines) to a single space
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def ArabicNormalizer(text: str) -> str:
    """
    Main normalization pipeline for Arabic legal text.
    Applies all cleaning and normalization steps in the correct order.
    
    Order of operations:
    1. Clean whitespace and invisible chars (fixes PDF extraction artifacts)
    2. Remove diacritics (standardizes Tashkeel)
    3. Normalize character variations (unifies Alef, Ya, Ta Marbuta)
    """
    if not text:
        return ""
    
    text = clean_whitespace(text)
    text = remove_diacritics(text)
    text = normalize_arabic_characters(text)
    
    return text


def safe_decode(text: str) -> str:
    """
    Safely decode text to UTF-8, replacing invalid bytes with the Unicode replacement character.
    This prevents crashes when processing corrupted or non-standard text files.
    """
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    return text

def count_arabic_characters(text: str) -> int:
    """
    Count the number of Arabic characters in the text.
    Useful for determining if a document is primarily Arabic or mixed-language.
    """
    return sum(1 for char in text if "\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F" or "\u08A0" <= char <= "\u08FF")


def is_arabic_text(text: str, threshold: float = 0.5) -> bool:
    """
    Determine if the text is primarily Arabic based on character count.
    Returns True if the proportion of Arabic characters exceeds the threshold.
    
    Args:
        text (str): The input text to analyze.
        threshold (float): The minimum proportion of Arabic characters required to classify as Arabic.
        
    Returns:
        bool: True if the text is primarily Arabic, False otherwise.
    """
    if not text:
        return False
    
    arabic_count = count_arabic_characters(text)
    total_count = len(text)
    
    if total_count == 0:
        return False
    
    arabic_ratio = arabic_count / total_count
    return arabic_ratio >= threshold