"""
Single source of truth for regex patterns and character maps.
Designed to handle Arabic, English, mixed-language, and diverse document formats.
"""
import re

# ==============================================================================
# 1. Character Maps (Used for normalization and ligature expansion)
# ==============================================================================

# Digits: Arabic-Indic and Eastern Arabic-Indic to Western Arabic
DIGIT_MAP: dict[str, str] = {
    "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3", "\u0664": "4",
    "\u0665": "5", "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
    "\u06F0": "0", "\u06F1": "1", "\u06F2": "2", "\u06F3": "3", "\u06F4": "4",
    "\u06F5": "5", "\u06F6": "6", "\u06F7": "7", "\u06F8": "8", "\u06F9": "9",
}

# Arabic Hamza and Ta Marbuta folding (For normalized view only)
HAMZA_MAP: dict[str, str] = {
    "\u0622": "\u0627", "\u0623": "\u0627", "\u0624": "\u0648",
    "\u0625": "\u0627", "\u0626": "\u064A",
}
TA_MARBUTA_MAP: dict[str, str] = {"\u0629": "\u0647"}

# Ligature expansion (Crucial for English OCR and Arabic PDF extraction)
LIGATURE_MAP: dict[str, str] = {
    "\uFB01": "fi", "\uFB02": "fl", "\u0132": "IJ", "\u0133": "ij",
    "\u0152": "OE", "\u0153": "oe", "\u00C6": "AE", "\u00E6": "ae",
    "\uFEFB": "\u0644\u0627", # ﻻ -> لا
}

# Smart quotes to standard quotes (Fixes OCR and Word doc artifacts)
SMART_QUOTE_MAP: dict[str, str] = {
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
    "\u00AB": '"', "\u00BB": '"',
}

# ==============================================================================
# 2. Cleaning Patterns (Encoding hygiene, PDF/OCR artifacts)
# ==============================================================================

# Invisible & Control Characters
BIDI_AND_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF\u00AD]")
TATWEEL_RE = re.compile(r"\u0640")
DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670]") # Tashkeel/Harakat

# PDF/OCR Artifacts: Stray digits/letters between words
PDF_STRAY_DIGIT_ARABIC_RE = re.compile(r"(?<=[\u0600-\u06FF])\d(?=[\u0600-\u06FF])")
PDF_STRAY_DIGIT_ENGLISH_RE = re.compile(r"(?<=[a-zA-Z])\d(?=[a-zA-Z])")

# PDF/OCR Artifacts: Stray punctuation and quotes
STRAY_QUOTES_RE = re.compile(r'"{2,}|\'{2,}')
DANGLING_PUNCTUATION_RE = re.compile(r"\s+([.,;:!?])\s+") # Fixes "word . word" -> "word. word"

# PDF Artifacts: Broken English hyphenation at line breaks (e.g., "pro-\ncessing")
BROKEN_HYPHENATION_RE = re.compile(r"(\w)-\s*\n\s*(\w)")

# PDF Artifacts: Page numbers, headers, footers (e.g., "- 12 -", "Page 1 of 10")
PAGE_HEADER_FOOTER_RE = re.compile(r"(?:^|\n)\s*(?:-?\s*\d+\s*-?|Page\s+\d+\s+of\s+\d+|صفحة\s+\d+)\s*(?:\n|$)", re.IGNORECASE)

# Whitespace normalization
WHITESPACE_RE = re.compile(r"\s+")

# ==============================================================================
# 3. Arabic Legal Structure Patterns
# ==============================================================================

# Articles, Chapters, Sections, Paragraphs
ARABIC_ARTICLE_RE = re.compile(
    r"(?:المادة|مادة)\s*(?:\[|\()?\s*(\d+|[أ-ي]+)\s*(?:\]|\))?\s*[-:]?", re.IGNORECASE
)
ARABIC_CHAPTER_RE = re.compile(r"(?:الباب|باب)\s*(?:\[|\()?\s*(\d+|[أ-ي]+)\s*(?:\]|\))?", re.IGNORECASE)
ARABIC_SECTION_RE = re.compile(r"(?:الفصل|فصل)\s*(?:\[|\()?\s*(\d+|[أ-ي]+)\s*(?:\]|\))?", re.IGNORECASE)

# Cassation Court Ruling Headers
CASSATION_HEADER_RE = re.compile(
    r"طعن\s+رقم\s+(\d+)\s+لسنة\s+(\d+)\s+مكتب\s+فنى\s+(\d+)\s+صفحة\s+رقم\s+(\d+)\s+بتاريخ\s+([\d\-/]+)"
)

# Amendment References
ARABIC_AMENDMENT_RE = re.compile(
    r"(?:عدلت|تعديل|يُلغى|يُستبدل|أُضيف)\s+(?:بالقانون|بمرسوم|بقرار)\s+(?:رقم|)\s*(\d+)\s+(?:لسنة|لعام|)\s*(\d{4})",
    re.IGNORECASE
)

# Robust pattern for Egyptian laws, decrees, and decisions.
# Handles variations like:
# - القانون رقم 4 لسنة 1994
# - مرسوم بقانون رقم 188 لسنة 1952
# - قرار رئيس مجلس الوزراء رقم 338 لسنة 1995
# - قانون رقم 27 لسنة 1994
# - القانون رقم 37 سنة 1960
LAW_NUMBER_YEAR_RE = re.compile(
    r"(?:ال)?(?:قانون|مرسوم\s+(?:بقانون|تشريعي)?|قرار\s+(?:رئيس\s+(?:الجمهورية|مجلس\s+الوزراء)|وزاري)?)\s*"
    r"(?:رقم|)\s*"
    r"(\d+)\s*"
    r"(?:لسنة|لعام|سنة|عام|)\s*"
    r"(\d{2,4})"
)

# ==============================================================================
# 4. English & International Legal Structure Patterns
# ==============================================================================

# Sections, Articles, Clauses (Common Law / International)
ENGLISH_SECTION_RE = re.compile(r"(?:Section|§|Sec\.)\s*(\d+[a-zA-Z]?(?:\.\d+)?)", re.IGNORECASE)
ENGLISH_ARTICLE_RE = re.compile(r"(?:Article|Art\.)\s*(\d+[a-zA-Z]?(?:\.\d+)?)", re.IGNORECASE)
ENGLISH_CLAUSE_RE = re.compile(r"(?:Clause|Cl\.)\s*(\d+[a-zA-Z]?(?:\.\d+)?)", re.IGNORECASE)

# Case Citations (e.g., Roe v. Wade, 410 U.S. 113)
ENGLISH_CASE_CITATION_RE = re.compile(
    r"[A-Z][a-zA-Z\s\.]+v\.\s*[A-Z][a-zA-Z\s\.]+,\s*\d+\s+(?:U\.S\.|F\.\d+d?|S\.Ct\.)\s+\d+",
    re.IGNORECASE
)

# Statutory References (e.g., 42 U.S.C. § 1983)
ENGLISH_STATUTE_RE = re.compile(
    r"\d+\s+(?:U\.S\.C\.|C\.F\.R\.)\s+(?:§|Section)\s*\d+", re.IGNORECASE
)

# ==============================================================================
# 5. Universal Document Patterns (Dates, Numbers, Currencies)
# ==============================================================================

# Dates (DD/MM/YYYY, DD-MM-YYYY, Month DD, YYYY)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
    re.IGNORECASE
)

# Currencies (EGP, USD, £, $)
CURRENCY_RE = re.compile(r"(?:EGP|USD|GBP|EUR|ج\.م|دولار|جنيه)\s*[\d,]+(?:\.\d+)?|[\$£€]\s*[\d,]+(?:\.\d+)?", re.IGNORECASE)


# Article detection regex
ARTICLE_PATTERN = re.compile(
    r"(?m)^[ \t]*"
    r"(?:المادة|مادة)"
    r"[ \t]*(?:رقم[ \t]*)?"
    r"[\(\[（]?[ \t]*"
    r"[0-9٠-٩۰-۹]+"
    r"[ \t]*[\)\]）]?"
    r"[ \t]*(?:[-:ـ.]?[ \t]*)"
)

DANGLING_PUNCTUATION_RE = re.compile(r"\s+([.,;:!?،؛؟۔])\s+")