"""
Metadata extractor for Egyptian legal documents.
Extracts document-level metadata from heterogeneous legal texts:
- Cassation rulings (appeal number, year, technical office, date)
- Statutory laws (law name, number, year, issuing authority)
- Legal petitions (lawyer name, court, case type)

This metadata enriches the document for filtering, faceted search, and provenance.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.preprocessing.patterns import (
    CASSATION_HEADER_RE,
    LAW_NUMBER_YEAR_RE,
    DIGIT_MAP,
)


@dataclass(slots=True)
class DocumentMetadata:
    """
    Comprehensive metadata for a legal document.
    
    Attributes:
        doc_type: Type of document (cassation_ruling, law, petition, mixed)
        title: Document title or subject
        law_number: Number of the law (if applicable)
        law_year: Year of the law (if applicable)
        appeal_number: Cassation appeal number (if applicable)
        appeal_year: Cassation appeal year (if applicable)
        technical_office: Technical office number (if applicable)
        page_number: Page number in technical office compilation
        date: Document date (ISO format string)
        court: Court name
        issuing_authority: Authority that issued the law/decision
        categories: List of legal categories/tags
        amending_laws: List of amending law references (number/year)
        extra: Additional key-value pairs for extensibility
    """
    doc_type: str = "unknown"
    title: Optional[str] = None
    law_number: Optional[str] = None
    law_year: Optional[str] = None
    appeal_number: Optional[str] = None
    appeal_year: Optional[str] = None
    technical_office: Optional[str] = None
    page_number: Optional[str] = None
    date: Optional[str] = None
    court: Optional[str] = None
    issuing_authority: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    amending_laws: List[str] = field(default_factory=list)
    extra: Dict[str, str] = field(default_factory=dict)


class MetadataExtractor:
    """
    Stateless metadata extractor for diverse Egyptian legal documents.
    Uses pattern matching and heuristics to identify document type and extract metadata.
    """

    def __init__(self):
        # Court name patterns
        self.court_patterns = [
            re.compile(r"محكمة\s+(النقض|الاستئناف|الابتدائية|الجزئية|العليا|الدستورية)", re.IGNORECASE),
            re.compile(r"(Court|Cour)\s+of\s+(Cassation|Appeal)", re.IGNORECASE),
        ]
        
        # Authority patterns
        self.authority_patterns = [
            re.compile(r"(رئيس\s+الجمهورية|مجلس\s+الوزراء|الوزير\s+الأول)", re.IGNORECASE),
            re.compile(r"(President|Prime\s+Minister|Council\s+of\s+Ministers)", re.IGNORECASE),
        ]
        
        # Date patterns (Arabic and English)
        self.date_patterns = [
            re.compile(r"بتاريخ\s+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"),
            re.compile(r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+م"),
            re.compile(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})", re.IGNORECASE),
        ]

    def extract(self, text: str, categories: Optional[List[str]] = None) -> DocumentMetadata:
        """
        Extracts metadata from legal document text.
        
        Args:
            text: Cleaned document text (first 2000 chars usually sufficient)
            categories: Optional list of categories from external source (e.g., CSV)
            
        Returns:
            DocumentMetadata object with extracted fields
        """
        if not text:
            return DocumentMetadata()
            
        # Sample only the first 2000 chars for efficiency (metadata is usually at the top)
        sample_text = text[:2000]
        
        # Detect document type
        doc_type = self._detect_document_type(sample_text)
        
        # Initialize metadata
        metadata = DocumentMetadata(doc_type=doc_type, categories=categories or [])
        
        # Extract based on document type
        if doc_type == "cassation_ruling":
            self._extract_cassation_metadata(sample_text, metadata)
        elif doc_type == "law":
            self._extract_law_metadata(sample_text, metadata)
        elif doc_type == "petition":
            self._extract_petition_metadata(sample_text, metadata)
            
        # Extract common metadata (dates, courts, authorities)
        self._extract_common_metadata(sample_text, metadata)
        
        # Extract title (first meaningful line)
        metadata.title = self._extract_title(text)
        
        return metadata

    def _detect_document_type(self, text: str) -> str:
        """Detects the type of legal document."""
        # Check for Cassation ruling indicators
        if CASSATION_HEADER_RE.search(text):
            return "cassation_ruling"
            
        # Check for law indicators
        if LAW_NUMBER_YEAR_RE.search(text) and ("مادة" in text or "قانون" in text):
            return "law"
            
        # Check for petition indicators
        if re.search(r"عريضة|محام|الأستاذ\s+\w+", text):
            return "petition"
            
        return "mixed" if "مادة" in text else "unknown"

    def _extract_cassation_metadata(self, text: str, metadata: DocumentMetadata) -> None:
        """Extracts metadata specific to Cassation rulings."""
        match = CASSATION_HEADER_RE.search(text)
        if match:
            metadata.appeal_number = match.group(1)
            metadata.appeal_year = match.group(2)
            metadata.technical_office = match.group(3)
            metadata.page_number = match.group(4)
            # Normalize date format
            date_str = match.group(5)
            metadata.date = self._normalize_date(date_str)
            metadata.court = "محكمة النقض"

    def _extract_law_metadata(self, text: str, metadata: DocumentMetadata) -> None:
        """Extracts metadata specific to statutory laws."""
        match = LAW_NUMBER_YEAR_RE.search(text)
        if match:
            metadata.law_number = match.group(1)
            metadata.law_year = match.group(2)
            
            # Detect issuing authority
            for pattern in self.authority_patterns:
                auth_match = pattern.search(text[:500])
                if auth_match:
                    metadata.issuing_authority = auth_match.group(0)
                    break

    def _extract_petition_metadata(self, text: str, metadata: DocumentMetadata) -> None:
        """Extracts metadata specific to legal petitions."""
        # Extract lawyer name (الأستاذ ...)
        lawyer_match = re.search(r"الأستاذ\s+([\w\s]+?)(?:\s+محام|\s+مكتب)", text)
        if lawyer_match:
            metadata.extra["lawyer_name"] = lawyer_match.group(1).strip()
            
        # Extract office name (مكتب ...)
        office_match = re.search(r"مكتب\s+([\w\s]+?)(?:\s+للمحاماة)", text)
        if office_match:
            metadata.extra["law_office"] = office_match.group(1).strip()

    def _extract_common_metadata(self, text: str, metadata: DocumentMetadata) -> None:
        """Extracts common metadata fields across all document types."""
        # Extract court if not already set
        if not metadata.court:
            for pattern in self.court_patterns:
                court_match = pattern.search(text)
                if court_match:
                    metadata.court = court_match.group(0)
                    break
                    
        # Extract date if not already set
        if not metadata.date:
            for pattern in self.date_patterns:
                date_match = pattern.search(text)
                if date_match:
                    if len(date_match.groups()) == 3:  # English month format
                        metadata.date = self._normalize_date(f"{date_match.group(1)} {date_match.group(2)}, {date_match.group(3)}")
                    else:
                        metadata.date = self._normalize_date(date_match.group(1))
                    break

    def _extract_title(self, text: str) -> Optional[str]:
        """Extracts the first meaningful line as title."""
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                # Skip lines that are just headers or metadata
                if not re.search(r"^\d+|طعن\s+رقم|مكتب\s+فنى", line):
                    return line
        return None

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalizes date string to ISO format (YYYY-MM-DD)."""
        if not date_str:
            return None
            
        # Try common formats
        formats = [
            "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d",
            "%d %b, %Y", "%b %d, %Y", "%d %B %Y"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
                
        # If all else fails, return original
        return date_str.strip()