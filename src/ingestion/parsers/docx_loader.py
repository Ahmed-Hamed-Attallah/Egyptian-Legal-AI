# src/ingestion/parsers/docx_loader.py
"""
DOCX Document Loader.
Extracts raw text and metadata from Microsoft Word (.docx) files.
Handles both paragraphs and tables while preserving their original reading order.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph
from src.ingestion.models import ParsedResult
from src.ingestion.parsers.base import BaseDocumentLoader

logger = logging.getLogger(__name__)

class DOCXLoader(BaseDocumentLoader):
    """
    Loader for DOCX files.
    Returns a single ParsedResult object containing the full text of the file.
    """
    def load(self, file_path: Path) -> List[ParsedResult]:
        """
        Loads a DOCX file and returns a list containing a single ParsedResult object.
        """
        logger.info(f"Starting DOCX load for: {file_path.name}")
        
        try:
            # Load the document into memory.
            # DOCX files are generally smaller than PDFs, so this is safe.
            doc = DocxDocument(file_path)
            
            # 1. Extract Metadata
            metadata = self._extract_metadata(doc, file_path)
            
            # 2. Extract Content (Paragraphs + Tables in order)
            raw_text = self._extract_content_preserving_order(doc)
            
            if not raw_text.strip():
                logger.warning(f"DOCX file {file_path.name} appears to be empty.")
                return []
            
            # 3. Create ParsedResult
            # We return a list of 1 item to satisfy the BaseDocumentLoader interface
            result = ParsedResult(
                content=raw_text,
                metadata=metadata
            )
            
            logger.info(f"Successfully extracted text from {file_path.name}")
            return [result]
            
        except Exception as e:
            logger.error(f"Failed to load DOCX {file_path.name}: {e}")
            raise ValueError(f"Failed to parse DOCX: {e}")

    def _extract_content_preserving_order(self, doc: DocxDocument) -> str:
        """
        Iterates through the document body elements to extract text from
        paragraphs and tables in the exact order they appear.
        """
        content_parts = []
        
        # Iterate over the XML elements in the document body
        for element in doc.element.body:
            # Check if element is a Paragraph
            if element.tag.endswith('}p'):
                para = Paragraph(element, doc)
                text = para.text.strip()
                if text:
                    content_parts.append(text)
                    
            # Check if element is a Table
            elif element.tag.endswith('}tbl'):
                table = Table(element, doc)
                table_text = self._extract_table_text(table)
                if table_text:
                    content_parts.append(table_text)
                    
        return "\n".join(content_parts)

    def _extract_table_text(self, table: Table) -> str:
        """
        Extracts text from a table, formatting it to be readable.
        Uses a pipe separator for columns and newlines for rows.
        """
        rows_text = []
        for row in table.rows:
            cells_text = []
            for cell in row.cells:
                # Clean up cell text (remove excess newlines)
                cell_text = " ".join(cell.text.split())
                cells_text.append(cell_text)
            # Join columns with a pipe separator
            rows_text.append(" | ".join(cells_text))
        return "\n".join(rows_text)

    def _extract_metadata(self, doc: DocxDocument, file_path: Path) -> Dict[str, Any]:
        """Extracts file-level and document-level metadata."""
        props = doc.core_properties
        return {
            "file_name": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_size_bytes": file_path.stat().st_size,
            "source_format": "docx",
            # Core Properties
            "doc_title": props.title or "",
            "doc_author": props.author or "",
            "doc_subject": props.subject or "",
            "doc_keywords": props.keywords or "",
            "doc_created": str(props.created) if props.created else "",
            "doc_modified": str(props.modified) if props.modified else "",
            "doc_last_modified_by": props.last_modified_by or "",
            "doc_revision": props.revision
        }