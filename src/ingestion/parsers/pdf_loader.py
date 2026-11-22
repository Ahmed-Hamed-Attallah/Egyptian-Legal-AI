# src/ingestion/parsers/pdf_loader.py
"""
PDF Document Loader.
Extracts raw text and metadata from PDF files using PyMuPDF (fitz).
Handles both text-based and scanned (image-based) PDFs by falling back to OCR.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from src.ingestion.models import ParsedResult
from src.ingestion.parsers.base import BaseDocumentLoader

logger = logging.getLogger(__name__)

# Minimum number of characters to consider a page as "text-based".
# If a page has fewer characters, we assume it's a scanned image and trigger OCR.
MIN_TEXT_LENGTH_THRESHOLD = 50

class PDFLoader(BaseDocumentLoader):
    """
    Loader for PDF files.
    Processes the PDF page-by-page to maintain a low memory footprint.
    """
    def __init__(self, ocr_engine: Optional[Any] = None):
        """
        Args:
            ocr_engine: An instance of the OCREngine. If None, scanned pages
            will be skipped or logged as warnings. (We will inject
            the actual OCR engine in the pipeline phase).
        """
        self.ocr_engine = ocr_engine

    def load(self, file_path: Path) -> List[ParsedResult]:
        """
        Loads a PDF file and returns a list of ParsedResult objects (one per page).
        """
        logger.info(f"Starting PDF load for: {file_path.name}")
        results: List[ParsedResult] = []
        
        try:
            # Open the PDF. We use 'open' which is memory efficient.
            # We do not load the entire file into RAM at once.
            with fitz.open(file_path) as doc:
                file_metadata = self._extract_file_metadata(doc, file_path)
                total_pages = len(doc)
                logger.info(f"PDF has {total_pages} pages.")
                
                for page_num in range(total_pages):
                    page = doc.load_page(page_num)
                    
                    # 1. Attempt standard text extraction
                    raw_text = page.get_text("text").strip()
                    
                    # 2. Fallback to OCR if text is insufficient (Scanned PDF)
                    if len(raw_text) < MIN_TEXT_LENGTH_THRESHOLD:
                        logger.debug(f"Page {page_num + 1} has insufficient text. Attempting OCR.")
                        raw_text = self._process_scanned_page(page)
                    
                    # 3. Create the ParsedResult for this page
                    if raw_text:
                        page_metadata = self._build_page_metadata(
                            file_metadata, page_num, total_pages, page
                        )
                        results.append(ParsedResult(
                            content=raw_text,
                            metadata=page_metadata
                        ))
                    else:
                        logger.warning(f"Page {page_num + 1} yielded no text after OCR. Skipping.")
                        
        except fitz.FileDataError as e:
            logger.error(f"Corrupt or invalid PDF file {file_path.name}: {e}")
            raise ValueError(f"Failed to parse PDF: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading PDF {file_path.name}: {e}")
            raise
            
        logger.info(f"Successfully extracted {len(results)} pages from {file_path.name}")
        return results

    def _process_scanned_page(self, page: fitz.Page) -> str:
        """
        Converts a scanned PDF page to an image and passes it to the OCR engine.
        """
        if not self.ocr_engine:
            logger.warning("OCR engine not provided. Skipping scanned page.")
            return ""
        
        try:
            # Render page to an image (Pixmap).
            # We use a moderate DPI (150) to balance OCR accuracy and memory usage.
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")
            
            # Delegate to the OCR engine
            return self.ocr_engine.extract_text_from_image(image_bytes)
        except Exception as e:
            logger.error(f"OCR failed for page: {e}")
            return ""

    def _extract_file_metadata(self, doc: fitz.Document, file_path: Path) -> Dict[str, Any]:
        """Extracts file-level metadata from the PDF properties."""
        meta = doc.metadata or {}
        return {
            "file_name": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_size_bytes": file_path.stat().st_size,
            "pdf_title": meta.get("title", ""),
            "pdf_author": meta.get("author", ""),
            "pdf_creation_date": meta.get("creationDate", ""),
            "pdf_modification_date": meta.get("modDate", ""),
            "total_pages": len(doc),
            "source_format": "pdf"
        }

    def _build_page_metadata(
        self,
        file_metadata: Dict[str, Any],
        page_num: int,
        total_pages: int,
        page: fitz.Page
    ) -> Dict[str, Any]:
        """Builds page-specific metadata, merging it with file-level metadata."""
        # page.rect gives the dimensions of the page
        page_width = page.rect.width
        page_height = page.rect.height
        
        return {
            **file_metadata,
            "page_number": page_num + 1,  # 1-indexed for legal citations
            "page_width_pt": page_width,
            "page_height_pt": page_height,
            "is_landscape": page_width > page_height
        }