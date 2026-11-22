# src/ingestion/ocr/pdf_ocr.py
"""
PDF OCR Loader.
Specialized loader for scanned PDFs that forces OCR on all pages.
Use this when you know the PDF is fully scanned (no text layer).
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from src.ingestion.models import ParsedResult
from src.ingestion.parsers.base import BaseDocumentLoader
from src.ingestion.ocr.engine import BaseOCREngine

logger = logging.getLogger(__name__)

class PDFOCRLoader(BaseDocumentLoader):
    """
    Loader for scanned PDF files.
    Forces OCR on every page, regardless of whether text exists.
    """
    def __init__(self, ocr_engine: BaseOCREngine, dpi: int = 200):
        """
        Args:
            ocr_engine: OCR engine instance.
            dpi: Resolution for rendering pages. Higher = better accuracy but slower.
        """
        self.ocr_engine = ocr_engine
        self.dpi = dpi

    def load(self, file_path: Path) -> List[ParsedResult]:
        """
        Loads a scanned PDF and extracts text via OCR from every page.
        """
        logger.info(f"Starting PDF OCR load for: {file_path.name}")
        results: List[ParsedResult] = []
        
        try:
            with fitz.open(file_path) as doc:
                file_metadata = self._extract_file_metadata(doc, file_path)
                total_pages = len(doc)
                
                logger.info(f"PDF has {total_pages} pages. Running OCR on all pages...")
                
                for page_num in range(total_pages):
                    page = doc.load_page(page_num)
                    
                    # Render page to image
                    pix = page.get_pixmap(dpi=self.dpi)
                    image_bytes = pix.tobytes("png")
                    
                    # Extract text via OCR
                    raw_text = self.ocr_engine.extract_text_from_image(image_bytes)
                    
                    if raw_text:
                        page_metadata = self._build_page_metadata(
                            file_metadata, page_num, total_pages, page
                        )
                        results.append(ParsedResult(
                            content=raw_text,
                            metadata=page_metadata
                        ))
                    else:
                        logger.warning(f"OCR failed for page {page_num + 1}. Skipping.")
                        
        except Exception as e:
            logger.error(f"Failed to load PDF with OCR {file_path.name}: {e}")
            raise ValueError(f"Failed to parse PDF with OCR: {e}")
            
        logger.info(f"Successfully extracted {len(results)} pages via OCR from {file_path.name}")
        return results

    def _extract_file_metadata(self, doc: fitz.Document, file_path: Path) -> Dict[str, Any]:
        meta = doc.metadata or {}
        return {
            "file_name": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_size_bytes": file_path.stat().st_size,
            "pdf_title": meta.get("title", ""),
            "pdf_author": meta.get("author", ""),
            "total_pages": len(doc),
            "source_format": "pdf_ocr"
        }

    def _build_page_metadata(
        self,
        file_metadata: Dict[str, Any],
        page_num: int,
        total_pages: int,
        page: fitz.Page
    ) -> Dict[str, Any]:
        return {
            **file_metadata,
            "page_number": page_num + 1,
            "page_width_pt": page.rect.width,
            "page_height_pt": page.rect.height,
            "is_landscape": page.rect.width > page.rect.height
        }