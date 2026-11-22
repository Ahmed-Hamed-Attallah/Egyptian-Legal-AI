# src/ingestion/ocr/image_ocr.py
"""
Image OCR Loader.
Extracts text from image files (PNG, JPG, TIFF) using OCR.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
from src.ingestion.models import ParsedResult
from src.ingestion.parsers.base import BaseDocumentLoader
from src.ingestion.ocr.engine import BaseOCREngine

logger = logging.getLogger(__name__)

class ImageOCRLoader(BaseDocumentLoader):
    """
    Loader for image files (PNG, JPG, TIFF, BMP).
    Extracts text using OCR.
    """
    def __init__(self, ocr_engine: BaseOCREngine):
        """
        Args:
            ocr_engine: OCR engine instance.
        """
        self.ocr_engine = ocr_engine

    def load(self, file_path: Path) -> List[ParsedResult]:
        """
        Loads an image file and extracts text via OCR.
        """
        logger.info(f"Starting Image OCR load for: {file_path.name}")
        
        try:
            # Read image bytes
            image_bytes = file_path.read_bytes()
            
            # Extract text via OCR
            raw_text = self.ocr_engine.extract_text_from_image(image_bytes)
            
            if not raw_text:
                logger.warning(f"OCR extracted no text from {file_path.name}")
                return []
            
            metadata = self._extract_metadata(file_path)
            
            result = ParsedResult(
                content=raw_text,
                metadata=metadata
            )
            
            logger.info(f"Successfully extracted text from image {file_path.name}")
            return [result]
            
        except Exception as e:
            logger.error(f"Failed to load image {file_path.name}: {e}")
            raise ValueError(f"Failed to parse image: {e}")

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        return {
            "file_name": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_size_bytes": file_path.stat().st_size,
            "source_format": "image_ocr"
        }