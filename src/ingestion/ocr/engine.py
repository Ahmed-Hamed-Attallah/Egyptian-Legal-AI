"""
OCR Engine.
Provides text extraction from images using Tesseract OCR with Arabic language support.
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional
from io import BytesIO

from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)

class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    def extract_text_from_image(self, image_bytes: bytes) -> str:
        """
        Extract text from an image.
        
        Args:
            image_bytes: Raw image bytes (PNG, JPG, etc.)
            
        Returns:
            Extracted text string.
        """
        pass


class TesseractOCREngine(BaseOCREngine):
    """
    Tesseract-based OCR engine with Arabic language support.
    Optimized for legal documents with mixed Arabic/English text.
    """
    
    def __init__(self, lang: str = "ara+eng", psm: int = 6):
        """
        Args:
            lang: Tesseract language code. "ara+eng" for Arabic + English.
            psm: Page segmentation mode. 6 = Assume a single uniform block of text.
        """
        self.lang = lang
        self.psm = psm
        logger.info(f"Initialized Tesseract OCR engine with lang={lang}, psm={psm}")
    
    def extract_text_from_image(self, image_bytes: bytes) -> str:
        """
        Extracts text from image bytes using Tesseract.
        """
        try:
            # Load image from bytes
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to grayscale for better OCR accuracy
            if image.mode != 'L':
                image = image.convert('L')
            
            # Configure Tesseract
            custom_config = f'--oem 3 --psm {self.psm}'
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=self.lang, config=custom_config)
            
            # Clean up the extracted text
            text = self._clean_extracted_text(text)
            
            logger.debug(f"OCR extracted {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def _clean_extracted_text(self, text: str) -> str:
        """Basic cleanup of OCR output."""
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        return '\n'.join(lines)