# src/ingestion/parsers/text_loader.py
"""
Plain Text Document Loader.
Extracts raw text from .txt files with basic encoding handling.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
from src.ingestion.models import ParsedResult
from src.ingestion.parsers.base import BaseDocumentLoader

logger = logging.getLogger(__name__)

class TextLoader(BaseDocumentLoader):
    """Loader for plain text (.txt) files."""
    
    def load(self, file_path: Path) -> List[ParsedResult]:
        logger.info(f"Starting Text load for: {file_path.name}")
        
        try:
            # Try UTF-8 first, fallback to latin-1 if needed (common in older files)
            try:
                content = file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {file_path.name}, falling back to latin-1")
                content = file_path.read_text(encoding='latin-1')
            
            metadata = self._extract_metadata(file_path)
            
            return [ParsedResult(content=content.strip(), metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Failed to load text file {file_path.name}: {e}")
            raise ValueError(f"Failed to parse text file: {e}")

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        return {
            "file_name": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_size_bytes": file_path.stat().st_size,
            "source_format": "txt"
        }