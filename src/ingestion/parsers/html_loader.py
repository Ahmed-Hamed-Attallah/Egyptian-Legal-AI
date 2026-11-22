# src/ingestion/parsers/html_loader.py
"""
HTML Document Loader.
Extracts readable text from .html files, stripping scripts and styles.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from src.ingestion.models import ParsedResult
from src.ingestion.parsers.base import BaseDocumentLoader

logger = logging.getLogger(__name__)

class HTMLLoader(BaseDocumentLoader):
    """Loader for HTML (.html, .htm) files."""
    
    def load(self, file_path: Path) -> List[ParsedResult]:
        logger.info(f"Starting HTML load for: {file_path.name}")
        
        try:
            html_content = file_path.read_text(encoding='utf-8')
            
            # Parse and clean HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements (they contain no readable text)
            for script_or_style in soup(['script', 'style', 'noscript']):
                script_or_style.decompose()
            
            # Extract text with basic whitespace cleanup
            raw_text = soup.get_text(separator='\n', strip=True)
            
            metadata = self._extract_metadata(file_path)
            
            return [ParsedResult(content=raw_text, metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Failed to load HTML file {file_path.name}: {e}")
            raise ValueError(f"Failed to parse HTML file: {e}")

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        return {
            "file_name": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_size_bytes": file_path.stat().st_size,
            "source_format": "html"
        }