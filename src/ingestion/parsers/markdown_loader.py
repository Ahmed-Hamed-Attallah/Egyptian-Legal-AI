# src/ingestion/parsers/markdown_loader.py
"""
Markdown Document Loader.
Extracts raw text from .md files, preserving markdown syntax for downstream chunking.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
from src.ingestion.models import ParsedResult
from src.ingestion.parsers.base import BaseDocumentLoader

logger = logging.getLogger(__name__)

class MarkdownLoader(BaseDocumentLoader):
    """Loader for Markdown (.md) files."""
    
    def load(self, file_path: Path) -> List[ParsedResult]:
        logger.info(f"Starting Markdown load for: {file_path.name}")
        
        try:
            content = file_path.read_text(encoding='utf-8')
            metadata = self._extract_metadata(file_path)
            
            return [ParsedResult(content=content.strip(), metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Failed to load markdown file {file_path.name}: {e}")
            raise ValueError(f"Failed to parse markdown file: {e}")

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        return {
            "file_name": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_size_bytes": file_path.stat().st_size,
            "source_format": "md"
        }