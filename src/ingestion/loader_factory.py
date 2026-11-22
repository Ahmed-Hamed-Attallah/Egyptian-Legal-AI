"""
Factory for creating the appropriate document loader based on file extension.
Uses lazy loading to prevent importing heavy dependencies (like PyMuPDF, openpyxl) 
into memory unless they are actually needed.
"""
import importlib
from pathlib import Path
from typing import Dict
import logging

from src.ingestion.parsers.base import BaseDocumentLoader

logger = logging.getLogger(__name__)

class LoaderFactory:
    """
    Factory class to instantiate the correct document loader based on file extension.
    """
    
    # Mapping of file extensions to their fully qualified module and class paths.
    # We use strings instead of direct imports to enable LAZY LOADING.
    _LOADER_MAP: Dict[str, str] = {
        ".pdf": "src.ingestion.parsers.pdf_loader.PDFLoader",
        ".docx": "src.ingestion.parsers.docx_loader.DOCXLoader",
        ".csv": "src.ingestion.parsers.csv_loader.CSVLoader",
        ".xlsx": "src.ingestion.parsers.excel_loader.ExcelLoader",
        ".xls": "src.ingestion.parsers.excel_loader.ExcelLoader",
        ".md": "src.ingestion.parsers.markdown_loader.MarkdownLoader",
        ".txt": "src.ingestion.parsers.text_loader.TextLoader",
        ".html": "src.ingestion.parsers.html_loader.HTMLLoader",
        ".htm": "src.ingestion.parsers.html_loader.HTMLLoader",
    }

    @classmethod
    def get_loader(cls, file_path: Path) -> BaseDocumentLoader:
        """
        Returns an instance of the appropriate loader for the given file path.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            An instance of a BaseDocumentLoader subclass.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        ext = file_path.suffix.lower()
        loader_path = cls._LOADER_MAP.get(ext)
        
        if not loader_path:
            supported = ", ".join(cls._LOADER_MAP.keys())
            raise ValueError(f"Unsupported file type '{ext}'. Supported types: {supported}")
            
        # Lazy import: Only loads the specific parser and its heavy dependencies when requested
        module_path, class_name = loader_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        loader_class = getattr(module, class_name)
        
        logger.info(f"Instantiating loader for file type: {ext}")
        return loader_class()