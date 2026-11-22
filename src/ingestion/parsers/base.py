"""
Base interface for all document parsers/loaders.
Ensures all loaders return a standardized list of ParsedResult objects.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from src.ingestion.models import ParsedResult

class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: Path) -> List[ParsedResult]:
        """
        Load and parse a file into a list of ParsedResult objects.
        Args:
            file_path: Path to the file to be loaded.
        Returns:
            A list of ParsedResult objects containing extracted text and metadata.
        """
        pass