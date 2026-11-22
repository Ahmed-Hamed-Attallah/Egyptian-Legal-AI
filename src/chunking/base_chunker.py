import logging
from abc import ABC, abstractmethod
from typing import List
from src.domain.chunks import Chunk
from src.preprocessing.pipeline import PreprocessedDocument
from src.config.settings import settings

logger = logging.getLogger(__name__)

class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.
    """
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, min_chunk_size: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.min_chunk_size = min_chunk_size or settings.MIN_CHUNK_SIZE
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    @abstractmethod
    def chunk(self, preprocessed_doc: PreprocessedDocument) -> List[Chunk]:
        pass

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        pass