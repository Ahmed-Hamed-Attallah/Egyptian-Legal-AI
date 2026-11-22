"""
Domain models for document chunks.

This module defines the data structures for document chunks. Chunks are 
semantically meaningful pieces of text extracted from a Document, 
preserving legal boundaries (Articles, Chapters) for accurate RAG retrieval.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class ChunkType(str, Enum):
    """
    Categorizes the structural role of the chunk within the legal document.
    """
    PREAMBLE = "preamble"
    CHAPTER_HEADER = "chapter_header"
    SECTION_HEADER = "section_header"
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    ANNEX = "annex"
    OTHER = "other"


# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------

class Chunk(BaseModel):
    """
    Represents a single, retrieval-ready piece of text extracted from a Document.
    
    Chunks preserve legal boundaries (like article numbers) to ensure 
    accurate citation and context retrieval in the RAG pipeline.
    """

    # Pydantic V2 configuration
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    # Core Identifiers & Linkage
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for this specific chunk."
    )
    document_id: uuid.UUID = Field(
        ..., 
        description="Links the chunk back to its parent physical Document."
    )
    chunk_index: int = Field(
        ..., 
        ge=0, 
        description="Sequential order of this chunk within the document."
    )

    # Content & Structure
    content: str = Field(default="", description="The text content of the chunk")
    
    chunk_type: ChunkType = Field(
        default=ChunkType.OTHER,
        description="The structural category of the chunk (e.g., ARTICLE, CHAPTER_HEADER)."
    )

    # Legal Context & Positioning
    page_number: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="Page number in the original document for citation."
    )
    section_title: Optional[str] = Field(
        default=None, 
        description="Heading of the section/chapter (e.g., 'الباب الأول: الأحكام العامة')."
    )
    article_number: Optional[str] = Field(
        default=None, 
        description="Specific article number (e.g., '12', '12 مكرر'). Uses str for complex numbering."
    )

    # Processing Metrics
    token_count: int = Field(
        default=0, 
        ge=0, 
        description="Number of tokens in the chunk for context window management."
    )

    # Extensibility & Timestamps
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Low-level extraction data (e.g., OCR bounding boxes, raw parser output)."
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Timestamp of chunk creation."
    )

    # -------------------------------------------------------------------------
    # Business Logic Methods
    # -------------------------------------------------------------------------

    def has_legal_reference(self) -> bool:
        """
        Checks if this chunk contains a specific legal reference 
        (an article number or a section title).
        """
        return bool(self.article_number) or bool(self.section_title)

    def is_article(self) -> bool:
        """
        Checks if this chunk represents a formal legal article.
        """
        return self.chunk_type == ChunkType.ARTICLE and bool(self.article_number)

    def get_citation_label(self) -> str:
        """
        Generates a short, human-readable label for UI citations.
        Falls back to page number if no article/section is available.
        """
        if self.article_number:
            return f"Article {self.article_number}"
        if self.section_title:
            return self.section_title
        if self.page_number:
            return f"Page {self.page_number}"
        return f"Chunk {self.chunk_index}"