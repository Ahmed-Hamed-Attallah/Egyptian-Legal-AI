# src/rag/citations.py
"""
Citation resolution for RAG pipeline.
"""
import logging
from typing import List, Dict
from src.domain.queries import RetrievedChunk
from src.domain.citations import Citation
from src.generation.citation_formatter import CitationFormatter

logger = logging.getLogger(__name__)

class CitationResolver:
    """
    Resolves and validates citations in RAG responses.
    """
    
    def __init__(self):
        self.formatter = CitationFormatter()

    def resolve_citations(self, chunks: List[RetrievedChunk]) -> List[Citation]:
        """
        Resolve citations from retrieved chunks.
        """
        return self.formatter.format_citations(chunks)

    def validate_citation_references(self, answer: str, citations: List[Citation]) -> dict:
        """
        Validate that citations are properly referenced in the answer.
        """
        if not citations:
            return {"valid": True, "unreferenced": [], "referenced": []}

        referenced = []
        unreferenced = []

        for citation in citations:
            # Basic validation - check if citation info appears in answer
            citation_text = self.formatter.format_citation_text(citation)
            if citation.law_number and str(citation.law_number) in answer:
                referenced.append(citation)
            elif citation.article_number and str(citation.article_number) in answer:
                referenced.append(citation)
            else:
                unreferenced.append(citation)

        return {
            "valid": len(unreferenced) == 0,
            "referenced": referenced,
            "unreferenced": unreferenced
        }