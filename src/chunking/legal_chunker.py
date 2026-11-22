import logging
from typing import List, Optional
from src.chunking.base_chunker import BaseChunker
from src.chunking.recursive_chunker import RecursiveChunker
from src.chunking.chunk_metadata import build_chunk_metadata
from src.domain.chunks import Chunk, ChunkType
from src.preprocessing.pipeline import PreprocessedDocument
from src.preprocessing.legal_structure_parser import LegalUnit, LegalUnitType

logger = logging.getLogger(__name__)

class LegalChunker(BaseChunker):
    """
    Chunks documents based on legal structure (Articles, Chapters).
    Falls back to RecursiveChunker for unstructured text or oversized articles.
    """
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, min_chunk_size: int = None):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        self.recursive_chunker = RecursiveChunker(chunk_size, chunk_overlap, min_chunk_size)

    def chunk(self, preprocessed_doc: PreprocessedDocument) -> List[Chunk]:
        text = preprocessed_doc.text_original
        structure = preprocessed_doc.structure
        if not text.strip():
            return []
        
        articles = [u for u in structure.units if u.unit_type == LegalUnitType.ARTICLE]
        if not articles:
            logger.debug(f"No articles found in doc {preprocessed_doc.doc_id}. Falling back to recursive chunking.")
            return self.recursive_chunker.chunk(preprocessed_doc)
        
        chunks = []
        chunk_idx = 0
        
        # Handle preamble (text before first article)
        if articles[0].start_offset > 0:
            preamble_text = text[:articles[0].start_offset].strip()
            if preamble_text:
                metadata = build_chunk_metadata(
                    preprocessed_doc=preprocessed_doc,
                    chunk_index=chunk_idx,
                    chunking_strategy="legal_preamble"
                )
                chunks.append(Chunk(
                    document_id=metadata["document_id"],
                    chunk_index=chunk_idx,
                    content=preamble_text,
                    chunk_type=ChunkType.PREAMBLE,
                    token_count=len(preamble_text.split()),
                    metadata=metadata
                ))
                chunk_idx += 1
                
        # Process each article
        for i, article in enumerate(articles):
            body_start = article.start_offset
            body_end = articles[i+1].start_offset if i+1 < len(articles) else len(text)
            article_body = text[body_start:body_end].strip()
            article_number = str(article.number) if article.number is not None else None
            
            parent_section = self._find_parent_section(article, structure.units)
            section_title = parent_section.title if parent_section else None
            
            if not article_body:
                continue
                
            # If article is too long, split it but preserve article metadata
            if len(article_body) > self.chunk_size:
                sub_texts = self.recursive_chunker.split_text(article_body)
                for part_idx, sub_text in enumerate(sub_texts):
                    if not sub_text.strip():
                        continue
                    metadata = build_chunk_metadata(
                        preprocessed_doc=preprocessed_doc,
                        chunk_index=chunk_idx,
                        chunking_strategy="legal_article_split",
                        article_number=article_number,
                        section_title=section_title,
                        chunk_part=part_idx + 1
                    )
                    chunks.append(Chunk(
                        document_id=metadata["document_id"],
                        chunk_index=chunk_idx,
                        content=sub_text,
                        chunk_type=ChunkType.ARTICLE,
                        article_number=article_number,
                        section_title=section_title,
                        token_count=len(sub_text.split()),
                        metadata=metadata
                    ))
                    chunk_idx += 1
            else:
                metadata = build_chunk_metadata(
                    preprocessed_doc=preprocessed_doc,
                    chunk_index=chunk_idx,
                    chunking_strategy="legal_article",
                    article_number=article_number,
                    section_title=section_title
                )
                chunks.append(Chunk(
                    document_id=metadata["document_id"],
                    chunk_index=chunk_idx,
                    content=article_body,
                    chunk_type=ChunkType.ARTICLE,
                    article_number=article_number,
                    section_title=section_title,
                    token_count=len(article_body.split()),
                    metadata=metadata
                ))
                chunk_idx += 1
                
        logger.info(f"LegalChunker produced {len(chunks)} chunks for doc {preprocessed_doc.doc_id}")
        return chunks

    def split_text(self, text: str) -> List[str]:
        return self.recursive_chunker.split_text(text)

    def _find_parent_section(self, article: LegalUnit, all_units: List[LegalUnit]) -> Optional[LegalUnit]:
        parent = None
        for unit in all_units:
            if unit.start_offset >= article.start_offset:
                break
            if unit.unit_type in (LegalUnitType.CHAPTER, LegalUnitType.BOOK, LegalUnitType.SECTION):
                parent = unit
        return parent