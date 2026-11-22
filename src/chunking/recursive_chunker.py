import re
import logging
from typing import List
from src.chunking.base_chunker import BaseChunker
from src.domain.chunks import Chunk, ChunkType
from src.chunking.chunk_metadata import build_chunk_metadata
from src.preprocessing.pipeline import PreprocessedDocument

logger = logging.getLogger(__name__)

class RecursiveChunker(BaseChunker):
    """
    Chunks text by recursively splitting on separators.
    Supports min_chunk_size merging and safe overlap bounding.
    """
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, 
                 min_chunk_size: int = None, separators: List[str] = None):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size)
        # Arabic and English aware separators
        self.separators = separators or ["\n\n", "\n", ". ", "؟ ", "! ", "، ", "؛ ", " ", ""]

    def chunk(self, preprocessed_doc: PreprocessedDocument) -> List[Chunk]:
        text = preprocessed_doc.text_original
        if not text.strip():
            return []
        
        text_chunks = self.split_text(text)
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
            
            metadata = build_chunk_metadata(
                preprocessed_doc=preprocessed_doc,
                chunk_index=idx,
                chunking_strategy="recursive"
            )
            chunks.append(Chunk(
                document_id=metadata["document_id"],
                chunk_index=idx,
                content=chunk_text,
                chunk_type=ChunkType.OTHER,
                token_count=len(chunk_text.split()),
                metadata=metadata
            ))
        logger.info(f"RecursiveChunker produced {len(chunks)} chunks for doc {preprocessed_doc.doc_id}")
        return chunks

    def split_text(self, text: str) -> List[str]:
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if re.search(re.escape(sep), text):
                separator = sep
                new_separators = separators[i+1:]
                break
        
        splits = text.split(separator) if separator else list(text)
        
        good_splits = []
        current_chunk = ""
        for s in splits:
            piece = s + separator if separator and s != splits[-1] else s
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk:
                    good_splits.append(current_chunk)
                if len(piece) > self.chunk_size and new_separators:
                    good_splits.extend(self._recursive_split(piece, new_separators))
                else:
                    current_chunk = piece
        if current_chunk:
            good_splits.append(current_chunk)
            
        # Apply safe overlap
        if self.chunk_overlap > 0 and len(good_splits) > 1:
            final_chunks = [good_splits[0]]
            for i in range(1, len(good_splits)):
                prev_text = good_splits[i-1]
                current_text = good_splits[i]
                
                max_overlap = min(self.chunk_overlap, len(prev_text))
                safe_overlap_len = min(max_overlap, self.chunk_size - len(current_text))
                safe_overlap_len = max(0, safe_overlap_len)
                
                overlap_text = prev_text[-safe_overlap_len:] if safe_overlap_len > 0 else ""
                final_chunks.append(overlap_text + current_text)
            good_splits = final_chunks

        # Apply min_chunk_size merging
        if self.min_chunk_size > 0 and len(good_splits) > 1:
            merged_splits = [good_splits[0]]
            for i in range(1, len(good_splits)):
                last_chunk = merged_splits[-1]
                current_chunk = good_splits[i]
                
                if len(last_chunk) < self.min_chunk_size and len(last_chunk) + len(current_chunk) <= self.chunk_size:
                    merged_splits[-1] = last_chunk + current_chunk
                elif len(current_chunk) < self.min_chunk_size and len(last_chunk) + len(current_chunk) <= self.chunk_size:
                    merged_splits[-1] = last_chunk + current_chunk
                else:
                    merged_splits.append(current_chunk)
            good_splits = merged_splits

        return good_splits