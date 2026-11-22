# src/chunking/semantic_chunker.py
"""
Semantic chunker.
Splits text based on semantic similarity drops between sentences.
Uses Dependency Injection for the embedding model to avoid circular imports.
"""
import re
import logging
from typing import List, Protocol, runtime_checkable
from src.chunking.base_chunker import BaseChunker
from src.domain.chunks import Chunk, ChunkType
from src.chunking.chunk_metadata import build_chunk_metadata
from src.preprocessing.pipeline import PreprocessedDocument

logger = logging.getLogger(__name__)

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding models to avoid circular imports."""
    def encode_documents(self, texts: List[str]) -> List[List[float]]: ...

class SemanticChunker(BaseChunker):
    """
    Chunks text by evaluating semantic similarity between consecutive sentences.
    """
    def __init__(self, embedding_model: EmbeddingProvider, 
                 similarity_threshold: float = 0.5,
                 chunk_size: int = None, chunk_overlap: int = None):
        super().__init__(chunk_size, chunk_overlap)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def chunk(self, preprocessed_doc: PreprocessedDocument) -> List[Chunk]:
        text = preprocessed_doc.text_original
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Embed sentences
        try:
            embeddings = self.embedding_model.encode_documents(sentences)
        except Exception as e:
            logger.error(f"Semantic chunking embedding failed: {e}. Falling back to recursive.")
            from src.chunking.recursive_chunker import RecursiveChunker
            return RecursiveChunker(self.chunk_size, self.chunk_overlap).chunk(preprocessed_doc)

        # Find boundaries
        boundaries = [0]
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i+1])
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)
        boundaries.append(len(sentences))

        # Group sentences into chunks
        chunks = []
        chunk_idx = 0
        for i in range(len(boundaries) - 1):
            group = sentences[boundaries[i]:boundaries[i+1]]
            chunk_text = " ".join(group).strip()
            
            # Respect max chunk size by falling back to recursive if a semantic group is too large
            if len(chunk_text) > self.chunk_size:
                from src.chunking.recursive_chunker import RecursiveChunker
                # Create a dummy doc for the recursive chunker
                sub_doc = PreprocessedDocument(
                    text_original=chunk_text,
                    text_normalized=chunk_text,
                    offset_map=tuple(range(len(chunk_text))),
                    metadata=preprocessed_doc.metadata,
                    structure=preprocessed_doc.structure,
                    amendments=preprocessed_doc.amendments,
                    doc_id=preprocessed_doc.doc_id,
                    source_path=preprocessed_doc.source_path
                )
                sub_chunks = RecursiveChunker(self.chunk_size, self.chunk_overlap).chunk(sub_doc)
                for sc in sub_chunks:
                    sc.chunk_index = chunk_idx
                    sc.metadata["chunk_index"] = chunk_idx
                    sc.metadata["chunking_strategy"] = "semantic_hybrid"
                    chunks.append(sc)
                    chunk_idx += 1
            else:
                metadata = build_chunk_metadata(
                    preprocessed_doc=preprocessed_doc,
                    chunk_index=chunk_idx,
                    chunking_strategy="semantic"
                )
                chunks.append(Chunk(
                    document_id=metadata["document_id"],
                    chunk_index=chunk_idx,
                    content=chunk_text,
                    chunk_type=ChunkType.OTHER,
                    token_count=len(chunk_text.split()),
                    metadata=metadata
                ))
                chunk_idx += 1

        logger.info(f"SemanticChunker produced {len(chunks)} chunks for doc {preprocessed_doc.doc_id}")
        return chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_sentences(text)

    def _split_sentences(self, text: str) -> List[str]:
        # Basic Arabic/English sentence splitting
        pattern = r'(?<=[.!?؟。])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)