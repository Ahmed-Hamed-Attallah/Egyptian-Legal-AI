import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from src.chunking.base_chunker import BaseChunker
from src.chunking.legal_chunker import LegalChunker
from src.chunking.recursive_chunker import RecursiveChunker
from src.chunking.semantic_chunker import SemanticChunker, EmbeddingProvider
from src.chunking.quality_filter import QualityReport, filter_chunks
from src.domain.chunks import Chunk
from src.domain.exceptions import DocumentChunkingError
from src.preprocessing.pipeline import PreprocessedDocument
from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    strategy: str = settings.CHUNKING_STRATEGY
    chunk_size: int = settings.CHUNK_SIZE
    chunk_overlap: int = settings.CHUNK_OVERLAP
    min_chunk_size: int = getattr(settings, 'MIN_CHUNK_SIZE', 50)
    semantic_threshold: float = 0.5
    # --- Quality gate -------------------------------------------------------
    # When True (default), low-quality chunks (mojibake, PDF UI garbage,
    # repetitive TOC loops, broken quotes, empty) are dropped before they
    # reach embedding / Qdrant. A QualityReport is returned alongside the
    # accepted chunks so the caller can log or persist drop statistics.
    apply_quality_filter: bool = True


class ChunkingPipeline:
    def __init__(
        self,
        config: Optional[ChunkingConfig] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.config = config or ChunkingConfig()
        self.embedding_provider = embedding_provider
        self.chunker = self._resolve_chunker()

    def _resolve_chunker(self) -> BaseChunker:
        strategy = self.config.strategy.lower()
        if strategy == "legal":
            return LegalChunker(self.config.chunk_size, self.config.chunk_overlap, self.config.min_chunk_size)
        elif strategy == "recursive":
            return RecursiveChunker(self.config.chunk_size, self.config.chunk_overlap, self.config.min_chunk_size)
        elif strategy == "semantic":
            if not self.embedding_provider:
                raise ValueError("Semantic chunking requires an embedding_provider.")
            return SemanticChunker(
                embedding_model=self.embedding_provider,
                similarity_threshold=self.config.semantic_threshold,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, preprocessed_doc: PreprocessedDocument) -> List[Chunk]:
        """
        Backward-compatible entry point.

        Returns the accepted chunk list (as before). When the quality filter
        is enabled the drop statistics are logged and, if the caller needs
        them, accessible via ``process_with_report``.
        """
        chunks, _report = self.process_with_report(preprocessed_doc)
        return chunks

    def process_with_report(
        self,
        preprocessed_doc: PreprocessedDocument,
    ) -> Tuple[List[Chunk], QualityReport]:
        """
        Chunk a document and return (accepted_chunks, quality_report).

        The quality filter runs AFTER chunking and BEFORE the result is
        handed back to the caller. No garbage chunk is ever returned.
        """
        try:
            raw_chunks = self.chunker.chunk(preprocessed_doc)
        except Exception as e:
            logger.error(f"Chunking failed for doc {preprocessed_doc.doc_id}: {e}")
            raise DocumentChunkingError(
                message=f"Failed to chunk document: {str(e)}",
                document_id=preprocessed_doc.doc_id,
                chunking_strategy=self.config.strategy,
            )

        if self.config.apply_quality_filter:
            clean_chunks, report = filter_chunks(raw_chunks)
            if report.total_rejected > 0:
                logger.info(
                    "Chunking doc_id=%s: produced=%d accepted=%d dropped=%d reasons=%s",
                    preprocessed_doc.doc_id,
                    report.total_input,
                    report.total_accepted,
                    report.total_rejected,
                    dict(report.by_reason),
                )
                for sample in report.rejected_samples[:2]:
                    logger.debug(
                        "Dropped chunk %s/%s reason=%s sample=%r",
                        sample.get("document_id"),
                        sample.get("chunk_index"),
                        sample.get("reason"),
                        sample.get("sample"),
                    )
            return clean_chunks, report

        return raw_chunks, QualityReport(
            total_input=len(raw_chunks),
            total_accepted=len(raw_chunks),
            total_rejected=0,
        )

    def process_batch(self, preprocessed_docs: List[PreprocessedDocument]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        success_count = 0
        failure_count = 0
        for doc in preprocessed_docs:
            try:
                chunks = self.process(doc)
                all_chunks.extend(chunks)
                success_count += 1
            except DocumentChunkingError as e:
                logger.error(f"Skipping doc {doc.doc_id} due to chunking error: {e.message}")
                failure_count += 1
            except Exception as e:
                logger.error(f"Unexpected error chunking doc {doc.doc_id}: {e}")
                failure_count += 1
        logger.info(
            f"Chunking batch complete. Success: {success_count}, "
            f"Failed: {failure_count}, Total Chunks: {len(all_chunks)}"
        )
        return all_chunks