"""
Preprocessing pipeline orchestrator.
Coordinates the execution of all preprocessing modules in the correct order:
1. Text Cleaning (encoding hygiene, artifact removal)
2. Metadata Extraction (document-level metadata)
3. Legal Structure Parsing (hierarchy extraction)
4. Amendment Detection (amendment directive identification)
5. Arabic Normalization (dual-text representation)

Produces a PreprocessedDocument ready for chunking (Phase 5).
"""
from dataclasses import dataclass
from typing import List, Optional

from src.preprocessing.text_cleaner import TextCleaner, CleanedText, CleaningConfig
from src.preprocessing.arabic_normalizer import ArabicNormalizer, NormalizationResult, NormalizationConfig
from src.preprocessing.metadata_extractor import MetadataExtractor, DocumentMetadata
from src.preprocessing.legal_structure_parser import LegalStructureParser, LegalStructure
from src.preprocessing.amendment_parser import AmendmentParser, AmendmentReference


@dataclass(frozen=True, slots=True)
class PreprocessedDocument:
    """
    Final output of the preprocessing pipeline.
    Contains all representations and metadata needed for downstream phases.
    
    Attributes:
        text_original: Verbatim text (cleaned only, for LLM context and UI)
        text_normalized: Folded text (for BM25/dense retrieval indexing)
        offset_map: Maps normalized indices back to original text indices
        metadata: Document-level metadata
        structure: Legal structure map (articles, chapters, etc.)
        amendments: List of detected amendment references
        doc_id: Unique document identifier
        source_path: Original file path or source reference
    """
    text_original: str
    text_normalized: str
    offset_map: tuple[int, ...]
    metadata: DocumentMetadata
    structure: LegalStructure
    amendments: tuple[AmendmentReference, ...]
    doc_id: str
    source_path: Optional[str] = None


@dataclass
class PreprocessingConfig:
    """
    Unified configuration for the preprocessing pipeline.
    """
    cleaning_config: CleaningConfig = None
    normalization_config: NormalizationConfig = None
    extract_metadata: bool = True
    parse_structure: bool = True
    detect_amendments: bool = True
    
    def __post_init__(self):
        if self.cleaning_config is None:
            object.__setattr__(self, 'cleaning_config', CleaningConfig())
        if self.normalization_config is None:
            object.__setattr__(self, 'normalization_config', NormalizationConfig())


class PreprocessingPipeline:
    """
    Orchestrates the preprocessing workflow for Egyptian legal documents.
    Stateless and thread-safe for parallel processing in workers/.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        
        # Initialize modules
        self.cleaner = TextCleaner(self.config.cleaning_config)
        self.normalizer = ArabicNormalizer(self.config.normalization_config)
        self.metadata_extractor = MetadataExtractor()
        self.structure_parser = LegalStructureParser()
        self.amendment_parser = AmendmentParser()

    def process(self, text: str, doc_id: str, source_path: Optional[str] = None, 
                categories: Optional[List[str]] = None) -> PreprocessedDocument:
        """
        Processes a raw document through the complete preprocessing pipeline.
        
        Args:
            text: Raw extracted text from document
            doc_id: Unique document identifier
            source_path: Optional path to source file
            categories: Optional list of categories/tags
            
        Returns:
            PreprocessedDocument ready for chunking
        """
        if not text:
            raise ValueError("Input text cannot be empty")
        
        # Step 1: Clean text (encoding hygiene, artifact removal)
        cleaned = self.cleaner.clean(text)
        
        # Step 2: Extract metadata (from cleaned text)
        metadata = DocumentMetadata()
        if self.config.extract_metadata:
            metadata = self.metadata_extractor.extract(cleaned.text, categories)
        
        # Step 3: Parse legal structure (from cleaned text)
        structure = LegalStructure()
        if self.config.parse_structure:
            structure = self.structure_parser.parse(cleaned.text)
        
        # Step 4: Detect amendments (requires structure for target resolution)
        amendments = tuple()
        if self.config.detect_amendments:
            amendments = tuple(self.amendment_parser.parse(cleaned.text, structure))
        
        # Step 5: Normalize text (dual-text representation)
        # This MUST be last to preserve structure detection accuracy
        normalization = self.normalizer.normalize(cleaned.text)
        
        return PreprocessedDocument(
            text_original=normalization.text_original,
            text_normalized=normalization.text_normalized,
            offset_map=normalization.offset_map,
            metadata=metadata,
            structure=structure,
            amendments=amendments,
            doc_id=doc_id,
            source_path=source_path
        )

    def process_batch(self, documents: List[dict]) -> List[PreprocessedDocument]:
        """
        Processes multiple documents in batch.
        
        Args:
            documents: List of dicts with keys: text, doc_id, source_path, categories
            
        Returns:
            List of PreprocessedDocument objects
        """
        results = []
        for doc in documents:
            try:
                result = self.process(
                    text=doc.get("text", ""),
                    doc_id=doc.get("doc_id", ""),
                    source_path=doc.get("source_path"),
                    categories=doc.get("categories")
                )
                results.append(result)
            except Exception as e:
                # Log error but continue processing other documents
                # In production, this should use proper logging
                print(f"Error processing document {doc.get('doc_id')}: {e}")
                continue
        return results