# src/preprocessing/__init__.py
"""
Preprocessing module for Egyptian Legal AI.
Provides text cleaning, normalization, metadata extraction, 
legal structure parsing, and amendment detection.
"""

from src.preprocessing.text_cleaner import TextCleaner, CleanedText, CleaningConfig
from src.preprocessing.arabic_normalizer import ArabicNormalizer, NormalizationResult, NormalizationConfig
from src.preprocessing.metadata_extractor import MetadataExtractor, DocumentMetadata
from src.preprocessing.legal_structure_parser import LegalStructureParser, LegalStructure, LegalUnit, LegalUnitType
from src.preprocessing.amendment_parser import AmendmentParser, AmendmentReference, AmendmentAction
from src.preprocessing.pipeline import PreprocessingPipeline, PreprocessedDocument, PreprocessingConfig
from src.preprocessing.patterns import DIGIT_MAP, HAMZA_MAP

__all__ = [
    # Pipeline
    "PreprocessingPipeline",
    "PreprocessedDocument",
    "PreprocessingConfig",
    
    # Cleaners
    "TextCleaner",
    "CleanedText",
    "CleaningConfig",
    
    # Normalizers
    "ArabicNormalizer",
    "NormalizationResult",
    "NormalizationConfig",
    
    # Metadata
    "MetadataExtractor",
    "DocumentMetadata",
    
    # Structure
    "LegalStructureParser",
    "LegalStructure",
    "LegalUnit",
    "LegalUnitType",
    
    # Amendments
    "AmendmentParser",
    "AmendmentReference",
    "AmendmentAction",
    
    # Patterns
    "DIGIT_MAP",
    "HAMZA_MAP",
]

__version__ = "1.0.0"