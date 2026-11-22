"""
Intermediate models for the ingestion phase.
These act as a bridge between raw file parsers and the strict Domain Models.
"""
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class ParsedResult:
    """
    Lightweight container for extracted data from a file.
    Parsers return this. The IngestionPipeline maps this to the strict Domain Document.
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)