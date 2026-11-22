"""
Quality report aggregation across documents.

Provides a small facade over the chunk quality filter so the API can
surface per-document drop statistics in the document detail response
without importing chunking internals directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DocumentQualityReport:
    """Aggregate quality-filter statistics for one ingested document."""

    document_id: str
    total_chunks_produced: int = 0
    total_chunks_accepted: int = 0
    total_chunks_dropped: int = 0
    drop_reasons: Dict[str, int] = field(default_factory=dict)
    samples: List[Dict] = field(default_factory=list)

    @property
    def drop_rate(self) -> float:
        if self.total_chunks_produced <= 0:
            return 0.0
        return self.total_chunks_dropped / self.total_chunks_produced

    def to_api_dict(self) -> Dict:
        return {
            "document_id": self.document_id,
            "total_chunks_produced": self.total_chunks_produced,
            "total_chunks_accepted": self.total_chunks_accepted,
            "total_chunks_dropped": self.total_chunks_dropped,
            "drop_rate": round(self.drop_rate, 4),
            "drop_reasons": dict(self.drop_reasons),
            "samples": list(self.samples),
        }