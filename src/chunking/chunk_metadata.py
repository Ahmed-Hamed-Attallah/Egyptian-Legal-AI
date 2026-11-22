import uuid
import dataclasses
from typing import Optional, Dict, Any
from src.preprocessing.pipeline import PreprocessedDocument
from src.preprocessing.legal_structure_parser import LegalUnit, LegalUnitType

def _safe_uuid(doc_id_str: str) -> uuid.UUID:
    try:
        return uuid.UUID(doc_id_str)
    except (ValueError, AttributeError):
        return uuid.uuid5(uuid.NAMESPACE_DNS, doc_id_str)

def build_chunk_metadata(
    preprocessed_doc: PreprocessedDocument,
    chunk_index: int,
    chunking_strategy: str,
    article_number: Optional[str] = None,
    section_title: Optional[str] = None,
    parent_unit: Optional[LegalUnit] = None,
    chunk_part: Optional[int] = None
) -> Dict[str, Any]:
    doc_meta = dataclasses.asdict(preprocessed_doc.metadata) if dataclasses.is_dataclass(preprocessed_doc.metadata) else {}
    metadata = {
        "document_id": _safe_uuid(preprocessed_doc.doc_id),
        "source_path": preprocessed_doc.source_path,
        "chunk_index": chunk_index,
        "chunking_strategy": chunking_strategy,
        "doc_type": doc_meta.get("doc_type"),
        "law_number": doc_meta.get("law_number"),
        "law_year": doc_meta.get("law_year"),
        "law_name": doc_meta.get("title") or doc_meta.get("law_name"),
        "title": doc_meta.get("title"),
        "appeal_number": doc_meta.get("appeal_number"),
        "appeal_year": doc_meta.get("appeal_year"),
        "court": doc_meta.get("court"),
        "date": doc_meta.get("date"),
        "article_number": article_number,
        "section_title": section_title,
        "chunk_part": chunk_part,
    }
    
    if article_number and preprocessed_doc.amendments:
        relevant_amendments = [
            {
                "action": amd.action.value,
                "law_number": amd.amending_law_number,
                "law_year": amd.amending_law_year
            }
            for amd in preprocessed_doc.amendments
            if amd.target_article_number == article_number
        ]
        if relevant_amendments:
            metadata["amendments"] = relevant_amendments
            
    return {k: v for k, v in metadata.items() if v is not None}