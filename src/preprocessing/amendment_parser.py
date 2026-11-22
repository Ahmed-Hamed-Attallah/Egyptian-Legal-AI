# src/preprocessing/amendment_parser.py
"""
Amendment parser for Egyptian legal documents.
Identifies amendment directives (amended, repealed, replaced, added) and 
links them to the specific legal articles they modify.

Crucial for Phase 11 (Document Versioning) and Phase 5 (Chunk Metadata).
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# Import LegalStructure and LegalUnit to link amendments to specific articles
# If circular imports occur, use TYPE_CHECKING or pass raw text only.
try:
    from src.preprocessing.legal_structure_parser import LegalStructure, LegalUnit, LegalUnitType
except Exception as e:
    print(f"Error Importing LegalStructure: {e}")


class AmendmentAction(str, Enum):
    """Types of legal amendments."""
    AMENDED = "amended"       # عدلت / تعديل
    REPLACED = "replaced"     # مستبدلة / يُستبدل
    REPEALED = "repealed"     # أُلغيت / يُلغى / الغي
    ADDED = "added"           # مضافة / أُضيف


@dataclass(frozen=True, slots=True)
class AmendmentReference:
    """
    Represents a detected amendment directive.
    
    Attributes:
        action: The type of amendment (AMENDED, REPLACED, etc.).
        amending_law_number: The number of the law/decision making the change.
        amending_law_year: The year of the amending law/decision.
        amending_law_type: The type of the amending instrument (قانون, مرسوم, قرار).
        target_article_number: The number of the article being amended (if resolved).
        char_start: Start index in the cleaned text.
        char_end: End index in the cleaned text.
        original_text_span: The exact text matched (for provenance/debugging).
    """
    action: AmendmentAction
    amending_law_number: str
    amending_law_year: str
    amending_law_type: str
    target_article_number: Optional[str]
    char_start: int
    char_end: int
    original_text_span: str


class AmendmentParser:
    """
    Stateless parser that detects amendment directives in legal text.
    Uses regex for speed and deterministic behavior.
    """

    def __init__(self):
        # Robust regex to handle messy PDF/OCR text (e.g., "عدلت بالقانون رقم 18 لسنة 1999")
        # Allows up to 30 non-digit characters between the verb and the law number.
        self._amendment_pattern = re.compile(
            r"(?P<verb>عدلت|تعديل|يُلغى|يُستبدل|أُضيف|مستبدلة|مضافة|أُلغيت|الغي|غيت|يلغى|تستبدل|تضاف|سبق\s+تعديل)"
            r"[^\d]{0,40}" 
            r"(?:بالقانون|بمرسوم|بقرار|بقانون|بمرسوم|بقرار)"
            r"[^\d]{0,15}"
            r"(?:رقم|)"
            r"[^\d]{0,5}"
            r"(?P<law_num>\d+)"
            r"[^\d]{0,15}"
            r"(?:لسنة|لعام|)"
            r"[^\d]{0,5}"
            r"(?P<law_year>\d{4})"
        )
        
        self._law_type_pattern = re.compile(r"(بالقانون|بمرسوم|بقرار|بقانون|بمرسوم|بقرار)")

    def parse(self, text: str, structure: Optional[object] = None) -> List[AmendmentReference]:
        """
        Parses text for amendment directives.
        
        Args:
            text: The cleaned text.
            structure: Optional LegalStructure object to resolve which article is being amended.
            
        Returns:
            List of AmendmentReference objects.
        """
        if not text:
            return []

        amendments = []
        
        for match in self._amendment_pattern.finditer(text):
            verb = match.group("verb")
            law_num = match.group("law_num")
            law_year = match.group("law_year")
            
            # Extract law type (قانون, مرسوم, قرار)
            law_type_match = self._law_type_pattern.search(match.group(0))
            law_type = law_type_match.group(1).replace("ب", "") if law_type_match else "unknown"
            
            # Classify action
            action = self._classify_action(verb)
            
            # Resolve target article if structure is provided
            target_article = self._resolve_target_article(match.start(), structure) if structure else None
            
            amendments.append(AmendmentReference(
                action=action,
                amending_law_number=law_num,
                amending_law_year=law_year,
                amending_law_type=law_type,
                target_article_number=target_article,
                char_start=match.start(),
                char_end=match.end(),
                original_text_span=match.group(0)
            ))
            
        return amendments

    def _classify_action(self, verb: str) -> AmendmentAction:
        """Maps the matched Arabic verb to the AmendmentAction enum."""
        verb = verb.strip()
        if verb in ("عدلت", "تعديل", "سبق تعديل"):
            return AmendmentAction.AMENDED
        elif verb in ("مستبدلة", "يُستبدل", "تستبدل"):
            return AmendmentAction.REPLACED
        elif verb in ("أُلغيت", "يُلغى", "الغي", "غيت", "يلغى"):
            return AmendmentAction.REPEALED
        elif verb in ("مضافة", "أُضيف", "تضاف"):
            return AmendmentAction.ADDED
        return AmendmentAction.AMENDED # Default fallback

    def _resolve_target_article(self, amendment_start_char: int, structure: object) -> Optional[str]:
        """
        Heuristic to find the article immediately preceding the amendment text.
        In Egyptian laws, amendment notes usually appear at the end of the article body 
        or in a footnote immediately following it.
        """
        if not structure or not hasattr(structure, "units"):
            return None
            
        # Find the closest LegalUnit of type ARTICLE that ends before the amendment starts
        closest_article = None
        min_distance = float('inf')
        
        for unit in structure.units:
            if hasattr(unit, "unit_type") and unit.unit_type == LegalUnitType.ARTICLE:
                if unit.end_offset <= amendment_start_char:
                    distance = amendment_start_char - unit.end_offset
                    # Only consider it the target if it's reasonably close (e.g., within 500 chars)
                    if distance < min_distance and distance < 500:
                        min_distance = distance
                        closest_article = unit
                        
        return closest_article.number if closest_article else None