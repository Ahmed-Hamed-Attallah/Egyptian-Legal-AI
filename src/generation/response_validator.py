# src/generation/response_validator.py
"""
Response validation for legal answers.
"""
import logging
import re
from typing import List, Optional
from src.domain.responses import GenerationResponse
from src.domain.citations import Citation
from src.domain.exceptions import LLMError

logger = logging.getLogger(__name__)

class ResponseValidator:
    """
    Validates generated legal responses.
    """
    
    INSUFFICIENT_EVIDENCE_PATTERNS = [
        "لا تتوفر معلومات كافية",
        "المعلومات غير كافية",
        "لا يمكن الإجابة",
        "Insufficient information",
        "Cannot answer",
        "Not enough information"
    ]

    def validate_response(self, response: GenerationResponse, 
                         citations: List[Citation],
                         has_context: bool) -> dict:
        """
        Validate a generated response.
        Returns validation result with status and issues.
        """
        issues = []
        
        # Check for empty response
        if not response.content or not response.content.strip():
            issues.append("Empty response generated")
            return {"valid": False, "issues": issues, "status": "empty_response"}

        # Check for insufficient evidence indicators
        has_insufficient_indicator = any(
            pattern in response.content for pattern in self.INSUFFICIENT_EVIDENCE_PATTERNS
        )
        
        if has_insufficient_indicator:
            if not has_context:
                return {"valid": True, "issues": [], "status": "no_evidence"}
            else:
                issues.append("Model indicated insufficient evidence despite context provided")
                return {"valid": False, "issues": issues, "status": "insufficient_evidence"}

        # Check citation consistency
        if citations:
            # Verify citations are referenced (basic check)
            citation_ids = [c.source_id for c in citations]
            # In a real implementation, you'd check if citations are actually referenced in the text
            # For now, we just verify they exist
            if not citation_ids:
                issues.append("No valid citations provided")
        
        # Check for potential hallucination indicators
        hallucination_indicators = self._check_hallucination_indicators(response.content)
        if hallucination_indicators:
            issues.extend(hallucination_indicators)

        # Determine overall validity
        is_valid = len(issues) == 0
        status = "valid" if is_valid else "invalid"

        return {
            "valid": is_valid,
            "issues": issues,
            "status": status
        }

    def _check_hallucination_indicators(self, content: str) -> List[str]:
        """
        Check for potential hallucination indicators.
        """
        issues = []
        
        # Check for suspicious patterns (e.g., multiple law numbers without context)
        law_pattern = r'القانون رقم \d+ لسنة \d+'
        law_matches = re.findall(law_pattern, content)
        
        # If many law references without clear structure, might be hallucination
        if len(law_matches) > 5:
            issues.append("Multiple law references detected - potential hallucination")

        return issues