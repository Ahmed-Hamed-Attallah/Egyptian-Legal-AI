"""
Domain models for legal sources.

This module defines the abstract legal entities (Laws, Decrees, Regulations).
A LegalSource represents the logical law itself, distinct from the physical 
Document (file) that contains it. This allows grouping multiple versions 
and amendments under a single legal authority.
"""

import uuid
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class LegalSourceType(str, Enum):
    """
    Categorizes the legal hierarchy and nature of the source.
    """
    CONSTITUTION = "constitution"             # دستور
    LAW = "law"                               # قانون
    DECREE_LAW = "decree_law"                 # مرسوم بقانون
    PRESIDENTIAL_DECREE = "presidential_decree" # قرار جمهوري
    PRIME_MINISTER_DECREE = "pm_decree"       # قرار رئيس مجلس الوزراء
    MINISTERIAL_DECREE = "ministerial_decree" # قرار وزاري
    REGULATION = "regulation"                 # لائحة
    JUDICIAL_PRECEDENT = "judicial_precedent" # سابقة قضائية / حكم


class LegalSourceStatus(str, Enum):
    """
    Tracks the validity of the legal source.
    """
    ACTIVE = "active"       # ساري
    REPEALED = "repealed"   # ملغي
    AMENDED = "amended"     # معدل (exists but modified)
    DRAFT = "draft"         # مسودة


# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------

class LegalSource(BaseModel):
    """
    Represents an abstract legal entity (e.g., 'The Egyptian Civil Code').
    
    This model focuses on the legal metadata and authority, 
    independent of the physical file format or storage.
    """

    # Pydantic V2 configuration
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    # Core Identifiers
    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for this legal entity."
    )
    name: str = Field(
        ..., 
        min_length=1, 
        description="Full official Arabic title (e.g., 'القانون المدني المصري')."
    )
    short_name: Optional[str] = Field(
        default=None, 
        description="Common abbreviation (e.g., 'القانون المدني')."
    )

    # Legal Classification
    law_number: Optional[str] = Field(
        default=None, 
        description="Official number (e.g., '131'). String to handle complex numbering."
    )
    year: Optional[int] = Field(
        default=None, 
        description="Year of issuance (e.g., 1948)."
    )
    type: LegalSourceType = Field(
        ..., 
        description="Category of the source (e.g., LAW, DECREE)."
    )
    status: LegalSourceStatus = Field(
        default=LegalSourceStatus.ACTIVE, 
        description="Current validity status."
    )

    # Authority & Jurisdiction
    issuing_authority: str = Field(
        ..., 
        description="Who issued it (e.g., 'مجلس النواب', 'رئيس الجمهورية')."
    )
    jurisdiction: str = Field(
        default="Arab Republic of Egypt", 
        description="Geographic scope of the law."
    )

    # Temporal Data
    effective_date: Optional[date] = Field(
        default=None, 
        description="Date when the law came into force."
    )

    # Descriptive Data
    summary: Optional[str] = Field(
        default=None, 
        description="Brief description for UI display."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Extra legal metadata (keywords, topics, etc.)."
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Timestamp of record creation."
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, 
        description="Timestamp of last update."
    )

    # -------------------------------------------------------------------------
    # Business Logic Methods
    # -------------------------------------------------------------------------

    def get_official_reference(self) -> str:
        """
        Generates the standard legal citation format.
        Example: 'Law No. 131 of 1948' or 'Presidential Decree No. 50 of 2020'.
        """
        type_map = {
            LegalSourceType.LAW: "Law",
            LegalSourceType.DECREE_LAW: "Decree-Law",
            LegalSourceType.PRESIDENTIAL_DECREE: "Presidential Decree",
            LegalSourceType.PRIME_MINISTER_DECREE: "Prime Minister Decree",
            LegalSourceType.MINISTERIAL_DECREE: "Ministerial Decree",
            LegalSourceType.REGULATION: "Regulation",
            LegalSourceType.CONSTITUTION: "Constitution",
            LegalSourceType.JUDICIAL_PRECEDENT: "Judicial Precedent",
        }
        
        type_str = type_map.get(self.type, "Legal Source")
        num_str = f" No. {self.law_number}" if self.law_number else ""
        year_str = f" of {self.year}" if self.year else ""
        
        return f"{type_str}{num_str}{year_str}"

    def is_currently_valid(self) -> bool:
        """
        Checks if the law is active or amended (usable).
        Returns False if repealed or draft.
        """
        return self.status in [LegalSourceStatus.ACTIVE, LegalSourceStatus.AMENDED]

    def is_primary_legislation(self) -> bool:
        """
        Checks if this is a high-level law (Constitution, Law, Decree-Law) 
        vs. a secondary regulation or decree.
        """
        return self.type in [
            LegalSourceType.CONSTITUTION,
            LegalSourceType.LAW,
            LegalSourceType.DECREE_LAW
        ]