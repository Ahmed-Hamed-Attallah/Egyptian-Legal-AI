"""
Domain models for legal source versions.

This module represents temporal versions of a LegalSource.

A LegalSource is the abstract legal entity, while LegalVersion represents
a specific temporal state of that source, such as:
- original issuance
- amendment
- consolidated current text
- correction
- repeal version

This file contains domain-only logic. It does not handle database persistence,
retrieval, embeddings, or citation generation.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# -----------------------------------------------------------------------------
# Internal Helpers
# -----------------------------------------------------------------------------

def _utc_now() -> datetime:
    """
    Return a timezone-aware UTC timestamp.

    Mission:
        Avoid deprecated datetime.utcnow() and keep domain timestamps consistent.
    """
    return datetime.now(timezone.utc)


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class LegalVersionStatus(str, Enum):
    """
    Lifecycle status of a legal version.

    Note:
        SUPERSEDED does not mean historically invalid.
        A superseded version may still be relevant for historical legal queries.
    """

    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REPEALED = "repealed"
    UNKNOWN = "unknown"


class LegalVersionType(str, Enum):
    """
    Type of legal version.

    Used to distinguish original texts from amendments and consolidated texts.
    """

    ORIGINAL = "original"
    AMENDMENT = "amendment"
    CONSOLIDATED = "consolidated"
    CORRECTION = "correction"
    REPEAL = "repeal"


# -----------------------------------------------------------------------------
# Domain Model
# -----------------------------------------------------------------------------

class LegalVersion(BaseModel):
    """
    Represents one temporal version of a legal source.

    Example:
        LegalSource:
            القانون المدني المصري

        LegalVersion:
            النسخة الأصلية الصادرة سنة 1948

        LegalVersion:
            نسخة معدلة بالقانون رقم ... لسنة ...

    Important:
        This model stores version metadata and simple legal-version rules.
        The service/repository layer is responsible for persistence and for
        ensuring that only one version is marked as current per LegalSource.
    """

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
        validate_assignment=True,
    )

    # -------------------------------------------------------------------------
    # Core Identifiers
    # -------------------------------------------------------------------------

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Unique identifier for this legal version.",
    )

    legal_source_id: uuid.UUID = Field(
        ...,
        description="Reference to the parent LegalSource.",
    )

    document_id: Optional[uuid.UUID] = Field(
        default=None,
        description=(
            "Optional reference to the physical Document containing this version. "
            "A version may exist before its file is uploaded."
        ),
    )

    # -------------------------------------------------------------------------
    # Version Identity
    # -------------------------------------------------------------------------

    version_number: int = Field(
        default=1,
        ge=1,
        description="Sequential version number within the same LegalSource.",
    )

    version_label: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "Human-readable version label, e.g. "
            "'النسخة الأصلية 1948' or 'تعديل سنة 2005'."
        ),
    )

    version_type: LegalVersionType = Field(
        default=LegalVersionType.ORIGINAL,
        description="Type of version: original, amendment, consolidated, etc.",
    )

    status: LegalVersionStatus = Field(
        default=LegalVersionStatus.UNKNOWN,
        description="Current lifecycle status of this version.",
    )

    is_current: bool = Field(
        default=False,
        description=(
            "Whether this version is marked as the current version. "
            "Only one version per LegalSource should usually be current."
        ),
    )

    # -------------------------------------------------------------------------
    # Temporal Validity
    # -------------------------------------------------------------------------

    effective_from: Optional[date] = Field(
        default=None,
        description="Date from which this version became effective.",
    )

    effective_to: Optional[date] = Field(
        default=None,
        description=(
            "Last date on which this version was effective. "
            "None means open-ended / still effective if current."
        ),
    )

    publication_date: Optional[date] = Field(
        default=None,
        description="Official publication date, if known.",
    )

    publication_reference: Optional[str] = Field(
        default=None,
        max_length=500,
        description=(
            "Official publication reference, e.g. الجريدة الرسمية، العدد ..."
        ),
    )

    # -------------------------------------------------------------------------
    # Amendment / Lineage Metadata
    # -------------------------------------------------------------------------

    change_summary: Optional[str] = Field(
        default=None,
        description="Short summary of what changed in this version.",
    )

    supersedes_version_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Previous LegalVersion ID replaced by this version.",
    )

    superseded_by_version_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Newer LegalVersion ID that replaced this version.",
    )

    amendment_source_id: Optional[uuid.UUID] = Field(
        default=None,
        description=(
            "Optional LegalSource ID of the law/decree that introduced "
            "this amendment."
        ),
    )

    # -------------------------------------------------------------------------
    # Extensibility / Timestamps
    # -------------------------------------------------------------------------

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Flexible metadata, e.g. amendment articles, gazette issue, "
            "ingestion notes, or manual verification flags."
        ),
    )

    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Timestamp of domain record creation.",
    )

    updated_at: datetime = Field(
        default_factory=_utc_now,
        description="Timestamp of last domain record update.",
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_version(self) -> "LegalVersion":
        """
        Validate legal-version consistency.

        Mission:
            Prevent impossible version states such as:
            - effective_to before effective_from
            - current version with a closed effective_to date
            - current version that is not active
            - version referencing itself as superseding/superseded version
        """
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_to < self.effective_from
        ):
            raise ValueError("effective_to cannot be earlier than effective_from.")

        if self.is_current and self.status != LegalVersionStatus.ACTIVE:
            raise ValueError("A current legal version must have ACTIVE status.")

        if self.is_current and self.effective_to is not None:
            raise ValueError("A current legal version cannot have effective_to set.")

        if self.supersedes_version_id == self.id:
            raise ValueError("A legal version cannot supersede itself.")

        if self.superseded_by_version_id == self.id:
            raise ValueError("A legal version cannot be superseded by itself.")

        return self

    # -------------------------------------------------------------------------
    # Business Logic Methods
    # -------------------------------------------------------------------------

    def has_linked_document(self) -> bool:
        """
        Check whether this version is linked to a physical Document.

        Mission:
            Useful during ingestion and citation validation.
        """
        return self.document_id is not None

    def is_effective_on(self, target_date: date) -> bool:
        """
        Check whether this version was effective on a specific date.

        Mission:
            Used later by temporal retrieval logic to answer questions like:
            'What was the law in 2010?'

        Important:
            A SUPERSEDED version can still be effective for a historical date.
            Therefore, this method mainly checks the temporal validity window.
        """
        if self.status == LegalVersionStatus.DRAFT:
            return False

        if self.effective_from is None:
            return False

        if target_date < self.effective_from:
            return False

        if self.effective_to is not None and target_date > self.effective_to:
            return False

        return True

    def is_current_version(self) -> bool:
        """
        Check whether this version is the current active version.

        Mission:
            Used by retrieval filters when the user asks about current law.
        """
        return (
            self.is_current
            and self.status == LegalVersionStatus.ACTIVE
            and self.effective_to is None
        )

    def is_amending_version(self) -> bool:
        """
        Check whether this version represents a legal change.

        Mission:
            Helps distinguish original/consolidated texts from amendments,
            corrections, and repeal-related versions.
        """
        return self.version_type in (
            LegalVersionType.AMENDMENT,
            LegalVersionType.CORRECTION,
            LegalVersionType.REPEAL,
        )

    def get_temporal_reference(self) -> str:
        """
        Build a readable Arabic temporal reference.

        Mission:
            Used for UI display, debugging, and later citation enrichment.

        Example:
            'تعديل سنة 2005 | السريان: 2005-03-01 → حتى الآن'
        """
        start = self.effective_from.isoformat() if self.effective_from else "غير محدد"

        if self.effective_to is not None:
            end = self.effective_to.isoformat()
        elif self.is_current_version():
            end = "حتى الآن"
        else:
            end = "غير محدد"

        return f"{self.version_label} | السريان: {start} ← {end}"

    def mark_superseded(
        self,
        end_date: date,
        superseded_by_version_id: Optional[uuid.UUID] = None,
    ) -> None:
        """
        Mark this version as superseded by a newer version.

        Mission:
            Encapsulates the basic state transition when a newer legal version
            replaces this one.

        Note:
            This method does not persist to the database.
            The service/repository layer must save the updated model.
        """
        if self.effective_from is not None and end_date < self.effective_from:
            raise ValueError("end_date cannot be earlier than effective_from.")

        if superseded_by_version_id == self.id:
            raise ValueError("A legal version cannot be superseded by itself.")

        self.is_current = False
        self.status = LegalVersionStatus.SUPERSEDED
        self.effective_to = end_date

        if superseded_by_version_id is not None:
            self.superseded_by_version_id = superseded_by_version_id

        self.updated_at = _utc_now()
        self._record_status_change(
            status=LegalVersionStatus.SUPERSEDED,
            reason="Version superseded by a newer legal version.",
        )

    def mark_active(self, effective_from: Optional[date] = None) -> None:
        """
        Mark this version as active and current.

        Mission:
            Encapsulates the basic state transition for a usable current version.

        Important:
            This method does not unset other active versions.
            The service/repository layer must enforce one current version
            per LegalSource.
        """
        if effective_from is not None:
            self.effective_from = effective_from

        if self.effective_from is None:
            raise ValueError("effective_from is required to mark a version active.")

        self.status = LegalVersionStatus.ACTIVE
        self.effective_to = None
        self.is_current = True
        self.updated_at = _utc_now()

        self._record_status_change(
            status=LegalVersionStatus.ACTIVE,
            reason="Version marked as active/current.",
        )

    def _record_status_change(
        self,
        status: LegalVersionStatus,
        reason: str,
    ) -> None:
        """
        Record a lightweight status-change event inside metadata.

        Mission:
            Provides simple audit/debug information without introducing
            a database audit model at this phase.
        """
        history = self.metadata.setdefault("status_history", [])

        if not isinstance(history, list):
            history = []
            self.metadata["status_history"] = history

        history.append(
            {
                "status": status.value,
                "reason": reason,
                "timestamp": _utc_now().isoformat(),
            }
        )