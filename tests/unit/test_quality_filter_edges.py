"""Edge cases and threshold contracts for the chunk quality gate.

Complements test_chunk_quality_filter.py (which covers the happy paths) by
pinning behaviour that is easy to change accidentally, plus the one documented
gap: MAX_PDF_UI_SHARE is declared but never evaluated.
"""

from __future__ import annotations

import uuid

import pytest

from src.chunking.quality_filter import (
    DropReason,
    MIN_CHUNK_CHARS,
    filter_chunks,
    is_garbage_chunk,
)
from src.domain.chunks import Chunk, ChunkType

# Real Article 147 — varied vocabulary, no extraction artefacts.
ARTICLE_147 = (
    "العقد شريعة المتعاقدين، فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين، "
    "أو للأسباب التي يقررها القانون. ومع ذلك، إذا طرأت حوادث استثنائية عامة "
    "لم يكن من الممكن توقعها، وترتب على حدوثها أن تنفيذ الالتزام التعاقدي، "
    "وإن لم يصبح مستحيلاً، صار مرهقاً للمدين، بحيث يهدده بخسارة فادحة، "
    "جاز للقاضي تبعاً للظروف، وبعد أن يوازن بين مصلحة الطرفين، أن يرد "
    "الالتزام المرهق إلى الحد المعقول، ويكون كل اتفاق يقضي بغير ذلك باطلاً."
)


def _chunk(content: str, index: int = 0) -> Chunk:
    return Chunk(
        document_id=uuid.uuid4(),
        chunk_index=index,
        content=content,
        chunk_type=ChunkType.ARTICLE,
    )


class TestBoundaryBehaviour:
    def test_real_article_is_accepted(self):
        assert not is_garbage_chunk(ARTICLE_147)
        clean, report = filter_chunks([_chunk(ARTICLE_147)])
        assert len(clean) == 1
        assert report.total_accepted == 1
        assert report.total_rejected == 0

    def test_whitespace_only_is_too_short_not_empty(self):
        """`not text` is False for whitespace, so it falls through to the length gate."""
        _, report = filter_chunks([_chunk("     ")])
        assert report.by_reason[DropReason.TOO_SHORT] == 1
        assert DropReason.EMPTY not in report.by_reason

    def test_length_gate_uses_stripped_text(self):
        padded = " " * 80 + "نص قصير" + " " * 80
        _, report = filter_chunks([_chunk(padded)])
        assert report.by_reason[DropReason.TOO_SHORT] == 1

    def test_threshold_constant_is_the_one_applied(self):
        assert MIN_CHUNK_CHARS == 60
        assert is_garbage_chunk("قانون " * 3)      # well under 60
        assert not is_garbage_chunk(ARTICLE_147)   # well over 60


class TestReportAccounting:
    def test_input_list_is_never_mutated(self):
        chunks = [_chunk(ARTICLE_147, 0), _chunk("", 1)]
        snapshot = list(chunks)
        filter_chunks(chunks)
        assert chunks == snapshot

    def test_totals_are_consistent(self):
        chunks = [_chunk(ARTICLE_147, i) for i in range(3)] + [_chunk("", 9)]
        clean, report = filter_chunks(chunks)
        assert report.total_input == 4
        assert report.total_accepted == len(clean) == 3
        assert report.total_rejected == 1
        assert report.total_input == report.total_accepted + report.total_rejected

    def test_rejected_samples_are_capped_at_five(self):
        _, report = filter_chunks([_chunk("", i) for i in range(20)])
        assert report.total_rejected == 20
        assert len(report.rejected_samples) == 5

    def test_by_reason_counts_every_rejection(self):
        _, report = filter_chunks([_chunk("", i) for i in range(7)])
        assert sum(report.by_reason.values()) == report.total_rejected

    def test_empty_batch_is_a_noop(self):
        clean, report = filter_chunks([])
        assert clean == []
        assert report.total_input == 0
        assert report.by_reason == {}


class TestArabicArtefacts:
    def test_glued_apostrophe_is_detected(self):
        # 'لكتاب instead of الكتاب — a broken ToUnicode CMap signature.
        text = ARTICLE_147 + " " + " ".join(["'لكتاب"] * 20)
        assert is_garbage_chunk(text)

    def test_impossible_glyphs_are_detected(self):
        text = ARTICLE_147 + " " + "ؠ" * 10
        assert is_garbage_chunk(text)

    def test_clean_arabic_with_diacritics_is_accepted(self):
        assert not is_garbage_chunk(
            "الْعَقْدُ شَرِيعَةُ الْمُتَعَاقِدَيْنِ، فَلَا يَجُوزُ نَقْضُهُ "
            "وَلَا تَعْدِيلُهُ إِلَّا بِاتِّفَاقِ الطَّرَفَيْنِ أَوْ لِلْأَسْبَابِ "
            "الَّتِي يُقَرِّرُهَا الْقَانُونُ."
        )

    def test_indian_digits_are_not_artefacts(self):
        assert not is_garbage_chunk(
            "المادة ١٤٧ من القانون المدني رقم ١٣١ لسنة ١٩٤٨: العقد شريعة "
            "المتعاقدين فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين."
        )


class TestDocumentedGap:
    @pytest.mark.xfail(
        reason="DEFECT: MAX_PDF_UI_SHARE=0.10 is declared but never read. "
               "_detect uses 'any UI match AND len<250', so a long chunk "
               "dominated by navigation labels is accepted.",
        strict=False,
    )
    def test_long_ui_dominated_chunk_should_be_rejected(self):
        filler = (
            "تسري أحكام هذا القانون على جميع العاملين في الدولة وفقا لأحكامه "
            "المبينة في المواد التالية من هذا الباب والفصول الخاصة به. "
        )
        garbage = (filler + "رجوع للفهرس اضغط للخروج الصفحة التالية ") * 4
        assert len(garbage) > 250
        assert is_garbage_chunk(garbage)