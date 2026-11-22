"""
Unit tests for the chunk quality filter.
"""
import uuid

import pytest

from src.chunking.quality_filter import (
    DropReason,
    filter_chunks,
    is_garbage_chunk,
)
from src.domain.chunks import Chunk, ChunkType


def _make_chunk(content: str, index: int = 0) -> Chunk:
    return Chunk(
        document_id=uuid.uuid4(),
        chunk_index=index,
        content=content,
        chunk_type=ChunkType.ARTICLE,
    )


class TestEmptyAndShort:
    def test_empty_is_rejected(self):
        _, report = filter_chunks([_make_chunk("")])
        assert report.total_rejected == 1
        assert report.by_reason[DropReason.EMPTY] == 1

    def test_short_is_rejected(self):
        _, report = filter_chunks([_make_chunk("نص قصير جدا")])
        assert report.total_rejected == 1
        assert report.by_reason[DropReason.TOO_SHORT] == 1


class TestMojibake:
    def test_mojibake_rejected(self):
        garbage = "مؠخالفةخ ادةؠ12- لغاة؂ ادةؠ13- لغاة؂ ادةؠ14- 9ليؠلمؠ'لكتابؠنؠرفضؠبولؠ حيفةؠ'لدعوي"
        assert is_garbage_chunk(garbage)
        _, report = filter_chunks([_make_chunk(garbage)])
        assert report.by_reason.get(DropReason.MOJIBAKE, 0) + \
               report.by_reason.get(DropReason.GLUED_QUOTES, 0) >= 1

    def test_clean_arabic_accepted(self):
        clean = (
            "المادة الأولى: تسري أحكام هذا القانون على جميع العاملين "
            "في الدولة، ويعتبر العقد شريعة المتعاقدين ولا يجوز نقضه "
            "ولا تعديله إلا باتفاق الطرفين أو للأسباب التي يقررها القانون."
        )
        assert not is_garbage_chunk(clean)
        chunks, report = filter_chunks([_make_chunk(clean)])
        assert len(chunks) == 1
        assert report.total_accepted == 1


class TestPdfUiGarbage:
    def test_repeated_index_label_rejected(self):
        loop = " ".join(["رجوع للفهرس"] * 50)
        assert is_garbage_chunk(loop)
        _, report = filter_chunks([_make_chunk(loop)])
        assert DropReason.REPETITIVE_LOOP in report.by_reason or \
               DropReason.PDF_UI_GARBAGE in report.by_reason

    def test_exit_button_rejected(self):
        text = "اضغط للخروج النهائي اضغط للخروج النهائي" * 10
        assert is_garbage_chunk(text)


class TestMixedQualityBatch:
    def test_good_and_bad_together(self):
        good = _make_chunk(
            "المادة 147 من القانون المدني: العقد شريعة المتعاقدين، "
            "فلا يجوز نقضه ولا تعديله إلا باتفاق الطرفين أو للأسباب "
            "التي يقررها القانون. ويعتبر هذا النص من أهم النصوص في "
            "القانون المدني المصري.",
            index=0,
        )
        bad = _make_chunk("مؠخالفةخ ةؠ12- ةؠ13- ةؠ14- ةؠ15-", index=1)
        empty = _make_chunk("", index=2)

        accepted, report = filter_chunks([good, bad, empty])

        assert len(accepted) == 1
        assert accepted[0] is good
        assert report.total_input == 3
        assert report.total_accepted == 1
        assert report.total_rejected == 2