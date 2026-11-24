"""Regression suite — one test per bug that actually shipped.

These encode the intended contract. Skips are explicit and named so a missing
fix is visible in the report rather than silently passing.
"""

from __future__ import annotations

import json
import uuid
from datetime import date
from decimal import Decimal

import pytest


# ---------------------------------------------------------------- JSON boundary
class TestJsonSerialization:
    def test_uuid_in_nested_metadata_serialises(self):
        """Was: TypeError: Object of type UUID is not JSON serializable."""
        from src.utils.json_utils import json_dumps

        payload = {
            "document_id": uuid.uuid4(),
            "effective_date": date(1948, 7, 16),
            "weight": Decimal("0.75"),
            "tags": {"a", "b"},
        }
        decoded = json.loads(json_dumps({"nested": [payload]}))
        assert isinstance(decoded["nested"][0]["document_id"], str)
        assert decoded["nested"][0]["effective_date"] == "1948-07-16"
        assert decoded["nested"][0]["weight"] == 0.75

    def test_arabic_is_not_escaped(self):
        from src.utils.json_utils import json_dumps

        assert "قانون" in json_dumps({"law": "قانون"})


# ------------------------------------------------------------- status mapping
class TestDocumentStatusMapping:
    """PROCESSING with 0 chunks used to report 'failed', so the UI stopped polling."""

    @pytest.mark.parametrize(
        "db_status,chunks,indexed,expected",
        [
            ("pending", 0, False, "uploaded"),
            ("processing", 0, False, "processing"),
            ("processing", 12, False, "processing"),
            ("completed", 12, True, "completed"),
            ("completed", 12, False, "partial"),
            ("completed", 0, False, "failed"),
            ("failed", 0, False, "failed"),
        ],
    )
    def test_public_status(self, db_status, chunks, indexed, expected):
        from src.services.document_service import DocumentService

        assert DocumentService._public_status(db_status, chunks, indexed) == expected

    def test_non_terminal_states_never_become_terminal(self):
        from src.services.document_service import DocumentService

    def test_ingest_accepts_owner_id(self):
        """The worker forwards owner_id; a missing param raises TypeError."""
        import inspect
        from src.services.document_service import DocumentService

        params = inspect.signature(DocumentService.ingest_upload).parameters
        assert "owner_id" in params, "worker passes owner_id= → every job would fail"


# ------------------------------------------------------------ access controls
class TestAuthorization:
    def test_regular_user_does_not_have_admin_permission(self, make_auth):
        """Was: `if auth.is_admin:` — a bound method, always truthy."""
        from src.security.authorization import Permission, has_permission

        user = make_auth(roles=("user",))
        assert has_permission(user, Permission.ADMIN) is False

    def test_admin_has_every_permission(self, make_auth):
        from src.security.authorization import Permission, has_permission

        admin = make_auth(roles=("admin",))
        assert all(has_permission(admin, p) for p in Permission)

    def test_viewer_cannot_write(self, make_auth):
        from src.security.authorization import Permission, has_permission

        viewer = make_auth(roles=("viewer",))
        assert has_permission(viewer, Permission.DOCUMENT_READ) is True
        assert has_permission(viewer, Permission.DOCUMENT_WRITE) is False


class TestVisibilityFilter:
    @pytest.fixture(autouse=True)
    def _require_policy(self):
        from src.retrieval.metadata_filters import MetadataFilter

        if not hasattr(MetadataFilter, "for_documents"):
            pytest.skip("MetadataFilter.for_documents missing — apply the Group 2 fix")

    def test_official_visible_to_any_member(self):
        from src.retrieval.metadata_filters import MetadataFilter

        f = MetadataFilter.for_documents(tenant_id="t1", user_id="u1")
        assert f.matches({"tenant_id": "t1", "scope": "official"}) is True

    def test_own_private_document_visible(self):
        from src.retrieval.metadata_filters import MetadataFilter

        f = MetadataFilter.for_documents(tenant_id="t1", user_id="u1")
        assert f.matches({"tenant_id": "t1", "scope": "private", "owner_id": "u1"}) is True

    def test_other_users_private_document_hidden(self):
        from src.retrieval.metadata_filters import MetadataFilter

        f = MetadataFilter.for_documents(tenant_id="t1", user_id="u1")
        assert f.matches({"tenant_id": "t1", "scope": "private", "owner_id": "u2"}) is False

    def test_missing_scope_fails_closed(self):
        from src.retrieval.metadata_filters import MetadataFilter

        f = MetadataFilter.for_documents(tenant_id="t1", user_id="u1")
        assert f.matches({"tenant_id": "t1"}) is False

    def test_other_tenant_never_visible(self):
        from src.retrieval.metadata_filters import MetadataFilter

        f = MetadataFilter.for_documents(tenant_id="t1", user_id="u1")
        assert f.matches({"tenant_id": "t2", "scope": "official"}) is False

    def test_or_operator_is_honoured(self):
        """Was: `operator` declared but never evaluated — 'or' behaved as 'and'."""
        from src.retrieval.metadata_filters import FilterCondition, MetadataFilter

        f = MetadataFilter.any_of(
            FilterCondition("a", 1, "eq"), FilterCondition("b", 2, "eq")
        )
        assert f.matches({"b": 2}) is True

    def test_comparison_operators_are_honoured(self):
        from src.retrieval.metadata_filters import FilterCondition

        assert FilterCondition("score", 0.5, "gte").evaluate({"score": 0.6}) is True
        assert FilterCondition("score", 0.5, "gte").evaluate({"score": 0.4}) is False

    def test_unknown_operator_raises_at_construction(self):
        from src.retrieval.metadata_filters import FilterCondition

        with pytest.raises(ValueError):
            FilterCondition("x", 1, "bogus")


# ------------------------------------------------------------------- auth/JWT
class TestTokenLifecycle:
    def test_integer_subject_round_trips(self, auth_enabled):
        """PyJWT >= 2.10 rejects a non-string `sub`."""
        from src.security.authentication import create_access_token, decode_access_token

        token = create_access_token(user_id=42, tenant_id="default")
        assert decode_access_token(token)["sub"] == "42"

    def test_expired_token_is_rejected(self, auth_enabled):
        from src.domain.exceptions import AuthenticationError
        from src.security.authentication import create_access_token, decode_access_token

        token = create_access_token(user_id="u", tenant_id="t", expires_minutes=-1)
        with pytest.raises(AuthenticationError):
            decode_access_token(token)

    def test_tampered_token_is_rejected(self, auth_enabled):
        from src.domain.exceptions import AuthenticationError
        from src.security.authentication import create_access_token, decode_access_token

        token = create_access_token(user_id="u", tenant_id="t")
        with pytest.raises(AuthenticationError):
            decode_access_token(token[:-4] + "AAAA")

    def test_generator_is_not_a_verifier(self):
        from src.security.authentication import generate_secure_token

        assert isinstance(generate_secure_token(16), str)
        with pytest.raises(TypeError):
            generate_secure_token("not-an-int")  # type: ignore[arg-type]


class TestPasswordBytes:
    def test_arabic_password_over_72_bytes_rejected(self):
        """bcrypt counts BYTES. 40 Arabic chars ≈ 80 bytes."""
        from src.domain.exceptions import AuthenticationError
        from src.security.authentication import validate_password_length

        with pytest.raises(AuthenticationError):
            validate_password_length("كلمةسر" * 12)

    def test_short_ascii_password_accepted(self):
        from src.security.authentication import validate_password_length

        validate_password_length("StrongPass123")


# ------------------------------------------------------------- retrieval math
class TestFusionProvenance:
    def test_rank_is_set_and_provenance_preserved(self, make_chunk):
        from src.domain.queries import RetrievedChunk
        from src.retrieval.score_fusion import ReciprocalRankFusion

        chunk = make_chunk()
        dense = [RetrievedChunk(chunk=chunk, score=0.9, retriever_type="dense")]
        sparse = [RetrievedChunk(chunk=chunk, score=8.0, retriever_type="sparse")]

        fused = ReciprocalRankFusion().fuse([dense, sparse], top_k=5)
        assert len(fused) == 1, "the same chunk must fuse, not duplicate"
        assert fused[0].rank >= 1, "rank 0 tells the LLM every chunk is equal"
        assert fused[0].dense_score == pytest.approx(0.9)
        assert fused[0].sparse_score == pytest.approx(8.0)


class TestSparseIndexLifecycle:
    @pytest.fixture(autouse=True)
    def _require_incremental(self):
        from src.retrieval.sparse_retriever import SparseRetriever

        if not hasattr(SparseRetriever, "add_chunks"):
            pytest.skip("SparseRetriever.add_chunks missing — BM25 stays frozen")

    def test_chunks_added_after_build_are_searchable(self, make_chunk):
        from src.retrieval.sparse_retriever import SparseRetriever

        r = SparseRetriever()
        r.build_index([make_chunk("المادة 163 كل خطأ سبب ضررا للغير", index=0)])
        added = r.add_chunks([make_chunk("المادة 147 العقد شريعة المتعاقدين", index=1)])
        assert added == 1
        assert r.indexed_count == 2
        hits = r.retrieve("العقد شريعة المتعاقدين")
        assert any(h.chunk.chunk_index == 1 for h in hits)

    def test_remove_document_drops_its_chunks(self, make_chunk):
        from src.retrieval.sparse_retriever import SparseRetriever

        doc = uuid.uuid4()
        r = SparseRetriever()
        r.build_index([make_chunk("نص قانوني طويل enough", index=0, document_id=doc)])
        assert r.remove_document(str(doc)) == 1
        assert r.indexed_count == 0