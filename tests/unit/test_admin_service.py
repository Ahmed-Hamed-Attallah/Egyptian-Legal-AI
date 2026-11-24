"""AdminService: subsystem independence and honest health reporting.

Restores the intent of the original test_7 / test_9 / test_10, which asserted
against a real API but were neutered by a fixture that patched
`src.config.settings.settings` — modules bind that singleton at import time,
so the mock was never seen by the code under test.

These use the real settings (isolated by tests/conftest.py) and MagicMock only
for the injected collaborators, which IS where mocking is correct.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.services.admin_service import AdminService


class TestHealthContract:
    def test_shape_is_stable(self):
        payload = AdminService().health()
        assert set(payload) == {"status", "components"}
        assert payload["status"] in {"ok", "degraded"}

    def test_no_collaborators_degrades_without_raising(self):
        """A cold start must report unavailable, not crash the readiness probe."""
        components = AdminService().health()["components"]
        assert components["chat"] == "unavailable"
        assert components["search"] == "unavailable"
        assert components["documents"] == "unavailable"
        assert components["postgres"] == "unknown"

    def test_postgres_exception_is_isolated(self):
        db = MagicMock()
        db.health_check.side_effect = Exception("Connection refused")
        payload = AdminService(
            chat_service=object(),
            search_service=object(),
            document_service=object(),
            database=db,
        ).health()
        assert payload["components"]["postgres"] == "error"
        assert payload["components"]["chat"] == "ok", "DB failure must not taint chat"
        assert payload["status"] == "degraded"

    def test_postgres_false_is_unreachable_not_error(self):
        db = MagicMock()
        db.health_check.return_value = False
        assert AdminService(database=db).health()["components"]["postgres"] == "unreachable"

    def test_overall_ok_requires_every_component_ok(self):
        db = MagicMock()
        db.health_check.return_value = True
        payload = AdminService(
            chat_service=object(),
            search_service=object(),
            document_service=object(),
            database=db,
        ).health()
        assert payload["status"] == "ok"

    @pytest.mark.xfail(
        reason="DEFECT: health() infers Qdrant from search_service presence and "
               "never probes the vector store. A live service over a dead Qdrant "
               "reports qdrant=ok.",
        strict=False,
    )
    def test_qdrant_should_be_probed_not_inferred(self):
        dead_store = MagicMock()
        dead_store.health_check.side_effect = ConnectionError("Qdrant down")
        search = MagicMock()
        search.vector_store = dead_store
        payload = AdminService(search_service=search).health()
        assert payload["components"]["qdrant"] != "ok"


class TestStatusContract:
    def test_missing_credentials_enumerated_without_router(self, app_settings):
        """conftest empties every API key, so all three must be listed."""
        llm = AdminService().status()["llm"]
        assert set(llm["missing_credentials"]) == {"groq", "openai", "gemini"}
        assert llm["primary_provider"] == app_settings.LLM_PRIMARY_PROVIDER
        assert llm["available"] == []

    def test_qdrant_degraded_without_search_service(self):
        assert AdminService().status()["qdrant"] == "degraded_or_uninitialized"

    def test_qdrant_degraded_when_retriever_missing(self):
        """A bare object has no hybrid_retriever attribute."""
        assert AdminService(search_service=object()).status()["qdrant"] == \
            "degraded_or_uninitialized"

    def test_postgres_error_does_not_affect_llm_block(self, app_settings):
        db = MagicMock()
        db.health_check.side_effect = Exception("boom")
        status = AdminService(database=db).status()
        assert status["postgres"] == "error"
        assert status["llm"]["primary_provider"] == app_settings.LLM_PRIMARY_PROVIDER

    def test_rag_initialized_requires_chat_and_search(self):
        assert AdminService().status()["rag"]["initialized"] is False
        assert AdminService(
            chat_service=object(), search_service=object()
        ).status()["rag"]["initialized"] is True

    def test_settings_block_never_leaks_secrets(self):
        """The block is hand-built. A future edit must not add credentials."""
        block = AdminService().status()["settings"]
        forbidden = {
            "jwt_secret_key", "database_url", "postgres_password",
            "groq_api_key", "openai_api_key", "gemini_api_key",
            "qdrant_api_key", "hf_token",
        }
        assert not (forbidden & {k.lower() for k in block})

    def test_router_statuses_take_precedence_over_key_scan(self):
        router = MagicMock()
        router.get_provider_statuses.return_value = {
            "groq": {"initialized": True},
            "openai": {"initialized": False},
        }
        llm = AdminService(provider_router=router).status()["llm"]
        assert llm["available"] == ["groq"]
        assert llm["unavailable"] == ["openai"]
        assert llm["missing_credentials"] == []