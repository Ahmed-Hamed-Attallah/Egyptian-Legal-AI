"""Shared pytest fixtures.

Isolation contract
------------------
Environment variables are forced BEFORE `src.config.settings` is imported,
because that module instantiates `settings = Settings()` at import time. Any
fixture that patched it afterwards would be too late — every module holds a
reference bound at import.

Consequences:
  * tests never read your real .env values for DB/storage/keys
  * no LLM provider can initialise (all API keys empty)
  * no model weights are downloaded (LOCAL_MODEL_NAME empty)
  * a session guard aborts if DATABASE_URL points at a real server
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import pytest

# --------------------------------------------------------------------------
# 1. Path + environment — MUST run before any `src.` import
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="mizan-tests-"))

os.environ.update(
    {
        "PROJECT_ROOT": str(ROOT),
        "APP_ENV": "development",
        "DEBUG": "false",
        "LOG_LEVEL": "WARNING",
        "LOG_DIR": str(_TMP / "logs"),
        # Storage — never touch data/documents
        "DATA_DIR": str(_TMP / "data"),
        "DOCUMENT_STORAGE_DIR": str(_TMP / "documents"),
        "ARTIFACTS_DIR": str(_TMP / "artifacts"),
        "INDEXES_DIR": str(_TMP / "indexes"),
        # Database — file-backed SQLite, isolated per session
        "DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}",
        "POSTGRES_PASSWORD": "",
        "DATABASE_ECHO": "false",
        # Auth — off by default; a fixed key lets auth tests enable it
        "AUTH_ENABLED": "false",
        "JWT_SECRET_KEY": "test-secret-key-not-for-production-use-0123456789abcdef",
        "JWT_ALGORITHM": "HS256",
        "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": "60",
        "DEFAULT_TENANT_ID": "default",
        # No provider can initialise, no weights are fetched
        "GROQ_API_KEY": "",
        "OPENAI_API_KEY": "",
        "GEMINI_API_KEY": "",
        "HF_TOKEN": "",
        "LLM_PRIMARY_PROVIDER": "groq",
        "LLM_FALLBACK_ORDER": "",
        "LOCAL_MODEL_NAME": "",
        # Vector store — local path mode, no server required
        "QDRANT_URL": str(_TMP / "qdrant"),
        "QDRANT_API_KEY": "",
        "QDRANT_COLLECTION_NAME": "test_collection",
        "EMBEDDING_DIMENSION": "768",
        "VECTOR_DIMENSION": "768",
        # Background work / side effects off
        "WORKER_ENABLED": "false",
        "WORKER_EMBEDDED": "false",
        "WORKER_JOB_STORE": "memory",
        "OCR_ENABLED": "false",
        "CACHE_ENABLED": "false",
        "REDIS_URL": "",
        "METRICS_ENABLED": "false",
        "TRACING_ENABLED": "false",
        "MONITORING_EXPORT_PATH": "",
        # Deterministic retrieval
        "RERANKER_ENABLED": "false",
        "RETRIEVAL_TOP_K": "5",
        "RERANKER_TOP_K": "3",
    }
)

from src.config.settings import settings  # noqa: E402  (must follow env setup)


# --------------------------------------------------------------------------
# 2. Safety guards
# --------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _no_production_database() -> None:
    """Abort the whole session if a real server slipped through."""
    url = settings.DATABASE_URL
    if url.startswith(("postgresql", "mysql", "mysql+pymysql")):
        pytest.exit(
            f"Refusing to run tests against a real database: {url!r}. "
            "conftest.py must set DATABASE_URL to SQLite before settings import.",
            returncode=3,
        )


@pytest.fixture(scope="session", autouse=True)
def _no_live_llm_credentials() -> None:
    """A leaked key would let a test bill your Groq account."""
    leaked = [
        name
        for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
        if getattr(settings, name, "")
    ]
    if leaked:
        pytest.exit(f"LLM credentials present during tests: {leaked}", returncode=3)


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-test upload/log directories so tests cannot see each other's files."""
    docs = tmp_path / "documents"
    logs = tmp_path / "logs"
    docs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "DOCUMENT_STORAGE_DIR", docs, raising=False)
    monkeypatch.setattr(settings, "LOG_DIR", logs, raising=False)


# --------------------------------------------------------------------------
# 3. Reusable fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def app_settings() -> Any:
    """The real settings singleton. Use monkeypatch to change a field."""
    return settings


@pytest.fixture
def auth_enabled(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Turn auth on for a single test."""
    monkeypatch.setattr(settings, "AUTH_ENABLED", True, raising=False)
    return settings


@pytest.fixture
def make_auth() -> Any:
    """Factory for verified AuthContext objects."""
    from src.security.authentication import AuthContext

    def _make(
        user_id: str | None = "user-a",
        tenant_id: str = "default",
        roles: tuple[str, ...] = ("user",),
        is_authenticated: bool = True,
    ) -> AuthContext:
        return AuthContext(
            user_id=user_id,
            tenant_id=tenant_id,
            roles=roles,
            is_authenticated=is_authenticated,
        )

    return _make


@pytest.fixture
def sqlite_database() -> Iterator[Any]:
    """A fresh in-memory Database with all ORM tables created.

    Uses the ORM Base, so `processing_jobs` (created by a migration, not the
    Base) is intentionally absent — job tests must use the memory store.
    """
    from src.storage.database.database import Database
    from src.storage.database.models import Base
    import src.storage.database.models.document  # noqa: F401
    import src.storage.database.models.chunk  # noqa: F401
    import src.storage.database.models.conversation  # noqa: F401
    import src.storage.database.models.user  # noqa: F401

    db = Database(database_url="sqlite://", echo=False)
    Base.metadata.create_all(bind=db.engine)
    try:
        yield db
    finally:
        db.dispose()


@pytest.fixture
def make_chunk() -> Any:
    """Factory for domain Chunks with legal metadata."""
    import uuid
    from src.domain.chunks import Chunk, ChunkType

    def _make(
        content: str = "المادة 147 — العقد شريعة المتعاقدين.",
        index: int = 0,
        document_id: Any = None,
        metadata: dict | None = None,
    ) -> Chunk:
        return Chunk(
            document_id=document_id or uuid.uuid4(),
            chunk_index=index,
            content=content,
            chunk_type=ChunkType.ARTICLE,
            metadata=metadata or {},
        )

    return _make