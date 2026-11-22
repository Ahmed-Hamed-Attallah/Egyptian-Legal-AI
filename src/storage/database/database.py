"""
Database connection and session management.

Uses SQLAlchemy 2.x with synchronous sessions, consistent with the rest of the
codebase (Phases 1-10). Structured so an async adapter can be added later
without rewriting repositories.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.config.settings import settings
from src.domain.exceptions import DatabaseError
from src.utils.json_utils import json_dumps

logger = logging.getLogger(__name__)

# Fail fast instead of waiting SQLAlchemy's 30s default, which outlives most
# client timeouts and surfaces as a mysterious "connection timed out".
POOL_TIMEOUT_SECONDS = 10
# Recycle connections idle longer than this (Postgres/proxy idle reaping).
POOL_RECYCLE_SECONDS = 1800


class Database:
    """Centralized database connection manager.

    Owns the SQLAlchemy Engine and SessionFactory and provides session
    lifecycle management through context managers.
    """

    def __init__(self, database_url: Optional[str] = None, echo: Optional[bool] = None):
        self._url = database_url or settings.DATABASE_URL
        self._echo = echo if echo is not None else settings.DATABASE_ECHO
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._initialize()

    def _initialize(self) -> None:
        """Create engine and session factory."""
        try:
            connect_args = {}
            pool_kwargs = {}

            if self._url.startswith("sqlite"):
                connect_args["check_same_thread"] = False
                pool_kwargs["poolclass"] = StaticPool
            else:
                pool_kwargs.update(
                    pool_size=settings.DATABASE_POOL_SIZE,
                    max_overflow=settings.DATABASE_MAX_OVERFLOW,
                    pool_pre_ping=True,
                    pool_timeout=POOL_TIMEOUT_SECONDS,
                    pool_recycle=POOL_RECYCLE_SECONDS,
                )

            self._engine = create_engine(
                self._url,
                echo=self._echo,
                # UUID/date/Decimal-safe JSON + JSONB for every model.
                json_serializer=json_dumps,
                connect_args=connect_args,
                **pool_kwargs,
            )

            @event.listens_for(self._engine, "connect")
            def _on_connect(dbapi_conn, connection_record):  # noqa: ARG001
                logger.debug("Database connection established")

            self._session_factory = sessionmaker(
                bind=self._engine,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )
            logger.info("Database initialized: %s", self._safe_url())

        except Exception as exc:
            logger.exception("Failed to initialize database")
            raise DatabaseError(f"Database initialization failed: {exc}") from exc

    def _safe_url(self) -> str:
        """Return URL with credentials masked."""
        if "@" not in self._url:
            return self._url
        prefix, suffix = self._url.rsplit("@", 1)
        scheme_end = prefix.find("://")
        if scheme_end == -1:
            return self._url
        return f"{prefix[:scheme_end + 3]}***@{suffix}"

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            raise DatabaseError("Database not initialized")
        return self._engine

    def create_session(self) -> Session:
        """Create a new session. Caller is responsible for closing it."""
        if self._session_factory is None:
            raise DatabaseError("Database not initialized")
        return self._session_factory()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional scope: commit on clean exit, rollback on error, always close."""
        session = self.create_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def read_only_session(self) -> Iterator[Session]:
        """Read-only session — never commits, rolls back on error, always closes."""
        session = self.create_session()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_all(self) -> None:
        """Create all tables. Development/testing only."""
        from src.storage.database.models import Base  # local import avoids cycle

        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def drop_all(self) -> None:
        """Drop all tables. Development/testing only."""
        from src.storage.database.models import Base

        Base.metadata.drop_all(bind=self.engine)
        logger.warning("Database tables dropped")

    def health_check(self) -> bool:
        """Verify the database connection is alive."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.warning("Database health check failed: %s", exc)
            return False

    def dispose(self) -> None:
        """Dispose of the engine and release connections."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database disposed")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_database: Optional[Database] = None


def get_database() -> Database:
    """Return the application-wide Database instance."""
    global _database
    if _database is None:
        _database = Database()
    return _database


def reset_database() -> None:
    """Reset the singleton. Tests only."""
    global _database
    if _database is not None:
        _database.dispose()
        _database = None


@contextmanager
def get_session() -> Iterator[Session]:
    """Convenience context manager for a transactional session."""
    db = get_database()
    with db.session() as session:
        yield session