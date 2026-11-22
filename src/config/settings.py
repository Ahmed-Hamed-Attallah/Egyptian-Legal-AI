"""
Application settings.

Loads configuration from environment variables with sensible defaults.
Sensitive values (API keys, secrets, DSNs) are never logged or included in repr().

Design notes
------------
* CORS_ORIGINS is kept as a raw `str` and exposed as a typed list via
  `cors_origins_list`. This avoids pydantic-settings' JSON decoding of complex
  types from dotenv values, and prevents Starlette from doing *substring*
  origin matching (which happens when a bare string is passed to allow_origins).
* Legacy / alternative .env key names are accepted through AliasChoices so a
  single canonical field exists per setting (no duplicated fields).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from the project root (src/config/settings.py -> project root).
_project_root = Path(
    os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent.parent.parent)
)
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_project_root / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Application metadata
    # =========================================================================
    APP_NAME: str = Field(default="Egyptian Legal AI")
    APP_ENV: str = Field(
        default="development", pattern="^(development|staging|production)$"
    )
    DEBUG: bool = Field(default=False)
    TIMEZONE: str = Field(default="Africa/Cairo")

    # =========================================================================
    # Core paths
    # =========================================================================
    PROJECT_ROOT: Path = Field(default=_project_root)
    DATA_DIR: Path = Field(default=_project_root / "data")
    RAW_DATA_DIR: Path = Field(default=_project_root / "data" / "raw")
    OFFICIAL_LAWS_DIR: Path = Field(
        default=_project_root / "data" / "raw" / "official_laws"
    )
    USER_UPLOADS_DIR: Path = Field(
        default=_project_root / "data" / "raw" / "user_uploads"
    )
    PROCESSED_DATA_DIR: Path = Field(default=_project_root / "data" / "processed")
    TEMP_DATA_DIR: Path = Field(default=_project_root / "data" / "temp")
    ARTIFACTS_DIR: Path = Field(default=_project_root / "artifacts")
    EMBEDDINGS_DIR: Path = Field(default=_project_root / "artifacts" / "embeddings")
    MODELS_DIR: Path = Field(default=_project_root / "artifacts" / "models")
    INDEXES_DIR: Path = Field(default=_project_root / "artifacts" / "indexes")
    LOG_DIR: Path = Field(
        default=_project_root / "logs",
        validation_alias=AliasChoices("LOG_DIR", "LOGS_DIR"),
    )
    LOG_LEVEL: str = Field(default="INFO")

    # =========================================================================
    # HTTP surface (Phase 13)
    # =========================================================================
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, gt=0, le=65535)
    API_PREFIX: str = Field(default="/api/v1")
    API_BASE_URL: str = Field(default="http://localhost:8000")

    # =========================================================================
    # CORS
    # =========================================================================
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated allowed origins. No trailing slashes.",
    )
    CORS_ALLOW_ORIGIN_REGEX: str = Field(
        default="",
        description=(
            "Optional regex for rotating origins (e.g. dev tunnels). "
            "Development only — must be empty in production."
        ),
    )

    # =========================================================================
    # CSV Column Classification
    # =========================================================================
    CSV_COLUMN_CLASSIFICATION_ENABLED: bool = Field(default=True)
    CSV_COLUMN_CLASSIFICATION_USE_LLM: bool = Field(default=True)
    CSV_COLUMN_CLASSIFICATION_SAMPLE_ROWS: int = Field(default=5, ge=1, le=50)

    # =========================================================================
    # Embeddings (Phase 6)
    # =========================================================================
    EMBEDDING_MODEL_NAME: str = Field(default="intfloat/multilingual-e5-base")
    EMBEDDING_DEVICE: str = Field(default="cpu")
    EMBEDDING_BATCH_SIZE: int = Field(default=32, gt=0)
    EMBEDDING_NORMALIZE: bool = Field(default=True)
    EMBEDDING_PROVIDER: str = Field(default="sentence_transformers")
    EMBEDDING_DIMENSION: int = Field(default=768, gt=0)

    # =========================================================================
    # Vector store (Phase 7)
    # =========================================================================
    QDRANT_URL: str = Field(default="./data/qdrant")
    QDRANT_API_KEY: str = Field(default="", exclude=True)
    QDRANT_COLLECTION_NAME: str = Field(
        default="egyptian_legal",
        validation_alias=AliasChoices("QDRANT_COLLECTION_NAME", "QDRANT_COLLECTION"),
    )
    VECTOR_DIMENSION: int = Field(default=768, gt=0)
    DISTANCE_METRIC: str = Field(default="cosine")

    # =========================================================================
    # Chunking (Phase 5)
    # =========================================================================
    CHUNK_SIZE: int = Field(default=1000, gt=0)
    CHUNK_OVERLAP: int = Field(default=100, ge=0)
    CHUNKING_STRATEGY: str = Field(default="legal")
    MIN_CHUNK_SIZE: int = Field(default=500)
    PRESERVE_LEGAL_ARTICLE_BOUNDARIES: bool = Field(default=True)

    # =========================================================================
    # Retrieval (Phase 8)
    # =========================================================================
    RETRIEVAL_TOP_K: int = Field(default=5, gt=0)
    RRF_K: int = Field(default=60, gt=0)
    BM25_K1: float = Field(default=1.5, gt=0)
    BM25_B: float = Field(default=0.75, ge=0, le=1)
    DENSE_RETRIEVAL_ENABLED: bool = Field(default=True)
    SPARSE_RETRIEVAL_ENABLED: bool = Field(default=True)
    HYBRID_FUSION_METHOD: str = Field(default="rrf")

    # =========================================================================
    # Reranking (Phase 8)
    # =========================================================================
    RERANKING_MODEL_NAME: str = Field(
        default="BAAI/bge-reranker-base",
        validation_alias=AliasChoices("RERANKING_MODEL_NAME", "RERANKER_MODEL_NAME"),
    )
    RERANKING_DEVICE: str = Field(
        default="cpu",
        validation_alias=AliasChoices("RERANKING_DEVICE", "RERANKER_DEVICE"),
    )
    RERANKING_BATCH_SIZE: int = Field(default=8, gt=0)
    RERANKER_ENABLED: bool = Field(default=True)
    RERANKER_TOP_K: int = Field(default=5, gt=0)
    CONTEXT_MAX_TOKENS: int = Field(default=1500, gt=0)

    # =========================================================================
    # API keys & secrets
    # =========================================================================
    GROQ_API_KEY: str = Field(default="", exclude=True)
    OPENAI_API_KEY: str = Field(default="", exclude=True)
    GEMINI_API_KEY: str = Field(default="", exclude=True)
    HF_TOKEN: str = Field(default="", exclude=True)

    # =========================================================================
    # LLM routing & fallback (Phase 9)
    # =========================================================================
    LLM_PRIMARY_PROVIDER: str = Field(default="groq")
    LLM_FALLBACK_ORDER: str = Field(
        default="openai,gemini,transformers",
        validation_alias=AliasChoices("LLM_FALLBACK_ORDER", "LLM_FALLBACK_PROVIDER"),
    )
    LLM_ROUTING_STRATEGY: str = Field(
        default="priority", pattern="^(priority|round_robin|lowest_latency)$"
    )
    LLM_AUTO_FALLBACK_ENABLED: bool = Field(default=True)
    LLM_TEMPERATURE: float = Field(default=0.25, ge=0.0)
    LLM_MAX_TOKENS: int = Field(
        default=500,
        gt=0,
        validation_alias=AliasChoices("LLM_MAX_TOKENS", "LLM_MAX_OUTPUT_TOKENS"),
    )
    LLM_REQUEST_TIMEOUT_SECONDS: int = Field(
        default=60,
        gt=0,
        validation_alias=AliasChoices("LLM_REQUEST_TIMEOUT_SECONDS", "LLM_TIMEOUT"),
    )

    # Retry & backoff
    LLM_MAX_RETRIES: int = Field(default=2, ge=0)
    LLM_RETRY_BACKOFF_SECONDS: float = Field(default=1.0, ge=0)
    LLM_RETRY_MAX_BACKOFF_SECONDS: float = Field(default=8.0, ge=0)

    # Cooldowns
    LLM_PROVIDER_COOLDOWN_SECONDS: int = Field(default=60, ge=0)
    LLM_RATE_LIMIT_COOLDOWN_SECONDS: int = Field(default=60, ge=0)
    LLM_QUOTA_COOLDOWN_SECONDS: int = Field(default=3600, ge=0)

    # Circuit breaker
    LLM_CIRCUIT_BREAKER_THRESHOLD: int = Field(default=3, gt=0)
    LLM_CIRCUIT_BREAKER_RECOVERY_SECONDS: int = Field(default=120, gt=0)

    # Provider-specific models
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash")

    # Local Transformers fallback
    LOCAL_MODEL_NAME: str = Field(default="")
    LOCAL_MODEL_DEVICE: str = Field(default="auto")
    LOCAL_MODEL_DTYPE: str = Field(default="auto")
    LOCAL_MODEL_LOAD_IN_4BIT: bool = Field(default=False)
    LOCAL_MODEL_MAX_NEW_TOKENS: int = Field(default=512, gt=0)
    LOCAL_MODEL_REPETITION_PENALTY: float = Field(default=1.2, ge=1.0)
    LOCAL_MODEL_NO_REPEAT_NGRAM_SIZE: int = Field(default=3, ge=0)
    LOCAL_MODEL_DO_SAMPLE: bool = Field(default=True)
    LOCAL_MODEL_TEMPERATURE: float = Field(default=0.2, ge=0.0)
    LOCAL_MODEL_TOP_P: float = Field(default=0.9, gt=0.0, le=1.0)

    # =========================================================================
    # Database (Phase 11)
    # =========================================================================
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="egyptian_legal")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="", exclude=True)
    DATABASE_URL: str = Field(
        default="sqlite:///./data/egyptian_legal.db", exclude=True
    )
    DATABASE_POOL_SIZE: int = Field(default=5, gt=0)
    DATABASE_MAX_OVERFLOW: int = Field(default=10, ge=0)
    DATABASE_ECHO: bool = Field(default=False)

    # =========================================================================
    # Document storage (Phase 11)
    # =========================================================================
    DOCUMENT_STORAGE_DIR: Path = Field(default=_project_root / "data" / "documents")
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, gt=0)
    MAX_UPLOAD_SIZE_BYTES: int = Field(default=50 * 1024 * 1024, gt=0)
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=["pdf", "docx", "txt", "md", "csv", "tsv"]
    )

    # =========================================================================
    # Cache (Phase 11)
    # =========================================================================
    REDIS_URL: str = Field(default="")
    CACHE_DEFAULT_TTL: int = Field(default=3600, gt=0)
    CACHE_ENABLED: bool = Field(default=False)

    # =========================================================================
    # Security (Phase 12)
    # =========================================================================
    JWT_SECRET_KEY: str = Field(
        default="",
        exclude=True,
        validation_alias=AliasChoices("JWT_SECRET_KEY", "SECRET_KEY"),
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        validation_alias=AliasChoices("JWT_ALGORITHM", "ALGORITHM"),
    )
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60 * 24,
        gt=0,
        validation_alias=AliasChoices(
            "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "ACCESS_TOKEN_EXPIRE_MINUTES"
        ),
    )
    DEFAULT_TENANT_ID: str = Field(default="default")
    AUTH_ENABLED: bool = Field(default=False)

    # =========================================================================
    # Conversation memory
    # =========================================================================
    CHAT_MEMORY_ENABLED: bool = Field(default=True)
    CHAT_MEMORY_MAX_MESSAGES: int = Field(default=20, gt=0)
    CHAT_MEMORY_MAX_CONVERSATIONS: int = Field(default=10, gt=0)
    CHAT_MEMORY_QUESTIONS_PER_CONVERSATION: int = Field(default=10, gt=0)

    # =========================================================================
    # Legal answer behaviour (ARCHITECTURE §7/§9, LLM_STRATEGY §7)
    # =========================================================================
    REQUIRE_CITATIONS: bool = Field(default=True)
    ALLOW_ANSWERS_WITHOUT_SOURCES: bool = Field(default=False)
    SHOW_SOURCE_METADATA: bool = Field(default=True)
    DISTINGUISH_USER_UPLOADS_FROM_OFFICIAL_LAWS: bool = Field(default=True)
    LEGAL_DISCLAIMER_ENABLED: bool = Field(default=True)

    # =========================================================================
    # OCR (Phase 3)
    # =========================================================================
    OCR_ENABLED: bool = Field(default=True)
    OCR_LANGUAGE: str = Field(default="ara+eng")
    OCR_ENGINE: str = Field(default="paddleocr")

    # =========================================================================
    # Background workers (Phase 15)
    # =========================================================================
    WORKER_ENABLED: bool = Field(default=True)
    WORKER_EMBEDDED: bool = Field(default=False)
    WORKER_CONCURRENCY: int = Field(default=2, gt=0)
    WORKER_POLL_INTERVAL_SECONDS: float = Field(default=2.0, gt=0)
    WORKER_JOB_TIMEOUT_SECONDS: int = Field(default=900, gt=0)
    WORKER_MAX_RETRIES: int = Field(default=2, ge=0)
    WORKER_RETRY_BACKOFF_SECONDS: float = Field(default=30.0, ge=0)
    WORKER_RECLAIM_STALE_AFTER_SECONDS: int = Field(default=600, gt=0)
    WORKER_JOB_STORE: str = Field(default="sql", pattern="^(sql|memory)$")

    # =========================================================================
    # Monitoring & evaluation (Phases 16–17)
    # =========================================================================
    METRICS_ENABLED: bool = Field(default=False)
    TRACING_ENABLED: bool = Field(default=False)
    MONITORING_EXPORT_PATH: str = Field(default="")
    EVALUATION_DATA_DIR: Path = Field(default=_project_root / "data" / "evaluation")

    # =========================================================================
    # Field validators
    # =========================================================================
    @field_validator("ALLOWED_FILE_TYPES", mode="before")
    @classmethod
    def _split_file_types(cls, value: Any) -> Any:
        """Accept a comma-separated string as well as a JSON/list value."""
        if isinstance(value, str):
            return [v.strip().lower().lstrip(".") for v in value.split(",") if v.strip()]
        return value

    @field_validator("API_PREFIX")
    @classmethod
    def _normalise_prefix(cls, value: str) -> str:
        """Ensure a leading slash and no trailing slash ('' disables prefixing)."""
        value = value.strip()
        if not value or value == "/":
            return ""
        return "/" + value.strip("/")

    # =========================================================================
    # Derived values (no duplicated state)
    # =========================================================================
    @property
    def cors_origins_list(self) -> list[str]:
        """Explicit origin list — never a substring match."""
        return [o.strip().rstrip("/") for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def QDRANT_COLLECTION(self) -> str:  # noqa: N802 - backwards-compatible alias
        """Legacy attribute name kept so existing callers do not break."""
        return self.QDRANT_COLLECTION_NAME

    @property
    def llm_fallback_providers(self) -> list[str]:
        return [p.strip() for p in self.LLM_FALLBACK_ORDER.split(",") if p.strip()]

    # =========================================================================
    # Cross-field validation
    # =========================================================================
    def model_post_init(self, context: Any) -> None:
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be strictly less than CHUNK_SIZE")

        # Derive upload limit from MB unless bytes were set explicitly.
        if "MAX_UPLOAD_SIZE_BYTES" not in self.model_fields_set:
            self.MAX_UPLOAD_SIZE_BYTES = self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

        # Keep the vector schema aligned with the embedding model.
        if "VECTOR_DIMENSION" not in self.model_fields_set:
            self.VECTOR_DIMENSION = self.EMBEDDING_DIMENSION
        elif self.VECTOR_DIMENSION != self.EMBEDDING_DIMENSION:
            raise ValueError(
                "VECTOR_DIMENSION "
                f"({self.VECTOR_DIMENSION}) != EMBEDDING_DIMENSION "
                f"({self.EMBEDDING_DIMENSION}). Qdrant writes would fail."
            )

        # Fail fast instead of signing JWTs with an empty/weak secret.
        if self.AUTH_ENABLED:
            if not self.JWT_SECRET_KEY:
                raise ValueError(
                    "AUTH_ENABLED=true but JWT_SECRET_KEY (or SECRET_KEY) is empty. "
                    "Generate one: python -c \"import secrets; "
                    'print(secrets.token_urlsafe(64))"'
                )
            if len(self.JWT_SECRET_KEY) < 32:
                raise ValueError(
                    "JWT_SECRET_KEY must be at least 32 characters when AUTH_ENABLED=true."
                )

        if self.APP_ENV == "production":
            if self.CORS_ALLOW_ORIGIN_REGEX:
                raise ValueError(
                    "CORS_ALLOW_ORIGIN_REGEX must be empty in production."
                )
            if "*" in self.cors_origins_list:
                raise ValueError(
                    "Wildcard CORS origin is not allowed in production."
                )
            if not self.POSTGRES_PASSWORD and self.DATABASE_URL.startswith("postgresql"):
                raise ValueError("POSTGRES_PASSWORD must be set in production.")

    # =========================================================================
    # Secret-safe representation
    # =========================================================================
    def __repr__(self) -> str:
        secret_fields = {
            name
            for name, field in type(self).model_fields.items()
            if field.exclude
        }
        safe = {
            key: ("***" if key in secret_fields else value)
            for key, value in self.__dict__.items()
        }
        return f"Settings({safe})"


settings = Settings()