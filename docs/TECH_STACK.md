# Tech Stack

Selection policy: a dependency must solve a real requirement, and no two
libraries may cover the same responsibility. Hardware constraints are treated as
requirements, not preferences.

## Backend

| Concern | Choice | Version | Notes |
|---|---|---|---|
| HTTP framework | FastAPI | `>=0.115` | Typed contracts, `/openapi.json` used to generate frontend types |
| ASGI server | Uvicorn | `>=0.32` | `--proxy-headers` behind nginx |
| Validation | Pydantic | `>=2.9` | Domain models *and* API schemas |
| Settings | pydantic-settings | `>=2.5` | `extra="ignore"` — **misspelled keys are silently dropped**; see `.env.example` |
| ORM | SQLAlchemy | `>=2.0.35` | 2.x typed `Mapped[]` style |
| Migrations | Alembic | `>=1.13` | Config at `src/storage/database/migrations/alembic.ini` |
| DB driver | psycopg | `>=3.2` | `postgresql+psycopg://` — v3, binary wheel, no compiler needed |
| Auth | PyJWT + passlib/bcrypt | `>=2.8`, `>=1.7.4` | **PyJWT, not `jwt`** (python-jwt has no `encode`). bcrypt `<5.0` — 5.x breaks passlib backend detection |
| HTTP client | httpx | `>=0.27` | Providers + integration tests |

## Retrieval & ML

| Concern | Choice | Detail |
|---|---|---|
| Embeddings | `intfloat/multilingual-e5-base` | 768 dims, cosine, query/prefix asymmetry handled in `embeddings/` |
| Sparse | `rank-bm25` | In-memory, Arabic-normalized, incremental refresh |
| Reranker | `BAAI/bge-reranker-base` | 278M, cross-encoder |
| Vector DB | Qdrant 1.12 | REST + gRPC, payload-indexed filtering |
| Tokenizers | `sentencepiece` (via HF) | `TOKENIZERS_PARALLELISM=false` — fork-unsafe otherwise |
| ML runtime | PyTorch ≥2.4 **CPU wheels** | Installed from `download.pytorch.org/whl/cpu` before `requirements.txt` |

## Document processing

| Format | Library |
|---|---|
| PDF | `pymupdf` (text) + `paddleocr` (scanned, optional) |
| DOCX | `python-docx` |
| HTML | `beautifulsoup4` |
| CSV/TSV/XLSX | `pandas` |
| Upload parsing | `python-multipart` |
| Email validation | `email-validator` (required by pydantic `EmailStr`) |

## Frontend

| Concern | Choice |
|---|---|
| Framework | Next.js 14 App Router, `reactStrictMode: true` |
| Language | TypeScript strict |
| Styling | Tailwind CSS + `clsx` |
| Icons | `lucide-react` |
| State | React hooks + `useSyncExternalStore` for the shared health poller |
| Types | `openapi-typescript` from the API's `/openapi.json` (recommended; currently hand-written) |

## Infrastructure

| Concern | Choice |
|---|---|
| Containers | Docker + Compose (profiles: `worker`, `cache`) |
| Edge | nginx — single origin, no browser CORS |
| Database | PostgreSQL 16 alpine |
| Vector store | `qdrant/qdrant:v1.12.4` |
| Cache | Redis 7 alpine — **optional profile, unused today** |
| Quality | `ruff`, `mypy`, `pytest`, `pre-commit`, `coverage` |

## Explicitly excluded

| Rejected | Because |
|---|---|
| Streamlit | Superseded by `web/`. Older docs referenced it; those docs were rewritten. |
| LangChain | Retrieval, chunking and prompt assembly are ~200 lines of explicit code. A framework would hide the failure modes this project had to debug. |
| Celery | `processing_jobs` in Postgres + a poll loop is sufficient at this scale and keeps one less broker. |
| bitsandbytes 4-bit | Requires Turing+ (sm_75). Target GPU is Pascal sm_61. |
| Async SQLAlchemy | Would require rewriting every repository in Phases 1–10 for no measured gain at current concurrency. |

## Hardware constraints (binding)

| Resource | Value | Consequence |
|---|---|---|
| RAM | 32 GB | Three CPU models + API + worker fit, but not four large models |
| GPU VRAM | 4 GB | Qwen2.5-3B in bf16 (~6 GB) does not fit |
| GPU arch | Pascal sm_61 | 4-bit quantisation unavailable |
| CPU cores | 12 | Torch defaults to all 12 per model → contention. Capped at 2 via `OMP_NUM_THREADS` |

**Do not "optimise" by enabling GPU inference** without first confirming
`torch.cuda.is_available()` and that the model fits. The measured CPU path is the
supported one.
