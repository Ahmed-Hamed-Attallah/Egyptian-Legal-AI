# Roadmap

Status is judged by **observable behaviour**, not by whether files exist. A module
that is present but never constructed by `factory.py` is marked incomplete.

## Completed

| Phase | Deliverable | Evidence |
|---|---|---|
| 1–2 | Corpus ingestion, Arabic normalization, patterns | `preprocessing/`, verified digit + hamza folding in `BM25 QUERY` logs |
| 3 | Parsers (pdf/docx/md/csv/tsv/html/xlsx) + OCR adapters | `ingestion/parsers/`, uploads succeed |
| 4 | Legal-aware chunking + quality gate | `chunking/`, 32 chunks from the corpus with article boundaries preserved |
| 5 | Embeddings (e5-base, 768, cosine) + model registry | `embeddings/`, `EMBEDDING_DIMENSION=768` matches the collection |
| 6 | Qdrant store, collection manager | `vector_store/`, payload-indexed filtering |
| 7 | Dense + sparse + RRF hybrid | `retrieval/`, `RRF FUSION` provenance now logged |
| 8 | Cross-encoder reranking | `reranking/`, activation trap diagnosed and fixed |
| 9 | LLM providers, routing, fallback, circuit breaker | `llm/`, Groq succeeds; local fallback now opt-in |
| 10 | Generation, context budget, citations, validation | `generation/`, `refusal ⇒ no citations` implemented |
| 11 | Postgres models, repositories, Alembic, document store | `storage/`, migration 0002 + owner backfill |
| 12 | Auth: JWT, bcrypt, roles, tenants, file validation | `security/`, login 200 in ~230 ms |
| 13 | FastAPI surface: 7 routers, prefix, error envelope, request id | `api/`, `Mounted 7 routers under prefix '/api/v1'` |
| 14 | Next.js UI: chat, library, document detail, sidebar, health badge | `web/`, RTL Arabic, two-panel library |
| 15 | Worker: job store, manager, reclaim, standalone process | `workers/`, `processing_jobs` + entry point |
| 16 | Monitoring: health probes, stage timers, metrics collector | `monitoring/`, `/health/live` + `/health/ready` |

## In progress — must close before release

| # | Item | State |
|---|---|---|
| A | **Job scoping** | Not started. `api/routes/jobs.py`, `job_store.py`, `job_manager.py` list every tenant job |
| B | **Retrieval scoping in chat** | Filter builder exists; `ChatService.ask` does not pass `filters` to the orchestrator |
| C | **Query routing** | `QueryClassifier` wired into `RAGPipeline`; `AnswerGenerator(conversational=True)` unverified |
| D | **BM25 persistence** | Incremental refresh works; `artifacts/indexes/` caching not implemented |
| E | **Deployment** | Files authored; not yet built or run |

## Not started

| Item | Why it matters |
|---|---|
| **Golden evaluation set** | `src/evaluation/` is complete and never runs. Without expected articles per question, every retrieval change is unverifiable. Highest-leverage gap in the project. |
| **Test coverage** | `tests/api/`, `tests/integration/`, `tests/evaluation/` contain only `__init__.py`. 3 unit test files exist. |
| **CI** | No `.github/workflows`. lint + typecheck + tests + migration-on-fresh-DB should gate merge. |
| **Streaming answers** | `BaseLLMProvider.generate_stream` is declared in `LLM_STRATEGY.md` but not exposed. Requires converting `RequestContextMiddleware` off `BaseHTTPMiddleware` first. |
| **Token revocation** | 60-minute tokens, logout is client-side only. |
| **Amendment/version resolution** | `domain/legal_versions.py` and `preprocessing/amendment_parser.py` exist; retrieval does not prefer the in-force version. |
| **Official-source weighting** | `DISTINGUISH_USER_UPLOADS_FROM_OFFICIAL_LAWS` is read into `RetrievedChunk.is_official`, but the prompt does not bias toward official text. |
| **Redis caching** | Answer + embedding caching. Biggest remaining latency win. |
| **Corpus expansion** | `civil_law.md`, `civil_status.md`, `personal_status.md`, `companies_law.md` exist in `data/raw/official_laws/` but are not ingested. Two more `.md` files are misplaced in the project root. |

## Suggested order

1. **A + B** — security. Non-negotiable before a second user exists.
2. **Corpus hygiene** — ingest the official laws properly, delete orphans, reconcile.
3. **Golden evaluation set** — 30–50 questions with expected articles. Everything
   after this is measurable instead of guessed.
4. **CI** — lint, mypy, pytest, fresh-DB migration check.
5. **C + D** — routing polish and index persistence.
6. **E** — build the compose stack, run the verification suite, cut the tunnels.
7. **Streaming** — perceived latency, after the middleware conversion.
8. **Redis caching** — repeat-question latency, after evaluation proves quality.

## Definition of done for release

- [ ] Cross-user isolation test passes for documents, chunks, **jobs** and **retrieval**
- [ ] `ما حكم المادة ١٤٧ من القانون المدني المصري؟` returns article 147 with both paragraphs cited
- [ ] `ما آخر سؤال سألته لك؟` answers from history with **zero** citations
- [ ] Refusal responses carry **zero** citations
- [ ] `RERANK RAW` shows a real logit spread on a clean corpus
- [ ] Corpus reconciliation reports zero orphans and zero unindexed documents
- [ ] `docker compose --profile worker up` reaches all-healthy from an empty volume
- [ ] Evaluation set runs and reports retrieval + citation metrics
