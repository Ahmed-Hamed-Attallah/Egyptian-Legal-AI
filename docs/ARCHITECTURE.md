# Architecture — Egyptian Legal AI (Mizan)

> **Scope of this document.** It describes the system **as it exists now**, not a
> plan. Where the design intent differs from the implementation, that gap is
> recorded in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) rather than papered over here.

## 1. Purpose

A retrieval-augmented Q&A system for **Egyptian law in Arabic**, with:

- **Verifiable citations** — every substantive answer names the law, article and
  section it came from.
- **Source authority separation** — curated official text is distinguishable from
  user uploads. A user's PDF must never be presented as authoritative law.
- **Tenant and owner isolation** — one user's private uploads are invisible to
  every other user, in the library **and** in retrieval.

This is a legal research aid, **not** legal advice. Every answer carries a
disclaimer, and the bundled corpus files are explicitly illustrative excerpts.

## 2. Layer map

```
web/        Next.js 14 App Router — UI, auth session, server-side API proxy
  │           browser → nginx → /api/v1/* (same origin, no CORS preflight)
  ▼
api/        FastAPI — HTTP surface, auth dependencies, error envelope, request id
  │           thin: no business logic, no ML, no SQL
  ▼
src/        Domain + services + retrieval/generation pipelines + storage adapters
  │
  ├── services/     orchestration (document, chat, search, conversation, auth, admin)
  ├── rag/          classify → retrieve → rerank → generate → validate
  ├── retrieval/    dense (Qdrant) + sparse (BM25) → RRF fusion
  ├── reranking/    cross-encoder
  ├── generation/   prompt assembly, citation formatting, response validation
  ├── ingestion/    parsers (pdf/docx/md/csv/txt/html) + OCR
  ├── preprocessing/ Arabic normalization, legal structure parsing, metadata extraction
  ├── chunking/     legal-aware chunking + quality gate
  ├── embeddings/   sentence-transformers adapter + model registry
  ├── vector_store/ Qdrant adapter + collection manager
  ├── storage/      SQLAlchemy models, repositories, migrations, document store, cache
  ├── security/     authentication, authorization, tenant isolation, file validation
  ├── llm/          provider abstraction, routing, fallback, circuit breaker
  ├── monitoring/   health, metrics, tracing, pipeline stage timers
  └── config/       settings (pydantic-settings), logging, retrieval/model constants
  ▼
workers/    Standalone ingestion process (root = CLI entry, src/workers = logic)

PostgreSQL  documents, chunks, users, conversations, messages, processing_jobs
Qdrant      768-dim vectors + legal payload (the access-control source of truth)
Redis       OPTIONAL — currently unused (see §9)
```

**Responsibility boundaries that must not be crossed:**

| Rule | Why |
|---|---|
| `api/` contains no ML and no SQL | Routes validate and delegate. Anything heavier makes the HTTP layer untestable. |
| `src/` never imports FastAPI or `api/` | The domain must be usable from the worker, scripts, notebooks and tests. |
| `web/` never calls Qdrant/Postgres directly | One path to data, one place to enforce visibility. |
| Business logic lives in `services/`, persistence in `repositories/`, transport in `api/` | Prevents the duplicated-logic class of bug this project already fought. |

## 3. Request flow — chat

```
POST /api/v1/chat
  → RequestContextMiddleware          assigns request_id, measures duration
  → get_auth_context                  JWT → AuthContext (user_id, tenant_id, roles)
  → ChatService.ask
      ├─ ConversationService          load history + memory digest
      ├─ RAGOrchestrator.ask
      │    └─ RAGPipeline.process
      │        0. QueryClassifier      LEGAL_RESEARCH | MULTI_CONCEPT | CONVERSATIONAL
      │        1. RetrievalPipeline    ensure_fresh() → hybrid retrieve (filters applied)
      │             ├─ DenseRetriever  embed query → Qdrant search + payload post-filter
      │             └─ SparseRetriever BM25 over normalized Arabic
      │             └─ ReciprocalRankFusion  k=RRF_K, preserves dense/sparse provenance
      │        2. RerankingPipeline    cross-encoder → probability, MIN_RERANK_SCORE
      │        3. ContextBuilder       token-budgeted assembly
      │        4. AnswerGenerator      ProviderRouter → Groq (fallback chain)
      │        5. CitationResolver     chunk → law/article/section citation
      │        6. ResponseValidator    grounding + citation policy
      └─ persist message + citations to Postgres
```

**Measured stage cost (CPU, 12 cores, clean corpus):**

| Stage | Cost |
|---|---|
| Dense retrieval | 0.15–0.40 s |
| Sparse retrieval | 0.04–0.14 s |
| Rerank 20 candidates | 7.4–8.8 s ← dominant |
| Groq generation | 0.7–3.6 s |
| **Total** | **~10–12 s** |

Reranking is 65–88% of latency. That is why it must be either working or off — see [`RETRIEVAL_DESIGN.md`](RETRIEVAL_DESIGN.md) §5.

## 4. The visibility model — one policy, three enforcers

This is the architectural core of the isolation work.

```
DocumentRecord.scope  ∈ { OFFICIAL, PRIVATE }      (SQL, uppercase enum name)
DocumentRecord.owner_id → users.id | NULL
Qdrant payload       scope: "official" | "private"   (lowercase enum VALUE)
                     owner_id: "<uuid>"
```

The policy is expressed **once** as a `MetadataFilter` tree:

```
tenant_id == T
AND ( scope == "official"
      OR ( scope == "private" AND owner_id == me ) )
```

and then evaluated by **three independent enforcers** that must agree:

| Enforcer | Mechanism | File |
|---|---|---|
| Dense retrieval | push-down to Qdrant `Filter` **plus mandatory Python post-filter** | `retrieval/metadata_filters.py` → `vector_store/qdrant_store.py` |
| Sparse retrieval | `MetadataFilter.matches(chunk.metadata)` per candidate | `retrieval/sparse_retriever.py` |
| SQL library | `_visibility_clause()` predicate | `storage/database/repositories/document_repository.py` |

**Rules:**

1. **Fail closed.** `scope` missing or unrecognised ⇒ treated as `private`.
   An unowned private row is admin-only, never public.
2. **Push-down is an optimisation, not a guarantee.** Any operator Qdrant cannot
   express (`contains` substring) is omitted from the pushed filter and settled by
   the post-filter. Over-fetching (`_OVERSAMPLE = 4`) keeps `top_k` full afterwards.
3. **A translation error is fatal, not ignorable.** `_build_qdrant_filter` raises
   rather than searching unfiltered. The previous behaviour — log a warning and
   search everything — was an access-control bypass.
4. **Payload and SQL must stay in sync.** `PATCH /documents/{id}/scope` calls
   `set_document_payload`, which rewrites `scope`/`owner_id` on every point of that
   document. A SQL-only change would silently make a document appear or disappear
   from retrieval.

**Why the payload is the source of truth for retrieval:** Qdrant cannot join to
Postgres. Filtering by "is this document still visible" would require a DB round
trip per candidate. Denormalising `scope`/`owner_id` into the point payload makes
visibility a single index lookup.

## 5. Index freshness

BM25 is an **in-memory** index built at startup by scrolling Qdrant
(`factory.py` → `scroll_all` → `build_index`). A 45k-chunk corpus costs ~28 s.

The staleness problem: a document uploaded *after* startup was invisible to sparse
retrieval until the process restarted.

The fix is layered:

| Mechanism | Where | Effect |
|---|---|---|
| `SparseRetriever.add_chunks()` / `remove_document()` | `retrieval/sparse_retriever.py` | Incremental mutation; keeps the tokenised corpus so a rebuild costs only BM25 statistics |
| `RetrievalPipeline.ensure_fresh()` | `retrieval/retrieval_pipeline.py` | Compares Qdrant point count vs indexed count before retrieval; refreshes if divergent |
| `RAGPipeline` injects `RetrievalPipeline` | `rag/pipeline.py` | Makes `ensure_fresh()` run on every query instead of calling `hybrid_retriever` directly |

**Consequence:** `RAGPipeline` must receive `retrieval_pipeline`. Passing
`hybrid_retriever` alone still works but silently loses auto-refresh — the class
logs a warning precisely so this cannot happen unnoticed.

## 6. Refusal and citation policy

A refusal must never look source-grounded.

| Condition | Behaviour |
|---|---|
| Zero chunks retrieved | `_no_evidence_response()` — `citations: []`, `validation.status = "no_evidence"` |
| All candidates below `MIN_RERANK_SCORE` | same path, with `RERANK THRESHOLD` logged |
| Reranker returns a degenerate spread | fall back to RRF order, log `RERANK DEGENERATE` — an arbitrary shuffle is worse than fusion |
| Answer contains a refusal phrase | citations cleared, confidence forced to 0, logged |
| Conversational intent (`ما آخر سؤال سألته؟`) | **no retrieval at all**, `citations: []`, `validation.status = "conversational"` |

This closes the defect where the UI displayed «لا تتوفر معلومات كافية» alongside
two cited sources — implying the refusal was evidenced when it was not.

## 7. Error contract

`api/exceptions.py` maps domain exceptions to a single envelope:

```json
{"error": {"message": "...", "error_code": "NOT_FOUND", "details": {}, "request_id": "..."}}
```

Middleware order matters and is non-obvious: **Starlette applies the last-added
middleware as the outermost layer.** CORS must be added *after*
`RequestContextMiddleware` so that error responses still carry CORS headers.
Exception handlers registered via `@app.exception_handler(Exception)` land on
`ServerErrorMiddleware`, which is *outside* CORS — a 500 therefore loses its CORS
headers unless CORS is the outermost layer. This cost several days of debugging.

## 8. Process topology

| Mode | Containers | Trade-off |
|---|---|---|
| `WORKER_EMBEDDED=true` | api, web, postgres, qdrant | Simpler. Ingestion contends with HTTP under the GIL. Safe now that the worker is built with `install_signal_handlers=False`. |
| `WORKER_EMBEDDED=false` + `--profile worker` | + worker | Isolated CPU budget. The worker builds its own embedding model + BM25, so it costs extra RAM and ~10–30 s at start. |

Uploads default to **job mode** (`useLocalPref("uploadMode", "job")`), so a
running worker is required in either mode. `NEXT_PUBLIC_DOCUMENT_UPLOAD_MODE` is
not read by any code — it is dead config.

## 9. Deliberately unused

| Component | Status | Reason |
|---|---|---|
| `src/storage/cache/` | Present, never constructed | `CACHE_ENABLED=false`, `redis` not in `requirements.txt`, `factory.py` builds no cache. Highest-value future use: answer + embedding caching. |
| `api/middleware/error_handling.py`, `request_logging.py` | Empty files | Duplicated by `request_context.py` + `api/exceptions.py`. Delete them. |
| `frontend/` | Removed | Superseded by `web/`. Older docs still referenced Streamlit. |
| `src/evaluation/` | Implemented, unwired | No CI step runs it. See `ROADMAP.md`. |

## 10. Hardware reality

NVIDIA Quadro P2000, **4 GB VRAM, Pascal sm_61**.

| Component | Device | Why |
|---|---|---|
| `multilingual-e5-base` (278M) | CPU | Fits in 4 GB at fp16 (~1.1 GB), but two models + CUDA context is tight and the P2000 is slow for transformer workloads. |
| `bge-reranker-base` (278M) | CPU | Same. |
| `Qwen2.5-3B-Instruct` | **cannot run usefully** | bf16 needs ~6 GB > 4 GB. `bitsandbytes` 4-bit requires Turing+ (sm_75+), so `LOCAL_MODEL_LOAD_IN_4BIT` does not work on Pascal. |
| Groq | remote API | The only viable generation path. |

Therefore `LOCAL_MODEL_NAME` is empty by default and the fallback chain is
API-only. `ProviderRouter` no longer force-appends `transformers`; loading a 3B
model that can never be used costs ~6 GB of RAM for nothing.

## 11. Documented decisions worth defending

1. **FastAPI + Next.js, not Node-only.** The retrieval stack
   (`sentence-transformers`, `rank-bm25`, `PyMuPDF`, `PaddleOCR`, `qdrant-client`)
   is Python-only. A Node backend would *add* a Python service rather than remove one.
2. **Synchronous `src/` core.** Phases 1–10 are sync Python. Endpoints are `def`,
   so FastAPI runs them in a threadpool. That is correct for I/O-bound work and
   *incorrect* for CPU-bound work — which is exactly why ingestion moved to its own
   process rather than being made async.
3. **Pure-ASGI-style middleware, single request id.** One id from nginx through
   FastAPI into logs, so a user-reported error maps to one grep.
4. **Denormalised visibility.** See §4.
5. **Measured, not assumed.** Every latency and score claim in this document comes
   from `logs/app.log`. The reranker diagnosis, the 28 s index build, the 8.5 s
   rerank cost — all observed, none inferred.
