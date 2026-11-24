# Known Issues

Ordered by severity. Each entry states the evidence, not a suspicion.

---

## 1. 🔴 Job listing is not owner-scoped — active data leak

**Evidence:** `api/routes/jobs.py` calls `job_manager.list_jobs(tenant_id=…, …)`
with no owner predicate. Observed directly: a brand-new account with **zero**
documents saw five jobs belonging to another user, including filenames
(`README_EXPERIMENTS.md`, `MANIFEST.md`, `04_synthetic_egyptian_legal_test.pdf`)
and full error messages.

**Impact:** any authenticated user can enumerate other users' uploads by filename,
see failure details, and — via `GET /jobs/{id}` and `POST /jobs/{id}/cancel` — read
and cancel foreign jobs.

**Fix:** mirror `DocumentRepository._visibility_clause()`:

| Operation | Rule |
|---|---|
| `list_jobs` | `requested_by == me OR is_admin()` |
| `get_job` | same |
| `cancel_job` | same |
| `submit_document_job` | already sets `requested_by` correctly |

`JobStore` needs an owner parameter on `list()`/`get()`. The worker keeps
unscoped access — it claims jobs by design.

---

## 2. 🔴 Chat retrieval ignores the visibility filter

**Evidence:** `ChatService.ask` → `orchestrator.ask(question, top_k, history,
memory_digest)`. No `filters` argument. `RAGPipeline.process` accepts
`filters: Optional[MetadataFilter]` and passes it down, but nothing supplies it.

**Impact:** the library correctly hides another user's documents, but a chat
answer can still be built from their private chunks — and the citation would
expose their content. This is a worse leak than #1 because it exfiltrates text.

**Fix:** `ChatService` builds
`MetadataFilter.for_documents(tenant_id, user_id, scope, is_admin)` from
`AuthContext` and threads it through `RAGOrchestrator.ask` → `RAGPipeline.process`.

---

## 3. 🔴 `authorization.py` grants every permission to every user

**Evidence:**

```python
def has_permission(auth, permission):
    if auth.is_admin:      # bound method → always truthy
        return True
```

`AuthContext.is_admin` is a method. `require_permission()` therefore never raises.

**Status:** fix authored (`auth.is_admin()`); **verify it is applied** —

```bash
grep -n "is_admin" src/security/authorization.py | head
```

Every occurrence must be a call `is_admin()`, never a bare attribute.

---

## 4. 🟠 No token revocation

60-minute HS256 tokens, logout clears client state only. A leaked token remains
valid for its full window. Mitigation today: short expiry. Real fix: denylist
(Redis would serve this well and is otherwise unused) or refresh/access split.

---

## 5. 🟠 BM25 rebuild cost scales with corpus pollution

**Measured:** 45,202 chunks → **28.1 s** blocking startup. 32 chunks → <1 s.

The build runs synchronously inside the async lifespan. With `--reload`, Uvicorn's
listening socket is created by the reloader **parent**, so requests arrive and
queue during those 28 s, then all complete at once — producing out-of-order access
logs that look like server lag but are startup contention.

**Fixes:** keep the corpus clean (#6); persist the index to `artifacts/indexes/`
(roadmap D); compose uses `depends_on: service_healthy` so nothing queues.

---

## 6. 🟠 Orphaned Qdrant vectors (recurring hazard)

Wiping Postgres without Qdrant leaves vectors with no backing row. Symptoms seen:
chat citing `قانون رقم ٣٦٣ لسنة ١٩٥٦` which is **not in any uploaded file**;
`GET /documents/{id}` → 404 for a document whose vectors exist; new uploads
out-ranked by orphans under RRF.

**Prevention:** never truncate `documents` alone. Use the RUNBOOK §5 reset, or
`delete_by_document()` on every document deletion. Run the reconciliation script
in `DATA_MODEL.md` §7 regularly.

---

## 7. 🟠 `AnswerGenerator(conversational=True)` unverified

`RAGPipeline._process_conversational` detects the parameter and degrades with a
warning if absent. If it is absent, meta questions get an **empty-context** call
rather than a memory-only prompt — better than legal retrieval, but not the
designed behaviour.

---

## 8. 🟡 Local 3B model still loads unless `.env` is fixed

`ProviderRouter` no longer force-appends `transformers`, but only when
`LOCAL_MODEL_NAME` is empty. Startup logs still showed:

```
src.config.model_config | GPU not available. llm will run on CPU.
transformers_provider | Loading local model: Qwen/Qwen2.5-3B-Instruct on cpu
```

Cost: ~6 GB RAM and ~1 s. Set `LOCAL_MODEL_NAME=` and `LLM_FALLBACK_ORDER=`.

---

## 9. 🟡 Empty duplicate middleware files

`api/middleware/error_handling.py` and `request_logging.py` are empty; their
responsibilities live in `request_context.py` + `api/exceptions.py`. Delete them
plus any references in `api/middleware/__init__.py`.

---

## 10. 🟡 Dead configuration keys

| Key | Status |
|---|---|
| `CACHE_ENABLED`, `REDIS_URL`, `CACHE_DEFAULT_TTL` | no cache constructed; `redis` not installed |
| `NEXT_PUBLIC_DOCUMENT_UPLOAD_MODE` | no code reads it; `uploadMode` comes from `localStorage`, default `"job"` |
| `QDRANT_COLLECTION` | duplicate of `QDRANT_COLLECTION_NAME`; the store reads the latter |
| `METRICS_ENABLED` | declared twice in `.env` (false, then true) — last wins, confusing |
| `TRACING_ENABLED` | `monitoring/tracing.py` not wired to any span emission |

---

## 11. 🟡 Repo hygiene

| Item | Problem |
|---|---|
| `القانون التجاري المصري.md`, `companies_law.md` in project root | inconsistent with `data/raw/official_laws/`; non-ASCII name complicates mounts |
| `*_Combined.txt`, `combined/`, `file.txt` | debugging dumps, not source |
| `egypt_legal_ai.egg-info/` committed | build artifact |
| `logs/app.log.1` … `.5` | rotation is fine, but logs should not be in git |
| `data/temp/` screenshots, sample CSVs | scratch data |
| no `.gitignore`, no `.dockerignore` | **added in this delivery** — verify before first commit |

---

## 12. 🟡 Test and CI gaps

`tests/api/`, `tests/integration/`, `tests/evaluation/` contain only
`__init__.py`. Three unit files exist. No GitHub Actions workflow. For a
"production-oriented" legal system, the untested paths are exactly the ones that
failed in this debugging arc: status mapping, filter evaluation, error envelopes.

---

## 13. ℹ️ Historical errors still visible in job rows

`[IngestionError] … Object of type UUID is not JSON serializable` on jobs from
Nov 16 and earlier. Root cause (`json_serializer` missing) is fixed. These rows are
**stale records**, not current failures. The reset in RUNBOOK §5 clears them; do
not re-diagnose them.

---

## 14. ℹ️ Reranker cost vs benefit — unresolved decision

Reranking is **65–88% of chat latency** (7.4–8.8 s of ~10–12 s). It is only worth
paying if `RERANK RAW` shows a real logit spread on a clean corpus. Decision rule:

- spread ≥ ~2.0 → keep it, consider GPU
- spread < 0.1 → `RERANKER_ENABLED=false`, rely on RRF over a clean corpus

Do not leave it enabled in the degenerate state: it costs 8.5 s to return a random
ordering.

---

## Non-issues (do not chase these)

| Observation | Why it is fine |
|---|---|
| Login takes 233–500 ms | bcrypt key derivation. Expected and desirable. |
| `GET /documents` takes 5–15 ms | Correct. The library is fast; the perception of slowness was transport (tunnel) and startup contention. |
| Two `POST /auth/login` 401s at 11:57 | `$TOKEN` was unset in the shell, so the header was empty. Not an API fault. |
| `OPTIONS` lines in the log | Correct CORS behaviour while the API was cross-origin. Disappears entirely behind the proxy/nginx. |
| `0 indexed / 0 chunks` for a new account | **Working as designed** — the ownership filter is functioning. |
```

---
