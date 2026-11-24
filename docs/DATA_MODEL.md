# Data Model

PostgreSQL is the system of record. Qdrant is a **derived index** that must be
reconstructible from Postgres — if it isn't, you have a split-brain corpus.

## 1. Tables

### `users`
| Column | Type | Notes |
|---|---|---|
| `id` | `String(36)` PK | UUID4 string |
| `email` | unique | login identifier |
| `password_hash` | | bcrypt via passlib |
| `tenant_id` | `String(64)` | assigned server-side, never from the request body |
| `roles` | JSON/list | `user`, `admin` |

### `documents`
| Column | Type | Notes |
|---|---|---|
| `id` | `String(36)` PK | UUID4 string |
| `tenant_id` | `String(64)` | indexed |
| `owner_id` | `String(36)` NULL | FK-ish; **NULL ⇒ admin-only for private rows** |
| `scope` | `Enum(DocumentScope)` | `OFFICIAL` / `PRIVATE`, default `PRIVATE` |
| `original_filename` | `String(512)` | |
| `storage_key` | `String(512)` | relative to `DOCUMENT_STORAGE_DIR` |
| `file_size_bytes` | `BigInteger` | |
| `status` | `Enum(DocumentStatus)` | `PENDING`/`PROCESSING`/`COMPLETED`/`FAILED` |
| `error_message` | `Text` | surfaced to the UI |
| `chunk_count` | `Integer` | |
| `indexed` | `Boolean` | Qdrant write succeeded |
| `law_number`,`law_year`,`title` | | extracted metadata |
| `legal_metadata` | `JSON` | |
| `version`, `source_url` | | amendment/version tracking |
| `created_at`,`updated_at`,`deleted_at` | | soft delete |

Constraints/indexes:
```
UNIQUE (tenant_id, storage_key)
INDEX (tenant_id, status)
INDEX (tenant_id, law_number, law_year)
INDEX (tenant_id, scope, owner_id)      ← added by migration 0002
INDEX (scope)                          ← added by migration 0002
```

### `document_chunks`
`id`, `document_id` FK CASCADE, `tenant_id`, `chunk_index`, `content`,
`chunk_type`, `page_number`, `section_title`, `law_name`, `law_number`,
`law_year`, `article_number`, `embedding_status` (`pending`/`done`/`failed`),
`vector_id`, `metadata_json` (JSON).

`metadata_json` is what BM25 filters on. It **must** contain `scope` and
`owner_id`, or sparse filtering silently admits everything.

### `conversations` / `messages`
`conversations`: `id`, `tenant_id`, `owner_id`, `document_id` (FK SET NULL),
`title`, `metadata_json`, `status`, timestamps.
`messages`: `id`, `conversation_id` FK CASCADE, `role`, `content`, `provider`,
`model`, `tokens_used`, `latency_ms`, `citations_json`, `created_at`.

`citations_json` is the audit trail — it is what the UI replays when reopening a
conversation, so a citation that was suppressed at generation time must also be
absent here.

### `processing_jobs`
`id`, `tenant_id`, `requested_by`, `job_type`, `document_id`, `filename`,
`file_path`, `file_hash`, `status`, `attempt`, `max_attempts`, `progress`,
`error_code`, `error_message`, `result_json`, `metadata_json`,
`created_at`/`updated_at`/`started_at`/`finished_at`/`next_run_at`.

Liveness is `updated_at` freshness, reclaimed after
`WORKER_RECLAIM_STALE_AFTER_SECONDS`. There is **no owner filter** on the jobs
API yet — see `KNOWN_ISSUES.md` #1.

## 2. Enums

| Enum | Members | Persisted as |
|---|---|---|
| `DocumentStatus` | `PENDING PROCESSING COMPLETED FAILED` | name (uppercase) |
| `DocumentScope` | `OFFICIAL PRIVATE` | name (uppercase), PG type `documentscope` |
| `JobStatus` | `PENDING RUNNING COMPLETED FAILED CANCELLED RETRYING` | name |

⚠️ **Case trap.** SQL stores the enum *name* (`OFFICIAL`). The Qdrant payload and
the API response store the enum *value* (`official`). `_coerce_scope()` and
`normalise_scope()` accept both and fail closed to `private`. Any new code must
use those helpers, never a raw string compare.

## 3. Status mapping — the contract that broke twice

`DocumentService._public_status(db_status, chunk_count, indexed)`:

| DB status | chunks | indexed | public |
|---|---|---|---|
| `PENDING` | 0 | ✗ | `uploaded` |
| `PROCESSING` | 0 | ✗ | `processing` |
| `PROCESSING` | 12 | ✗ | `processing` |
| `COMPLETED` | 12 | ✓ | `completed` |
| `COMPLETED` | 12 | ✗ | `partial` |
| `COMPLETED` | 0 | ✗ | `failed` |
| `FAILED` | any | any | `failed` |

**Rule: non-terminal DB states must map to non-terminal public states.** The
previous implementation derived status from `chunk_count`, so `PROCESSING` with
0 chunks became `failed`. The client then stopped polling and displayed
"Ingestion failed" while the worker was still running.

## 4. Qdrant payload

```json
{
  "document_id": "a81dc349-…",
  "chunk_index": 7,
  "content": "…",
  "chunk_type": "article",
  "article_number": "147",
  "section_title": "أثر العقد",
  "law_name": "القانون المدني",
  "law_number": "131",
  "law_year": "1948",
  "page_number": 42,
  "tenant_id": "default",
  "owner_id": "36f80851-…",
  "scope": "official",
  "chunk_id": "…",
  "vector_id": "…",
  "storage_key": "default/a81dc349-…/source.md"
}
```

Point id: `uuid5(NAMESPACE_DNS, f"{document_id}_{chunk_index}")` — deterministic,
so re-ingesting a document overwrites rather than duplicates.

`scope` + `owner_id` here are **load-bearing for access control**, not metadata.

## 5. Filesystem layout

```
data/
├── documents/<tenant>/<document_id>/source.<ext>   # original bytes, retained
├── documents/jobs/<tenant>/<job_id>/<filename>     # staged upload awaiting claim
├── raw/official_laws/                              # curated corpus (read-only in compose)
├── processed/                                      # generated chunk exports
└── temp/                                           # scratch, not referenced by code
```

`data/documents` is a **bind mount shared by api and worker with the same uid
(10001)**. If the worker runs as a different uid it cannot read files the API
wrote, and jobs fail with permission errors that look like ingestion bugs.

## 6. Migrations

```
src/storage/database/migrations/
├── alembic.ini                 # NOT at project root — use alembic -c <path>
├── env.py
└── versions/
    ├── 0001_initial_schema.py
    └── 0002_document_scope.py  # idempotent: inspects live schema first
```

```bash
alembic -c src/storage/database/migrations/alembic.ini upgrade head
```

The container entrypoint runs this before serving, so drift cannot happen in
Docker. On a host run it is easy to forget — which is how `scope` ended up
existing in SQL but not in Alembic's history.

**Migration 0002 also backfills `owner_id`** from `processing_jobs.requested_by`,
because job-mode uploads historically wrote `owner_id = NULL`. Without that
backfill, enabling owner filtering makes every existing document invisible to its
uploader.

## 7. Consistency hazards

| Hazard | Symptom | Guard |
|---|---|---|
| Qdrant has points, Postgres has no row | Chat cites documents that 404; 28 s wasted BM25 build | `QdrantVectorStore.list_document_ids()` vs `documents.id` |
| Postgres row, Qdrant missing | `partial` status; chunks visible in library, not in search | `indexed` flag + `embedding_status` per chunk |
| SQL `scope` changed, payload not | Document appears/disappears unpredictably | `set_document_payload` on every scope change |
| Schema created by `create_all` instead of Alembic | `alembic_version` lies; future migrations collide | entrypoint always migrates; 0002 is idempotent |

Reconciliation check:

```bash
python - <<'PY'
from sqlalchemy import text
from src.storage.database.database import get_database
from src.vector_store.qdrant_store import QdrantVectorStore

db, vs = get_database(), QdrantVectorStore()
with db.engine.connect() as c:
    sql_ids = {r[0] for r in c.execute(text("SELECT id FROM documents"))}
qdrant_ids = vs.list_document_ids()
print("sql docs      :", len(sql_ids))
print("qdrant docs   :", len(qdrant_ids))
print("orphans       :", sorted(qdrant_ids - sql_ids))
print("unindexed     :", sorted(sql_ids - qdrant_ids))
PY
```

`orphans` must be empty. Anything in it is invisible dead weight that still costs
BM25 build time and can be retrieved.
