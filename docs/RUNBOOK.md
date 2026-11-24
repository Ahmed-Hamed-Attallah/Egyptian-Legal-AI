# Runbook

Host: Ubuntu 24.04, Python 3.11, 32 GB RAM, 12 cores, Quadro P2000 4 GB (unused for
inference). Project root: `/run/media/ahmed/Data/RAG/Egyptian-Legal-AI` (ext4 on NVMe).

## 1. Development — 4 terminals

### Terminal A — data plane

```bash
# PostgreSQL: a system service, enabled at boot. Verify, don't start.
pg_isready -h localhost -p 5432
sudo systemctl start postgresql        # only if the above fails

# Qdrant. Use a named container with a restart policy, not a random one.
docker start mizan-qdrant 2>/dev/null || docker run -d \
  --name mizan-qdrant --restart unless-stopped \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v qdrant-data:/qdrant/storage \
  qdrant/qdrant:v1.12.4

curl -s http://localhost:6333/collections/egyptian_legal \
  | python -c "import sys,json;d=json.load(sys.stdin);print(d['result']['status'],d['result']['points_count'])"
```

### Terminal B — API

```bash
cd /run/media/ahmed/Data/RAG/Egyptian-Legal-AI
conda activate genai

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

uvicorn api.main:app --host 0.0.0.0 --port 8000 \
  --reload --reload-dir api --reload-dir src
```

**Wait for `Application startup complete.`** Startup costs:

| Component | Cold | Warm (models cached) |
|---|---|---|
| Embedding model | ~5 s | ~1 s |
| Qdrant scroll + BM25 (32 chunks) | <1 s | <1 s |
| Qdrant scroll + BM25 (45k chunks) | **~28 s** | **~28 s** |
| Cross-encoder | ~3 s | ~1 s |
| Qwen2.5-3B *(should be disabled)* | ~1 s + ~6 GB | — |

`--reload-dir` matters: plain `--reload` watches the whole project including
`web/node_modules` and `artifacts/models`.

### Terminal C — Web

```bash
cd web && npm run dev
```

`NEXT_PUBLIC_*` is inlined at build time. After changing `web/.env.local`:

```bash
rm -rf .next && npm run dev
```

### Terminal D — worker (only if `WORKER_EMBEDDED=false`)

```bash
cd /run/media/ahmed/Data/RAG/Egyptian-Legal-AI && conda activate genai
python -m workers.ingestion_worker
```

| `WORKER_EMBEDDED` | Do you run Terminal D? | Notes |
|---|---|---|
| `true` | **No** | Worker runs inside the API process. Safe: built with `install_signal_handlers=False` |
| `false` | **Yes, mandatory** | Otherwise uploads return `202` and sit in the queue forever |

Uploads default to **job mode** (`useLocalPref("uploadMode", "job")`), so a worker
must exist in either mode. `NEXT_PUBLIC_DOCUMENT_UPLOAD_MODE` is dead config.

## 2. Phone / instructor access

One tunnel, frontend only:

```bash
devtunnel host -p 3000 --allow-anonymous
```

Requires `NEXT_PUBLIC_API_BASE_URL=/api/v1` and the `next.config.mjs` rewrites.
**Do not** also tunnel 8000 — that publishes your data plane and reintroduces CORS.

Verify the proxy without any tunnel:

```bash
curl -s -o /dev/null -w 'health %{http_code}\n' http://localhost:3000/health/live
curl -s -o /dev/null -w 'docs   %{http_code}\n' http://localhost:3000/docs
curl -s -X POST http://localhost:3000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"…","password":"…"}' \
  -w '\nlogin  %{http_code} in %{time_total}s\n'
```

## 3. Production — Compose

```bash
cp .env.example .env                 # then fill in secrets
docker compose --profile worker up -d --build

docker compose ps                    # all healthy?
docker compose logs -f api
docker compose --profile worker logs -f worker
```

| Flag | Effect |
|---|---|
| `--profile worker` | separate ingestion process; set `WORKER_EMBEDDED=false` |
| `--profile cache` | starts Redis (currently unused by code) |
| `-e INSTALL_OCR=1` | build OCR into the images; otherwise keep `OCR_ENABLED=false` |

Migrations run automatically in the API entrypoint before serving
(`MIGRATE_STRICT=1` aborts on failure).

## 4. Verification suite

```bash
# 0. Services
pg_isready -h localhost -p 5432
curl -s -o /dev/null -w 'qdrant %{http_code}\n' http://localhost:6333/collections/egyptian_legal
curl -s -o /dev/null -w 'api    %{http_code}\n' http://localhost:8000/health/live
curl -s -o /dev/null -w 'web    %{http_code}\n' http://localhost:3000/

# 1. Authenticated timings — the only meaningful latency measurement
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' -d '{"email":"…","password":"…"}' \
  | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
echo "token length: ${#TOKEN}"          # must be > 100, else you measured a 401

for p in /api/v1/documents /api/v1/conversations "/api/v1/jobs?limit=30"; do
  curl -s -o /dev/null -w "%{http_code}  %{time_total}s  $p\n" \
    "http://localhost:8000$p" -H "Authorization: Bearer $TOKEN"
done
# expect 200 in 5–15 ms each

# 2. Startup telemetry
grep -E "Service construction|degraded components|BM25 READY" logs/app.log | tail -5

# 3. Retrieval quality — the lines that tell you whether ranking works
grep -E "RERANK RAW|RERANK NO SIGNAL|RERANK DEGENERATE|RERANK THRESHOLD|RRF FUSION|DENSE FILTER|BM25 FILTER" \
  logs/app.log | tail -20

# 4. Corpus reconciliation
python - <<'PY'
from sqlalchemy import text
from src.storage.database.database import get_database
from src.vector_store.qdrant_store import QdrantVectorStore
db, vs = get_database(), QdrantVectorStore()
with db.engine.connect() as c:
    sql = {r[0] for r in c.execute(text("SELECT id FROM documents"))}
q = vs.list_document_ids()
print("sql:", len(sql), "| qdrant docs:", len(q),
      "| orphans:", sorted(q - sql), "| unindexed:", sorted(sql - q))
PY
```

## 5. Full reset

Empties Postgres (including users) and Qdrant. Use this, **not**
`Base.metadata.drop_all` — `processing_jobs` is created by a migration and is not
registered on `Base.metadata`, so `drop_all` skips it and old jobs survive.

```bash
cd /run/media/ahmed/Data/RAG/Egyptian-Legal-AI && conda activate genai

python - <<'PY'
from sqlalchemy import text
from src.storage.database.database import get_database
db = get_database()
KEEP = {"alembic_version"}
with db.engine.begin() as conn:
    tables = sorted(r[0] for r in conn.execute(text(
        "SELECT tablename FROM pg_tables WHERE schemaname='public'"))
        if r[0] not in KEEP)
    assert tables, "no tables found — wrong DATABASE_URL?"
    conn.execute(text(f'TRUNCATE TABLE {", ".join(chr(34)+t+chr(34) for t in tables)} '
                      f'RESTART IDENTITY CASCADE'))
    print("truncated:", tables)
    for t in tables:
        print(f"  {t}: {conn.execute(text(f'SELECT count(*) FROM \"{t}\"')).scalar()} rows")
PY

python - <<'PY'
from src.vector_store.qdrant_store import QdrantVectorStore
s = QdrantVectorStore()
try: before = s.client.count(s.collection_name, exact=True).count
except Exception as e: before = 0; print("count skipped:", e)
s.client.delete_collection(s.collection_name)
print(f"dropped '{s.collection_name}' (had {before} points); API recreates it empty")
PY

# Staged upload bytes
rm -rf data/documents/jobs/*
```

Restart the API, hard-refresh the browser (Ctrl+Shift+R), re-register.

> A reset does **not** fix job scoping. A second real user will still see the
> first user's jobs until `KNOWN_ISSUES.md` #1 is closed.

## 6. Ingestion smoke test

```bash
curl -s -X POST http://localhost:8000/api/v1/jobs/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@egyptian_legal_corpus.md" | python -m json.tool

# Watch it progress: 202 → queued → processing → completed
curl -s "http://localhost:8000/api/v1/jobs?limit=5" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool

grep -E "Ingestion finished|Job completed|Job failed|BM25 INCREMENTAL" logs/app.log | tail -10
```

Expected log: `Ingestion finished document=… owner=… scope=private chunks=32 indexed=True`
followed by `BM25 INCREMENTAL | added=32`.

## 7. Troubleshooting — symptom → cause

| Symptom | Cause | Fix |
|---|---|---|
| `Chat service is unavailable.` (503) | `rag_orchestrator` failed to build. Grep for `RAG orchestrator initialization failed` | Usually a `RAGPipeline.__init__()` kwarg mismatch — `factory.py` must pass `retrieval_pipeline=` and `pipeline.py` must accept it |
| Every chat answer is «لا تتوفر معلومات كافية» | Orphaned vectors dominating RRF, or `MIN_RERANK_SCORE` now correctly rejecting garbage | Run the reconciliation in §4.4; delete orphans |
| `RERANK RAW` absent from logs | Old `cross_encoder_reranker.py` still running | Restart; the fixed version logs `RERANK RAW` on every call |
| `top_score=0.5000 bottom_score=0.5000` | Double sigmoid (activation trap) | Apply the identity-activation fix; confirm `default_activation=Sigmoid() (bypassed…)` at startup |
| Uploaded document not found by search | BM25 frozen at startup | `RetrievalPipeline` must be injected into `RAGPipeline`; look for `BM25 INCREMENTAL` |
| Document shows "Ingestion failed" while processing | `_public_status` mapped `PROCESSING`+0 chunks → `failed` | Use the status table in `DATA_MODEL.md` §3 |
| `Object of type UUID is not JSON serializable` | Engine missing `json_serializer=json_dumps` | `create_engine(..., json_serializer=json_dumps)` in `storage/database/database.py` |
| `API unreachable` badge while the API serves 200s | A component with its own health poller flipping on one abort | Only `ApiStatusBadge` + `lib/health.ts` may poll; needs 3 consecutive transport failures |
| `انتهت مهلة الاتصال` on login via tunnel | Dev-tunnel transport, not the API | Use the proxy + single tunnel (§2); measure with `curl` on localhost |
| Requests logged minutes after sending, out of order | Uvicorn `--reload` opens the listening socket in the parent, so traffic queues during the 28 s startup | Wait for `Application startup complete`; compose uses `depends_on: service_healthy` |
| `FAILED: No 'script_location' key found in configuration` | `alembic.ini` is under `src/…/migrations/`, not the root | `alembic -c src/storage/database/migrations/alembic.ini upgrade head` |
| Worker exits immediately after loading models | Root `workers/ingestion_worker.py` lacks an event loop | Use the entry point provided in `deployment/` notes; it calls `asyncio.run(worker.run())` |
| `Must provide an API key` from Qdrant curl | `QDRANT_API_KEY` is set | Use `QdrantVectorStore` from Python, or add `-H "api-key: $QDRANT_API_KEY"` |
| Login 200 in 233–500 ms | bcrypt cost — **normal** | Do not "optimise" this |
| `duration_ms` lines missing from `logs/app.log` | `setup_logging()` never called ⇒ `logging.lastResort` drops INFO | `api/lifespan.py` must call `setup_logging()`, not `get_logger()` |

## 8. Shutdown

```
Ctrl+C  web
Ctrl+C  worker (if separate)
Ctrl+C  api            # embedded worker gets 20 s grace to finish its job
docker stop mizan-qdrant
# leave PostgreSQL running — it is a system service
```

Stop the API **before** Qdrant: an in-flight ingestion writes vectors, and killing
Qdrant first produces `VectorDatabaseError` and a failed job.

## 9. Backups

| What | Command | Frequency |
|---|---|---|
| Postgres | `pg_dump -Fc egyptian_legal_ai > backup.dump` | daily |
| Qdrant | `POST /collections/egyptian_legal/snapshots` → `qdrant-snapshots` volume | before bulk ingest |
| Uploads | `data/documents/` (bind mount) | rsync daily — these are the only copy of user originals |
| Models | none needed | re-downloadable; `hf-cache` volume is disposable |

Qdrant is **derived** data. If Postgres + `data/documents` are intact, a full
re-index is possible. If `data/documents` is lost, uploaded originals are gone
forever and chunks cannot be regenerated after a parser change.
