# =============================================================================
# Mizan — Egyptian Legal AI · Task Registry
#
# One place that encodes the CORRECT invocation of every operation, so the
# knowledge lives in the repo instead of in shell history.
#
#   make            → list every target
#   make doctor     → verify the environment before anything else
#   make up         → full Docker stack
#
# Destructive targets require CONFIRM=yes. They cannot fire by accident.
# =============================================================================

SHELL         := /bin/bash
.SHELLFLAGS   := -eu -o pipefail -c
.DEFAULT_GOAL := help
MAKEFLAGS     += --no-print-directory

# ---------------------------------------------------------------- overridable
PYTHON           ?= python
PIP              ?= $(PYTHON) -m pip
CONDA_ENV        ?= genai
API_HOST         ?= 0.0.0.0
API_PORT         ?= 8000
WEB_PORT         ?= 3000
WEB_DIR          ?= web
COMPOSE          ?= docker compose
QDRANT_CONTAINER ?= mizan-qdrant
QDRANT_IMAGE     ?= qdrant/qdrant:v1.12.4
CONFIRM          ?= no
EMAIL            ?= demo@mizan.local
PASSWORD         ?=

# Alembic config is NOT at the project root. Bare `alembic upgrade head` fails
# with "No 'script_location' key found in configuration".
ALEMBIC_INI      ?= src/storage/database/migrations/alembic.ini
ALEMBIC          := alembic -c $(ALEMBIC_INI)

# ------------------------------------------------------- runtime environment
# 12 cores + three CPU models: torch's default intra-op parallelism oversubscribes
# and slows EVERY request. Capping at 2 is the measured sweet spot.
export OMP_NUM_THREADS         := 2
export MKL_NUM_THREADS         := 2
export OPENBLAS_NUM_THREADS    := 2
export TOKENIZERS_PARALLELISM  := false
export PYTHONUNBUFFERED        := 1

# =============================================================================
##@ Help
# =============================================================================

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mMizan — task registry\033[0m\n"} \
	  /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 } \
	  /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } \
	  END { printf "\n" }' $(MAKEFILE_LIST)

# =============================================================================
##@ Setup
# =============================================================================

.PHONY: env
env: ## Create .env from the template (never overwrites)
	@test -f .env && echo ".env already exists — left untouched." \
	  || { cp .env.example .env; echo "Created .env — fill in POSTGRES_PASSWORD, JWT_SECRET_KEY, GROQ_API_KEY."; }
	@test -f $(WEB_DIR)/.env.local && true \
	  || { test -f $(WEB_DIR)/.env.example && cp $(WEB_DIR)/.env.example $(WEB_DIR)/.env.local || true; }

.PHONY: install
install: ## Install Python runtime dependencies
	$(PIP) install -r requirements.txt

.PHONY: install-dev
install-dev: ## Install runtime + tooling, and git hooks
	$(PIP) install -r requirements.txt
	$(PIP) install py-spy openapi-typescript 2>/dev/null || true
	-pre-commit install

.PHONY: web-install
web-install: ## Install frontend dependencies (clean, lockfile-exact)
	cd $(WEB_DIR) && npm ci

.PHONY: doctor
doctor: ## Verify env, services, config and required files
	@echo "── python ──"
	@$(PYTHON) -V
	@$(PYTHON) -c "import fastapi, sqlalchemy, qdrant_client, jwt, passlib; print('core imports OK')" \
	  || { echo "FAIL: run 'conda activate $(CONDA_ENV)' then 'make install'"; exit 1; }
	@echo "── services ──"
	@pg_isready -h localhost -p 5432 || echo "  postgres: DOWN"
	@$(PYTHON) -c "from src.vector_store.qdrant_store import QdrantVectorStore as Q; Q(); print('  qdrant: OK')" \
	  || echo "  qdrant: DOWN (make qdrant-up)"
	@echo "── config ──"
	@$(PYTHON) -c "from src.config.settings import settings as s; print('  api prefix:', s.API_PREFIX); print('  auth:', s.AUTH_ENABLED); print('  worker embedded:', s.WORKER_EMBEDDED)"
	@echo "── files ──"
	@test -f $(ALEMBIC_INI) && echo "  alembic.ini OK" || echo "  MISSING $(ALEMBIC_INI)"
	@test -f deployment/docker/entrypoint.api.sh && echo "  entrypoint OK" || echo "  MISSING entrypoint.api.sh (Docker build will fail)"
	@grep -q 'output: *"standalone"' $(WEB_DIR)/next.config.mjs && echo "  standalone OK" || echo "  MISSING output:'standalone' (web image will fail)"

# =============================================================================
##@ Services (native dev)
# =============================================================================

.PHONY: qdrant-up
qdrant-up: ## Start the Qdrant container (creates it if absent)
	@docker start $(QDRANT_CONTAINER) 2>/dev/null || docker run -d \
	  --name $(QDRANT_CONTAINER) --restart unless-stopped \
	  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
	  -v qdrant-data:/qdrant/storage $(QDRANT_IMAGE)
	@echo "Qdrant → http://localhost:6333"

.PHONY: qdrant-down
qdrant-down: ## Stop the Qdrant container (data volume survives)
	docker stop $(QDRANT_CONTAINER)

.PHONY: services
services: qdrant-up ## Bring up everything the native dev flow needs
	@pg_isready -h localhost -p 5432 || sudo systemctl start postgresql

# =============================================================================
##@ Database
# =============================================================================

.PHONY: migrate
migrate: ## Apply all pending migrations
	$(ALEMBIC) upgrade head

.PHONY: migration
migration: ## Autogenerate a revision: make migration m="add scope"
	@test -n "$(m)" || { echo 'usage: make migration m="describe the change"'; exit 1; }
	$(ALEMBIC) revision --autogenerate -m "$(m)"

.PHONY: db-current
db-current: ## Show the revision the database is actually on
	$(ALEMBIC) current

.PHONY: db-history
db-history: ## Show the full migration chain
	$(ALEMBIC) history --verbose

.PHONY: db-rollback
db-rollback: ## Step back one revision
	$(ALEMBIC) downgrade -1

.PHONY: db-reset
db-reset: ## DESTROY all rows (keeps schema + alembic_version). CONFIRM=yes
	@test "$(CONFIRM)" = "yes" || { echo "Refusing. Re-run: make db-reset CONFIRM=yes"; exit 1; }
	@$(PYTHON) -c "$$RESET_SCRIPT"

# Base.metadata.drop_all() MISSES processing_jobs — that table is created by a
# migration, not registered on the ORM Base. Discovering tables from pg_tables
# is the only reset that actually empties everything.
define RESET_SCRIPT
from sqlalchemy import text
from src.storage.database.database import get_database
db = get_database()
KEEP = {"alembic_version"}
with db.engine.begin() as conn:
    rows = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).fetchall()
    tables = sorted(r[0] for r in rows if r[0] not in KEEP)
    if not tables:
        raise SystemExit("No tables found — check DATABASE_URL")
    joined = ", ".join(chr(34) + t + chr(34) for t in tables)
    conn.execute(text("TRUNCATE TABLE " + joined + " RESTART IDENTITY CASCADE"))
    print("truncated:", tables)
    for t in tables:
        n = conn.execute(text("SELECT count(*) FROM " + chr(34) + t + chr(34))).scalar()
        print("  " + t + ":", n, "rows")
endef
export RESET_SCRIPT

# =============================================================================
##@ Run (native dev — one target per terminal)
# =============================================================================

.PHONY: api
api: ## Terminal A — FastAPI with scoped reload
	uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) \
	  --reload --reload-dir api --reload-dir src

.PHONY: api-prod
api-prod: ## Run the API without reload (production-like)
	uvicorn api.main:app --host $(API_HOST) --port $(API_PORT) --proxy-headers

.PHONY: worker
worker: ## Terminal B — standalone ingestion worker (WORKER_EMBEDDED=false)
	$(PYTHON) -m workers.ingestion_worker

.PHONY: web
web: ## Terminal C — Next.js dev server
	cd $(WEB_DIR) && npm run dev

.PHONY: dev
dev: ## Convenience: API + web in ONE terminal (logs interleave; Ctrl+C stops both)
	@echo "Logs interleave. For real work use separate terminals: make api / make web"
	@trap 'kill 0' INT TERM; $(MAKE) api & $(MAKE) web & wait

.PHONY: tunnel
tunnel: ## Expose the FRONTEND only (never tunnel 8000 as well)
	devtunnel host -p $(WEB_PORT) --allow-anonymous

# =============================================================================
##@ Docker
# =============================================================================

.PHONY: build
build: ## Build all images
	$(COMPOSE) build

.PHONY: up
up: ## Start the full stack (worker embedded in the API)
	$(COMPOSE) up -d
	@$(MAKE) ps

.PHONY: up-worker
up-worker: ## Start the stack with a SEPARATE worker process
	$(COMPOSE) --profile worker up -d
	@$(MAKE) ps

.PHONY: down
down: ## Stop the stack (volumes survive)
	$(COMPOSE) down

.PHONY: down-volumes
down-volumes: ## DESTROY the stack AND its volumes. CONFIRM=yes
	@test "$(CONFIRM)" = "yes" || { echo "Refusing. Re-run: make down-volumes CONFIRM=yes"; exit 1; }
	$(COMPOSE) down -v

.PHONY: ps
ps: ## Container status
	$(COMPOSE) ps

.PHONY: logs
logs: ## Follow API logs
	$(COMPOSE) logs -f api

.PHONY: logs-worker
logs-worker: ## Follow worker logs
	$(COMPOSE) logs -f worker

.PHONY: shell-api
shell-api: ## Open a shell inside the API container
	$(COMPOSE) exec api bash

.PHONY: restart
restart: ## Restart the API container only
	$(COMPOSE) restart api

# =============================================================================
##@ Quality
# =============================================================================

.PHONY: lint
lint: ## Ruff — lint api, src, workers, scripts
	ruff check api src workers scripts

.PHONY: format
format: ## Ruff — auto-fix and format
	ruff check --fix api src workers scripts
	ruff format api src workers scripts

.PHONY: typecheck
typecheck: ## mypy
	mypy api src

.PHONY: test
test: ## Run the whole test suite
	pytest tests -q

.PHONY: test-unit
test-unit: ## Unit tests only
	pytest tests/unit -q

.PHONY: test-cov
test-cov: ## Tests with a coverage report
	coverage run -m pytest tests -q
	coverage report -m
	coverage html
	@echo "HTML report → htmlcov/index.html"

.PHONY: web-lint
web-lint: ## Frontend lint
	cd $(WEB_DIR) && npm run lint

.PHONY: web-build
web-build: ## Frontend production build (verifies output:'standalone')
	cd $(WEB_DIR) && rm -rf .next && npm run build
	@test -f $(WEB_DIR)/.next/standalone/server.js \
	  && echo "standalone OK" \
	  || { echo "FAIL: add output:'standalone' to next.config.mjs"; exit 1; }

.PHONY: check
check: lint typecheck test ## Everything CI should gate on

# =============================================================================
##@ Data & corpus
# =============================================================================

.PHONY: ingest
ingest: ## Bulk-ingest a file: make ingest f=data/raw/official_laws/civil_law.md
	@test -n "$(f)" || { echo 'usage: make ingest f=path/to/file.md'; exit 1; }
	$(PYTHON) scripts/bulk_ingest.py "$(f)"

.PHONY: qdrant-stats
qdrant-stats: ## Point count + collection health (uses the configured API key)
	@$(PYTHON) -c "$$QDRANT_STATS"

define QDRANT_STATS
from src.vector_store.qdrant_store import QdrantVectorStore
s = QdrantVectorStore()
info = s.client.get_collection(s.collection_name)
print("collection :", s.collection_name)
print("status     :", info.status)
print("points     :", s.client.count(s.collection_name, exact=True).count)
print("dimension  :", s.collection_manager and info.config.params.vectors)
endef
export QDRANT_STATS

.PHONY: reconcile
reconcile: ## Compare Postgres documents against Qdrant vectors
	@$(PYTHON) -c "$$RECONCILE_SCRIPT"

define RECONCILE_SCRIPT
from sqlalchemy import text
from src.storage.database.database import get_database
from src.vector_store.qdrant_store import QdrantVectorStore

db = get_database()
with db.engine.connect() as conn:
    sql_ids = {r[0] for r in conn.execute(text("SELECT id FROM documents WHERE deleted_at IS NULL"))}

store = QdrantVectorStore()
if hasattr(store, "list_document_ids"):
    q_ids = set(store.list_document_ids())
else:
    q_ids = {str(c.document_id) for c in store.scroll_all()}

orphans = sorted(q_ids - sql_ids)
missing = sorted(sql_ids - q_ids)
print("postgres documents :", len(sql_ids))
print("qdrant document ids:", len(q_ids))
print("ORPHANED in qdrant :", len(orphans), "(vectors with no SQL row)")
print("MISSING from qdrant:", len(missing), "(SQL rows never indexed)")
for d in orphans[:20]:
    print("   orphan:", d)
for d in missing[:20]:
    print("   unindexed:", d)
if orphans:
    print("")
    print("Orphans out-rank fresh uploads under RRF and cost BM25 rebuild time.")
    print("Remove them: make qdrant-purge-orphans CONFIRM=yes")
endef
export RECONCILE_SCRIPT

.PHONY: qdrant-purge-orphans
qdrant-purge-orphans: ## DESTROY vectors with no Postgres row. CONFIRM=yes
	@test "$(CONFIRM)" = "yes" || { echo "Refusing. Re-run: make qdrant-purge-orphans CONFIRM=yes"; exit 1; }
	@$(PYTHON) -c "$$PURGE_SCRIPT"

define PURGE_SCRIPT
from sqlalchemy import text
from qdrant_client.models import Filter, FieldCondition, MatchAny, FilterSelector
from src.storage.database.database import get_database
from src.vector_store.qdrant_store import QdrantVectorStore

db, store = get_database(), QdrantVectorStore()
with db.engine.connect() as conn:
    keep = sorted(r[0] for r in conn.execute(text("SELECT id FROM documents")))

before = store.client.count(store.collection_name, exact=True).count
if keep:
    selector = FilterSelector(filter=Filter(
        must_not=[FieldCondition(key="document_id", match=MatchAny(any=keep))]
    ))
else:
    selector = FilterSelector(filter=Filter(must_not=[]))
store.client.delete(collection_name=store.collection_name, points_selector=selector, wait=True)
after = store.client.count(store.collection_name, exact=True).count
print("points before:", before, "after:", after, "removed:", before - after)
print("Restart the API so BM25 rebuilds from the clean corpus.")
endef
export PURGE_SCRIPT

.PHONY: reset-all
reset-all: ## DESTROY every row AND every vector. CONFIRM=yes
	@test "$(CONFIRM)" = "yes" || { echo "Refusing. Re-run: make reset-all CONFIRM=yes"; exit 1; }
	@$(MAKE) db-reset CONFIRM=yes
	@$(PYTHON) -c "from src.vector_store.qdrant_store import QdrantVectorStore as Q; s=Q(); s.client.delete_collection(s.collection_name); print('qdrant collection dropped; API recreates it empty')"
	@rm -rf data/documents/jobs/* 2>/dev/null || true
	@echo "Reset complete. Restart the API, hard-refresh the browser, re-register."

# =============================================================================
##@ Diagnostics
# =============================================================================

.PHONY: health
health: ## Probe every layer, from the host
	@curl -s -o /dev/null -w 'api    /health/live  %{http_code}\n' http://localhost:$(API_PORT)/health/live || true
	@curl -s -o /dev/null -w 'api    /health/ready %{http_code}\n' http://localhost:$(API_PORT)/health/ready || true
	@curl -s -o /dev/null -w 'web    /            %{http_code}\n' http://localhost:$(WEB_PORT)/ || true
	@curl -s -o /dev/null -w 'proxy  /health/live  %{http_code}\n' http://localhost:$(WEB_PORT)/health/live || true

.PHONY: smoke
smoke: ## Authenticated end-to-end timing: make smoke PASSWORD=...
	@test -n "$(PASSWORD)" || { echo 'usage: make smoke PASSWORD=yourpassword [EMAIL=you@x.com]'; exit 1; }
	@TOKEN=$$(curl -s -X POST http://localhost:$(API_PORT)/api/v1/auth/login \
	  -H 'Content-Type: application/json' \
	  -d '{"email":"$(EMAIL)","password":"$(PASSWORD)"}' \
	  | $(PYTHON) -c 'import sys,json;print(json.load(sys.stdin).get("access_token",""))'); \
	test -n "$$TOKEN" || { echo "LOGIN FAILED — check credentials"; exit 1; }; \
	echo "token length: $${#TOKEN}"; \
	for p in /api/v1/documents /api/v1/conversations "/api/v1/jobs?limit=30"; do \
	  curl -s -o /dev/null -w "%{http_code}  %{time_total}s  $$p\n" \
	    "http://localhost:$(API_PORT)$$p" -H "Authorization: Bearer $$TOKEN"; \
	done

.PHONY: logs-tail
logs-tail: ## Follow the application log
	tail -f logs/app.log

.PHONY: logs-latency
logs-latency: ## Per-request timing with correlation ids
	@grep "duration_ms" logs/app.log | tail -40

.PHONY: logs-retrieval
logs-retrieval: ## The lines that prove whether ranking works
	@grep -E "RERANK RAW|RERANK NO SIGNAL|RERANK THRESHOLD|RRF FUSION|DENSE FILTER|BM25 QUERY|BM25 READY|BM25 INCREMENTAL" \
	  logs/app.log | tail -30

.PHONY: logs-errors
logs-errors: ## Warnings and errors only
	@grep -E "\| (WARNING|ERROR|CRITICAL) \|" logs/app.log | tail -40

.PHONY: logs-startup
logs-startup: ## Startup cost and degraded components
	@grep -E "Logging initialized|Service construction|degraded components|Application startup" \
	  logs/app.log | tail -10

.PHONY: profile
profile: ## Live CPU profile of the API (needs py-spy + the pid)
	@test -n "$(pid)" || { echo 'usage: make profile pid=12345   (find it: pgrep -f uvicorn)'; exit 1; }
	py-spy top --pid $(pid)

.PHONY: dump-stacks
dump-stacks: ## Instant stack of every thread — for hangs
	@test -n "$(pid)" || { echo 'usage: make dump-stacks pid=12345'; exit 1; }
	py-spy dump --pid $(pid)

# =============================================================================
##@ Cleanup
# =============================================================================

.PHONY: clean
clean: clean-py clean-web ## Remove build artefacts (never touches data)

.PHONY: clean-py
clean-py:
	find . -type d -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -rf *.egg-info

.PHONY: clean-web
clean-web:
	rm -rf $(WEB_DIR)/.next $(WEB_DIR)/out

.PHONY: clean-logs
clean-logs: ## Truncate logs (keeps the files)
	@: > logs/app.log
	@rm -f logs/app.log.* logs/metrics.jsonl
	@echo "Logs cleared."