#!/usr/bin/env bash
# =============================================================================
# API entrypoint: migrate, then serve.
#
# Migrations run BEFORE the app starts because factory.py scrolls Qdrant at
# startup and reads the documents.scope column. Serving against an unmigrated
# schema is how you get a 500 on every document request.
#
# MIGRATE_STRICT=1 (default in production) aborts the container on failure
# rather than serving a degraded app.
# =============================================================================
set -euo pipefail

MIGRATE_STRICT="${MIGRATE_STRICT:-1}"
CONFIG="${ALEMBIC_CONFIG:-src/storage/database/migrations/alembic.ini}"

echo "[entrypoint] applying migrations (config=${CONFIG})"
if alembic -c "${CONFIG}" upgrade head; then
    echo "[entrypoint] migrations applied."
else
    if [ "${MIGRATE_STRICT}" = "1" ]; then
        echo "[entrypoint] FATAL: migration failed; refusing to start." >&2
        exit 1
    fi
    echo "[entrypoint] WARNING: migration failed; starting anyway." >&2
fi

# Uvicorn owns SIGTERM/SIGINT. The embedded worker (when enabled) is started with
# install_signal_handlers=False precisely so it cannot hijack these.
exec uvicorn api.main:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --workers "${UVICORN_WORKERS:-1}" \
    --proxy-headers \
    --forwarded-allow-ips '*'