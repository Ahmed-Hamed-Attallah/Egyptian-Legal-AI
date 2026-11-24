# Security

Legal data has a stricter bar than ordinary SaaS: a leaked document or a
misattributed citation is a professional harm, not just a bug.

## 1. Identity and tenancy

| Layer | Mechanism |
|---|---|
| Authentication | JWT HS256, `Authorization: Bearer`, expiry `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` |
| Tenant | `AuthContext.tenant_id`, assigned server-side at registration — **never accepted from a request body** |
| Owner | `AuthContext.user_id` → `documents.owner_id`, `conversations.owner_id`, `processing_jobs.requested_by` |
| Roles | `user`, `admin` — permission matrix in `security/authorization.py` |
| Upload validation | extension allow-list, size cap, empty-file rejection (`security/file_validation.py`) |

Every service method takes explicit `tenant_id` / `user_id` / `is_admin`
parameters. There is no implicit "current user" inside `src/` — that is what makes
the worker able to reuse the same code path safely.

## 2. Visibility policy

```
tenant_id == T
AND ( scope == "official"
      OR ( scope == "private" AND owner_id == me ) )
```

Enforced in three places that must agree: Qdrant push-down + post-filter, BM25
`matches()`, SQL `_visibility_clause()`. Full rationale in `ARCHITECTURE.md` §4.

**Fail-closed rules:**

| Situation | Behaviour |
|---|---|
| `scope` missing/unrecognised in payload | treated as `private` ⇒ not visible to non-owners |
| `owner_id IS NULL` and `scope = PRIVATE` | admin only |
| Filter cannot be translated to Qdrant | **raise**, never search unfiltered |
| `tenant_id` absent on a search call | return nothing, log a warning |
| Access denied to a document | `404 Document not found`, not `403` — do not confirm existence |

## 3. Fixed vulnerabilities (recorded so they are not reintroduced)

| # | Defect | Impact | Fix |
|---|---|---|---|
| 1 | `if auth.is_admin:` in `has_permission()` | `is_admin` is a **bound method**, always truthy ⇒ every user had every permission | call it: `auth.is_admin()` |
| 2 | `qdrant_store.search()` swallowed filter-build errors and searched unfiltered | access-control bypass | `_build_qdrant_filter` raises |
| 3 | `_build_qdrant_filter` handled only `eq`, ignored nested groups | `and/or` policies silently became tenant-only ⇒ all documents visible | full tree translation + mandatory post-filter |
| 4 | `MetadataFilter.operator` declared but never evaluated; `neq/gt/lt/in` skipped | `or` behaved as `and`; comparison filters were no-ops | single `matches()` evaluator |
| 5 | `search_service._tenant_allowed`: `doc_tenant in (None, "", "default", tenant_id)` | missing tenant metadata passed; `"default"` passed for **any** tenant | removed; filter push-down instead |
| 6 | `CORS_ORIGINS: str` instead of `list[str]` | Starlette did substring matching, `OPTIONS` returned 400 | typed list + `cors_origins_list` |
| 7 | `CORS_ALLOW_ORIGIN_REGEX` single-segment | `re.fullmatch` never matched a two-dot dev-tunnel origin | `(\.[a-z0-9-]+)*` |
| 8 | Exception handler outside the CORS layer | 500 responses had no CORS headers ⇒ browsers reported "connection lost" | CORS added last (outermost) |

## 4. Open issues

See `KNOWN_ISSUES.md`. The material ones:

- **Job listing is not owner-scoped.** Any authenticated user can list every job
  in the tenant, including filenames and error messages of other users' uploads.
  The document library is correctly isolated; the job feed is not.
- **Chat retrieval does not pass the visibility filter.** `ChatService.ask` calls
  the orchestrator without `filters`, so answers can draw on other users' private
  chunks even though the library hides them. Retrieval scoping is the more
  important of the two.
- **No token revocation.** A stolen access token is valid for the full expiry
  window. Logout clears client state only.

## 5. Transport

**Production:** nginx is the only published port. The browser sees one origin, so
there are no CORS preflights at all. `api` and `web` publish on `127.0.0.1` only,
for host debugging.

```
browser ──► nginx:80 ─┬─► web:3000      (UI, SSR)
                      ├─► api:8000      (/api/v1/*, /health/*, /docs)
                      └─ postgres, qdrant: never published publicly
```

**Development:** the equivalent is `web/next.config.mjs` rewrites plus
`NEXT_PUBLIC_API_BASE_URL=/api/v1`. Same-origin, no preflight, and the API needs
no public tunnel.

**Never** expose the API through a second public tunnel while the proxy is in
place — that is how the login timeout was reproduced, and it publishes your data
plane.

## 6. Secrets

| Rule | Enforcement |
|---|---|
| No secrets in code | all values via `Settings` from `.env` |
| `.env` gitignored | `.gitignore` + `.env.example` template committed |
| Secret fields excluded from serialisation | `exclude=True` on `DATABASE_URL`, `*_API_KEY`, `JWT_SECRET_KEY`, `POSTGRES_PASSWORD` |
| Compose fails loudly | `${POSTGRES_PASSWORD:?set …}` |
| Logs mask credentials | `Database._safe_url()` prints `postgresql+psycopg://***@host/db` |

**Incident record:** `DATABASE_URL` was once pasted with a live password because
the redaction pattern matched only variable *names* containing
`KEY|PASSWORD|SECRET|TOKEN|DSN`. Redact by **value**, and rotate anything that
appears in a chat log.

## 7. Upload safety

| Control | Location |
|---|---|
| Extension allow-list | `settings.ALLOWED_FILE_TYPES` → `DocumentService.ingest_upload` |
| Size cap | `MAX_UPLOAD_SIZE_MB`, enforced server-side **and** in `nginx.conf` (`client_max_body_size 60m`) |
| Empty-file rejection | `IngestionError` before storage |
| Path sanitisation | `_safe_component()` strips anything outside `[A-Za-z0-9-_.]` and truncates |
| Tenant-scoped storage path | `data/documents/<sanitized_tenant>/<document_id>/source.<ext>` |
| Original bytes retained | audit requirement; also the only way to re-chunk after a parser fix |

Uploads are parsed, never executed. PDF parsing (`pymupdf`) and OCR
(`paddleocr`) are the attack surface — keep them updated, and treat OCR output as
untrusted text that must not reach a shell or SQL.

## 8. Prompt injection

Retrieved legal text is untrusted input, especially user uploads. Mitigations:

- Context is assembled as delimited blocks by `generation/context_builder.py`,
  not concatenated into the instruction.
- `response_validator.py` checks citation grounding, so an answer asserting
  something absent from the evidence is flagged.
- Citations are resolved from **metadata**, never from model-authored strings —
  a model cannot invent a law number that then renders as a real citation.
- Refusals suppress citations (§ `ARCHITECTURE.md` 6).

Not mitigated: a malicious upload could still contain text designed to steer
answers. For a legal product, **official sources should be weighted above user
uploads** in ranking, and user-derived claims labelled as such. `DISTINGUISH_USER_UPLOADS_FROM_OFFICIAL_LAWS`
exists in `Settings` and is now reflected in `RetrievedChunk.scope` / `is_official`,
but the generator does not yet use it to bias the prompt.

## 9. Audit trail

| Question | Answer source |
|---|---|
| Who uploaded what, when? | `documents.owner_id`, `created_at`, `storage_key` + retained bytes |
| Who asked what? | `messages` with `provider`, `model`, `tokens_used`, `latency_ms` |
| What evidence backed an answer? | `messages.citations_json` |
| Which model produced it? | `provider`/`model` columns + `[LLM] SUCCESS: groq …` log lines |
| What did the server do with a request? | `request_id` in every log line, propagated from nginx via `X-Request-ID` |

`logs/app.log` rotates at 10 MB × 5 files. For real retention, ship logs to
external storage — rotation is not auditing.

## 10. Pre-release checklist

- [ ] Job listing/get/cancel owner-scoped (`KNOWN_ISSUES.md` #1)
- [ ] Chat retrieval passes the visibility filter (#2)
- [ ] Token revocation or short expiry + refresh (#4)
- [ ] `/docs`, `/redoc`, `/openapi.json` blocked at nginx for non-admin IPs
- [ ] HTTPS terminated (HSTS only after TLS exists)
- [ ] `JWT_SECRET_KEY` ≥64 random chars, rotated
- [ ] Postgres/Qdrant not reachable outside the compose network
- [ ] Rate limits exercised (login brute force, chat flooding)
- [ ] Cross-user access test: A uploads, B cannot list, fetch, chunk-read, or retrieve it
- [ ] Corpus reconciliation clean: zero orphaned Qdrant points
