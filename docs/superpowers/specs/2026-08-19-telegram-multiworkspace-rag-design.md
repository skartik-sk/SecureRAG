# Telegram Multi-Workspace RAG — Design

**Date:** 2026-08-19
**Status:** Approved in conversation; spec pending user review
**Constraint:** No git commits or pushes at any point — working tree only.

## 1. Goal

Turn the existing single-collection RAG into a multi-tenant, Telegram-first assistant:

1. A Telegram bot answers questions from workspace documents (RAG via Groq + PGVector).
2. Multiple **workspaces**, each with its own isolated vector collection. Seeded with three themed workspaces from `my_docs_folder/`. Users create new workspaces from Telegram or the API.
3. **Open onboarding with per-workspace access control**: anyone on Telegram may talk to the bot and gets a user profile automatically. Workspaces are public by default; owners can make them private and invite members.
4. Documents enter a workspace by **Telegram file upload** or **REST upload**; both run the same background ingestion pipeline (docling → markdown → chunk → embed).
5. Conversations persist across restarts (Postgres-backed LangGraph checkpointer).
6. Production-grade hygiene: typed config, structured logging, error handling, health check, Dockerfile, docker-compose, seed script, README runbook.

### Non-goals (explicitly out of scope)

- Job queue / worker processes (asyncio background task is enough for one worker; README notes the upgrade path).
- Chunk-level role filtering (the current `role` metadata filter in `rag_engine.py` is removed — workspace membership is the access boundary).
- JWT/user-password auth for the REST API (API keys only; Telegram identity handles humans).
- Streaming/token-by-token Telegram replies.
- Payments, analytics, admin web UI.

## 2. Architecture

Single FastAPI process. Telegram connectivity via `python-telegram-bot` (v21+, async), with two modes selected by `BOT_MODE`:

- **`polling`** (local dev): PTB `Application` runs long-polling inside FastAPI's lifespan; no public URL needed.
- **`webhook`** (production): Telegram POSTs updates to `POST /telegram/webhook`; the endpoint validates `X-Telegram-Bot-Api-Secret-Token`, deserializes the `Update`, and feeds it to the same PTB `Application` via `application.update_queue`.

One Postgres (pgvector) server stores relational tables, one PGVector collection per workspace, and LangGraph checkpoint tables.

### Repo layout

```
app/
├── __init__.py
├── config.py          # pydantic-settings; single source of env truth
├── db.py              # SQLAlchemy engine/session factory, schema init, PGVector connection
├── models.py          # SQLAlchemy ORM: users, workspaces, workspace_members, documents
├── schemas.py         # Pydantic request/response models
├── security.py        # X-API-Key dependency; workspace access predicates
├── main.py            # app factory + lifespan (DB init, bot startup/shutdown)
├── routers/
│   ├── chat.py        # POST /api/v1/workspaces/{slug}/chat
│   ├── workspaces.py  # GET/POST /api/v1/workspaces (+ members, privacy toggle)
│   ├── documents.py   # GET/POST /api/v1/workspaces/{slug}/documents
│   └── telegram.py    # POST /telegram/webhook, GET /health
├── bot/
│   ├── app.py         # build PTB Application, register handlers, start/stop helpers
│   ├── handlers.py    # /start /help /workspaces /newworkspace /invite /public /private /workspace, callback_query, text, document
│   └── keyboards.py   # inline workspace picker
└── rag/
    ├── store.py       # per-workspace PGVector collection registry (ws_<slug>)
    ├── ingestion.py   # download/parse/chunk/embed pipeline + documents row updates
    └── engine.py      # LangGraph build; workspace-bound retriever; PostgresSaver checkpointer
scripts/seed.py        # create admin + 3 themed workspaces from my_docs_folder/
tests/                 # pytest (unit + API + handler tests)
data/uploads/          # uploaded originals, per-workspace subfolders (gitignored)
Dockerfile
docker-compose.yml     # pgvector db + app
.env.example
README.md              # rewritten runbook
```

**Superseded files:** root `main.py` and `rag_engine.py` are deleted (their logic moves into `app/`); `rag_phase1–4.py` remain untouched as learning artifacts.

## 3. Data model

All timestamps UTC. IDs are UUIDs (except `telegram_id`).

### users
| column | type | notes |
|---|---|---|
| id | UUID PK | |
| telegram_id | BIGINT UNIQUE NULLABLE | null = API-only/service identity (not used initially) |
| telegram_username | TEXT NULLABLE | without `@` |
| display_name | TEXT | falls back to `"User {telegram_id}"` |
| is_platform_admin | BOOL default false | from `TELEGRAM_ADMIN_ID` env |
| current_workspace_id | UUID FK→workspaces NULLABLE | selected workspace |
| created_at / updated_at | TIMESTAMPTZ | |

### workspaces
| column | type | notes |
|---|---|---|
| id | UUID PK | |
| slug | TEXT UNIQUE | lowercase kebab; PGVector collection name is `ws_<slug>` |
| name | TEXT | |
| description | TEXT NULLABLE | |
| is_private | BOOL default false | |
| owner_id | UUID FK→users | creator |
| created_at | TIMESTAMPTZ | |

### workspace_members
| column | type | notes |
|---|---|---|
| workspace_id | UUID FK | PK(workspace_id, user_id) |
| user_id | UUID FK | |
| role | TEXT | `owner` \| `editor` \| `viewer` |
| created_at | TIMESTAMPTZ | |

Owner row is always inserted on workspace create (`role=owner`).

### documents
| column | type | notes |
|---|---|---|
| id | UUID PK | |
| workspace_id | UUID FK | |
| filename | TEXT | |
| file_ext | TEXT | normalized lowercase, no dot |
| byte_size | INT | |
| status | TEXT | `processing` \| `ready` \| `failed` |
| chunk_count | INT default 0 | |
| error | TEXT NULLABLE | failure reason |
| source | TEXT | `telegram` \| `api` \| `seed` |
| uploaded_by | UUID FK→users NULLABLE | |
| created_at / updated_at | TIMESTAMPTZ | |

### Vectors & checkpoints
- One PGVector collection per workspace: `ws_<slug>`, `use_jsonb=True`. Chunk metadata: `workspace_slug`, `document_id`, `source_file`, plus header metadata from the splitter.
- LangGraph `PostgresSaver` (tables `checkpoints`, `checkpoint_writes`, `checkpoint_blobs`) — created via its migrator at startup. Thread IDs: `tg:{telegram_id}:{workspace_slug}` for Telegram; `api:{user_ref}:{workspace_slug}` for REST chat.

### Access rules (single source of truth: `app/security.py`)

| action | public workspace | private workspace |
|---|---|---|
| list/see workspace | everyone (registered) | members only |
| chat | everyone (registered) | members only |
| ingest docs | everyone (registered) | owner + editor |
| invite members | n/a | owner + editor |
| toggle privacy | owner | owner |
| delete workspace | owner | owner (not exposed via Telegram in v1) |

Platform admins (`is_platform_admin`) act as owners everywhere. "Everyone (registered)" = any Telegram user who has sent `/start` (auto-registered).

## 4. Telegram bot UX

| command / trigger | behavior |
|---|---|
| `/start` | auto-create user; welcome; show inline keyboard of accessible workspaces |
| `/help` | command list |
| `/workspaces` | re-list accessible workspaces (inline buttons) |
| `/newworkspace <name> [private]` | create workspace; caller = owner; optional `private` flag |
| `/workspace` | show current workspace + doc stats |
| `/invite @username` | add Telegram user (must be registered) to current workspace as `viewer`; owner/editor only, private workspaces |
| `/public` / `/private` | owner toggles current workspace privacy |
| callback query (workspace button) | set `current_workspace_id`; confirm; show doc stats |
| document message | if a workspace is selected and caller may ingest → "📥 Processing <name>…" → background pipeline → "✅ N chunks from <name> are searchable" or failure notice; else explain what's missing |
| text message | if a workspace is selected and caller may read → RAG answer with source citations; else prompt to pick a workspace |

Message rules: replies split at 4096 chars (Telegram limit); MarkdownV2 escaping handled via `html` parse mode for safety; errors surface a friendly one-liner while the traceback is logged.

## 5. REST API

Auth: `X-API-Key` header, matched against `API_KEYS` env (comma-separated). Unauthenticated → 401 JSON. Webhook endpoint authenticated by Telegram secret token instead.

| method + path | body / params | response |
|---|---|---|
| `GET /health` | – | `{status, db: ok}` (no auth) |
| `GET /api/v1/workspaces` | – | list with doc counts |
| `POST /api/v1/workspaces` | `{name, description?, is_private?}` | created workspace (409 on slug collision) |
| `GET /api/v1/workspaces/{slug}/documents` | – | documents with status |
| `POST /api/v1/workspaces/{slug}/documents` | multipart `file` (+`uploaded_by` optional ref) | 202 `{document_id, status: processing}` |
| `POST /api/v1/workspaces/{slug}/chat` | `{message, user_ref}` | `{thread_id, response, sources[]}` |
| `POST /telegram/webhook` | Telegram `Update` JSON | 200 (secret-token validated) |

Errors: uniform JSON `{detail}` via exception handlers; 404 unknown slug; 403 no access; 400 bad input; 500 logged with correlation id.

## 6. Ingestion pipeline

1. Acquire bytes: Telegram `getFile` download or multipart upload; save to `data/uploads/{workspace_slug}/{document_id}.{ext}`.
2. Insert `documents` row (`status=processing`).
3. Background asyncio task: docling `DocumentConverter` → markdown → `MarkdownHeaderTextSplitter` (`#`/`##`) → fallback: if a file yields zero header chunks, split on raw markdown with `RecursiveCharacterTextSplitter` so headerless files still ingest.
4. `add_documents` to `ws_<slug>` with metadata; set `status=ready`, `chunk_count`.
5. On exception: `status=failed`, `error=str(e)`; notify the Telegram uploader if applicable.

Supported extensions: `.pdf .docx .pptx .xlsx .html .md`. Retry-with-backoff on the vector insert is retained from the current code.

## 7. RAG query path

Same single-node LangGraph as today, with three changes:

1. Retriever is bound to the caller's current workspace collection (`ws_<slug>`).
2. Checkpointer is `PostgresSaver`; thread per (user, workspace) so history persists.
3. Off-domain guard retained: if the best-match distance exceeds a threshold (config `OFF_TOPIC_DISTANCE`, default 0.95), answer "I can only answer questions about this workspace's documents." Answer format appends `Sources: file1.md, file2.pdf`.

## 8. Configuration (`app/config.py`, pydantic-settings)

| var | example | notes |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://langchain:langchain@localhost:5432/langchain` | |
| `GROQ_API_KEY` | | |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | |
| `TELEGRAM_BOT_TOKEN` | | from @BotFather |
| `TELEGRAM_ADMIN_ID` | `123456` | this Telegram user becomes platform admin |
| `TELEGRAM_WEBHOOK_SECRET` | random string | set via setWebhook |
| `BOT_MODE` | `polling` \| `webhook` | default `polling` |
| `WEBHOOK_URL` | `https://host/telegram/webhook` | required when mode=webhook |
| `API_KEYS` | `key1,key2` | REST auth |
| `OFF_TOPIC_DISTANCE` | `0.95` | |
| `UPLOAD_DIR` | `data/uploads` | |

`.env.example` lists all of them; `.env` stays gitignored.

## 9. Deployment & ops

- `Dockerfile`: python-slim, install requirements, non-root user, uvicorn entrypoint.
- `docker-compose.yml`: `db` (pgvector/pgvector:pg16) + `app`; app waits on db healthcheck; `python scripts/seed.py` runnable via `--profile seed`.
- README runbook: local dev (polling), production (webhook + `setWebhook` instructions with secret), seed usage, troubleshooting.
- New dependencies: `python-telegram-bot`, `sqlalchemy`, `langgraph-checkpoint-postgres`, `pydantic-settings`, `python-multipart`, `pytest`, `pytest-asyncio`.

## 10. Testing

- **Unit:** access-rule matrix (§3), slug→collection naming, workspace listing visibility, chunking fallback, reply splitting at 4096.
- **API:** FastAPI `TestClient`/`httpx ASGI` against a scratch schema with embeddings + LLM mocked (fake retriever returns fixed docs) — covers auth 401/403, workspace CRUD, 202 upload flow, chat response shape.
- **Bot handlers:** feed fabricated PTB `Update` objects to handlers with mocked services; assert replies and DB effects.
- Built test-first (superpowers TDD) during implementation.

## 11. Migration from current code

| current | becomes |
|---|---|
| `main.py` root FastAPI | `app/main.py` factory + routers |
| `rag_engine.py` sync_documents_from_folder | `app/rag/ingestion.py` (file-based entry kept for seed script) |
| `rag_engine.py` graph + MemorySaver | `app/rag/engine.py` + PostgresSaver |
| hardcoded connection string / collection | `app/config.py` + `app/rag/store.py` per-workspace registry |
| `user_role` in chat request | real identity: Telegram user or API key |
| `.env` GROQ key only | full typed config set |
