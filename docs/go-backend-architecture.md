# Multi-Workspace Telegram RAG — Full Architecture for a Go Implementation

**Date:** 2026-08-19
**Origin:** Design approved for the Python prototype in this repo (`docs/superpowers/specs/2026-08-19-telegram-multiworkspace-rag-design.md`). This document is self-contained: you can implement from it without reading the Python code.
**Scope:** One Go backend + Postgres(pgvector) + one tiny Python parser sidecar. Multi-workspace RAG served over a Telegram bot and a REST API.

---

## 1. What the system does

1. A **Telegram bot** answers questions using only the documents of the workspace the user has selected (RAG: pgvector retrieval + Groq LLM).
2. **Workspaces** isolate document sets. Each workspace owns its own vector space. Seeded with three themed workspaces (see §12); users create more via bot or API.
3. **Open onboarding, per-workspace access control**: anyone on Telegram may use the bot — a user profile is auto-created on first contact. Workspaces are public by default; owners can make them private and invite members.
4. Documents enter a workspace via **Telegram file upload** or **REST upload** → background pipeline: parse → markdown → chunk → embed → store.
5. Conversations persist (Postgres-backed message history), survive restarts.
6. Production hygiene: typed config, structured logging, health checks, Docker deploy, seed script.

### Non-goals

- Distributed job queue (goroutine + status column is enough for one instance; §10 notes the upgrade path).
- JWT/password auth (Telegram identity for humans, API keys for machines).
- Streaming replies, admin web UI, analytics, workspace deletion UX in v1.

---

## 2. System overview

```
                    ┌────────────────────────────────────────────┐
 Telegram ──HTTP──▶ │  Go backend (single binary/service)        │
 (users, files)     │                                            │
                    │  ┌──────────┐   ┌───────────────────────┐  │
 REST clients ────▶ │  │ REST API │   │ Telegram bot module   │  │
 (X-API-Key)        │  └────┬─────┘   │ (webhook or polling)  │  │
                    │       │         └──────────┬────────────┘  │
                    │       ▼                    ▼               │
                    │  ┌──────────────────────────────┐          │
                    │  │ core services                │          │
                    │  │  auth · workspaces · ingest  │          │
                    │  │  rag (embed/retrieve/answer) │          │
                    │  └───────┬──────────────┬──────┘          │
                    └──────────┼──────────────┼─────────────────┘
                               │              │
              ┌────────────────▼───┐   ┌──────▼────────────┐   ┌──────────────┐
              │ Postgres 16        │   │ Groq API (HTTPS)  │   │ parser       │
              │ + pgvector         │   │ chat completions  │   │ sidecar      │
              │ tables + vectors   │   └───────────────────┘   │ (Python,     │
              └────────────────────┘   ┌──────────────────────┐  docling)     │
                                       │ embeddings (local    │  §5.3         │
                                       │ ONNX or HF API)      │  only at      │
                                       └──────────────────────┘  ingestion    │
                                                                  └──────────────┘
```

Two runtime modes for the bot, switched by env (`BOT_MODE`):

- **`polling`** — dev: the bot module long-polls `getUpdates`. No public URL needed.
- **`webhook`** — prod: Telegram POSTs to `POST /telegram/webhook`; you validate `X-Telegram-Bot-Api-Secret-Token` against your secret, then process the update.

Ingestion is asynchronous everywhere: the API/bot accepts the file, creates a `documents` row with `status=processing`, and a goroutine finishes the work while the user gets an immediate acknowledgment.

---

## 3. Go technology decisions (and why)

| Concern | Choice | Rationale / import path |
|---|---|---|
| HTTP router | `chi` v5 | idiomatic, middleware (request-id, logging, auth) without a framework. `github.com/go-chi/chi/v5` |
| Postgres driver | `pgx` v5 (pool) | de-facto production driver. `github.com/jackc/pgx/v5` |
| Vectors | `pgvector-go` | `vector.NewVector([]float32)` binds to pgvector. `github.com/pgvector/pgvector-go` |
| Migrations | `golang-migrate` | plain SQL up/down files in `migrations/`. `github.com/golang-migrate/migrate/v4` |
| Telegram | `go-telegram/bot` | actively maintained, context-based, supports both webhook and polling. `github.com/go-telegram/bot` |
| Config | `caarlos0/env` or stdlib `os.Getenv` + typed struct | keep it boring; validate at startup, fail fast |
| Logging | stdlib `log/slog` (JSON handler) | structured, zero deps |
| Embeddings (recommended) | `fastembed-go` | in-process ONNX, supports `BAAI/bge-small-en-v1.5` (384 dims) — same model as the Python prototype. `github.com/Anush008/fastembed-go` |
| Embeddings (fallback) | HuggingFace Inference API or Ollama sidecar | if ONNX runtime is a problem on your machine; same 384-dim model |
| LLM | Groq chat completions over plain HTTPS | no SDK needed; `POST https://api.groq.com/openai/v1/chat/completions` |
| Document parsing | **Python sidecar running docling** | see §5.4 — this is the one place Go has no good equivalent |
| Tests | stdlib `testing` + `net/http/httptest` | |

### 3.1 Deliberate non-choices

- **No LangChain-for-Go / langchaingo.** The pipeline is three explicit functions (embed, retrieve, answer). Orchestration frameworks buy nothing here and cost debuggability. The Python prototype's LangGraph graph compiles to: guard → retrieve → prompt → call LLM → return. Write that as straight-line Go.
- **No ORM.** pgx + hand-written SQL in a thin repository layer. You have ~8 queries that matter; typed rows beat reflection.

### 3.2 Python → Go concept mapping

| Python prototype | Go equivalent |
|---|---|
| FastAPI app + routers | chi router + handlers in `internal/httpapi` |
| python-telegram-bot | `go-telegram/bot` |
| PGVector per-workspace collections | **one `chunks` table with `workspace_id` + pgvector column** (Go side skips LangChain's collection-per-workspace convention; a partial index per workspace gives the same isolation with simpler DDL) |
| HuggingFaceEmbeddings (bge-small) | fastembed-go (same model) |
| docling (in-process) | docling in a sidecar HTTP service (§5.4) |
| MarkdownHeaderTextSplitter | ~60-line header splitter in `internal/rag/chunk.go` (§9.2) |
| MemorySaver checkpoints | `chat_messages` table + bounded window (§9.4) |
| `user_role` claim in request body | real identity: Telegram user id or API key |

---

## 4. Repository layout (for the new Go folder)

```
tgrag/                       # your new Go module, e.g. github.com/<you>/tgrag
├── cmd/
│   ├── server/main.go       # wire everything, start HTTP + bot
│   └── seed/main.go         # seed workspaces + admin (§12)
├── internal/
│   ├── config/config.go     # env struct, parsed+validated once
│   ├── store/               # pgx pool, migrations runner, repositories
│   │   ├── store.go
│   │   ├── users.go
│   │   ├── workspaces.go
│   │   ├── documents.go
│   │   ├── chunks.go
│   │   └── messages.go
│   ├── auth/access.go       # access-control predicates (§8) — pure functions, table-tested
│   ├── httpapi/             # chi router, middleware, DTOs, handlers
│   ├── bot/                 # telegram update routing, handlers, keyboards
│   ├── ingest/pipeline.go   # download → parse → chunk → embed → store (§10)
│   └── rag/
│       ├── embed.go         # fastembed wrapper (iface Embedder for swappable impls)
│       ├── chunk.go         # markdown header splitter + fallback
│       ├── retrieve.go      # pgvector SQL + off-topic guard
│       ├── prompt.go        # prompt template + citation formatting
│       ├── groq.go          # Groq HTTP client (iface LLM for mocks)
│       └── answer.go        # orchestrates retrieve→prompt→groq (+ memory window)
├── parser-sidecar/          # tiny FastAPI + docling service (§5.4)
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── migrations/              # 0001_init.up.sql / .down.sql, ...
├── deployments/
│   ├── Dockerfile           # multi-stage Go build
│   └── docker-compose.yml   # db + parser + app
├── .env.example
└── README.md
```

Every unit has one purpose and a narrow interface (`Embedder`, `LLM`, `Parser`, repositories) — mock them in tests, swap implementations (local embeddings ↔ API) without touching callers.

---

## 5. External dependencies & contracts

### 5.1 Postgres (pgvector)

Image `pgvector/pgvector:pg16`. One database holds relational tables, the `chunks` vector table, and message history. Enable `CREATE EXTENSION IF NOT EXISTS vector;` in migration 0001.

### 5.2 Groq (chat LLM)

Plain HTTPS, one function:

```
POST https://api.groq.com/openai/v1/chat/completions
Authorization: Bearer $GROQ_API_KEY
{ "model": "openai/gpt-oss-20b", "temperature": 0, "messages": [ ...system+history+user ] }
→ choices[0].message.content
```

Timeout 60s, 3 retries with backoff on 429/5xx.

### 5.3 Embeddings

- **Primary:** fastembed-go, model `BAAI/bge-small-en-v1.5` → 384-dim vectors. Load at startup (fails fast), embed in batches of 32.
- **Fallback:** HF Inference API `feature-extraction` for `BAAI/bge-small-en-v1.5` (or an Ollama sidecar with `bge-m3`, 1024-dim — then change the column size in §6 and `EMBEDDING_DIM`).
- Whatever you pick, **normalize embeddings** (bge models expect it; cosine distance then equals 1 − dot product).

### 5.4 Parser sidecar (the honest call)

Go has no docling. PDF layout parsing in pure Go (ledongthuc/pdf, etc.) loses tables/structure; pptx has no real library; unidoc is commercial. Docling's parse quality is a core reason this RAG works. So: a ~30-line FastAPI service exposing one endpoint, used **only during ingestion**:

```
POST /parse      (multipart "file")
→ 200 { "markdown": "..." }      (422 on unsupported/corrupt file)
```

```python
# parser-sidecar/main.py — the whole service
import tempfile, docling.document_converter as dc
from fastapi import FastAPI, UploadFile, HTTPException

app = FastAPI()
@app.post("/parse")
async def parse(file: UploadFile):
    with tempfile.NamedTemporaryFile(suffix=file.filename and "_" + file.filename) as tmp:
        tmp.write(await file.read()); tmp.flush()
        try:
            return {"markdown": dc.DocumentConverter().convert(tmp.name).document.export_to_markdown()}
        except Exception as e:
            raise HTTPException(422, f"parse failed: {e}")
```

The Go side defines `type Parser interface { Parse(ctx, name string, data []byte) (string, error) }` with one implementation calling the sidecar. If you later want zero Python, swap the implementation — not the pipeline.

Supported formats (enforced in Go before calling the sidecar): `.pdf .docx .pptx .xlsx .html .md`. Max upload 20 MB (Telegram bot file limit).

---

## 6. Database schema (migration 0001)

All timestamps `timestamptz` (UTC). PKs UUIDv4 generated in Go.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    id                  UUID PRIMARY KEY,
    telegram_id         BIGINT UNIQUE,              -- NULL = service/API identity
    telegram_username   TEXT,
    display_name        TEXT NOT NULL,
    is_platform_admin   BOOLEAN NOT NULL DEFAULT FALSE,
    current_workspace_id UUID REFERENCES workspaces(id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE workspaces (
    id          UUID PRIMARY KEY,
    slug        TEXT UNIQUE NOT NULL,               -- lowercase kebab-case
    name        TEXT NOT NULL,
    description TEXT,
    is_private  BOOLEAN NOT NULL DEFAULT FALSE,
    owner_id    UUID NOT NULL REFERENCES users(id),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE workspace_members (
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN ('owner','editor','viewer')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, user_id)
);

CREATE TABLE documents (
    id           UUID PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    filename     TEXT NOT NULL,
    file_ext     TEXT NOT NULL,                     -- "pdf", no dot, lowercase
    byte_size    INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'processing'
                 CHECK (status IN ('processing','ready','failed')),
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    source       TEXT NOT NULL CHECK (source IN ('telegram','api','seed')),
    uploaded_by  UUID REFERENCES users(id),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id           BIGSERIAL PRIMARY KEY,
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_file  TEXT NOT NULL,
    section      TEXT,                              -- nearest "#"/"##" header, nullable
    content      TEXT NOT NULL,
    embedding    vector(384) NOT NULL
);
CREATE INDEX chunks_ws_embedding_idx ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_ws_idx ON chunks (workspace_id);

CREATE TABLE chat_messages (
    id           BIGSERIAL PRIMARY KEY,
    thread_key   TEXT NOT NULL,                     -- "tg:<telegram_id>:<slug>" | "api:<ref>:<slug>"
    role         TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content      TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX chat_messages_thread_idx ON chat_messages (thread_key, id);
```

Notes:
- `vector(384)` matches bge-small. If you choose a different embedding model, change the dimension here and in `EMBEDDING_DIM`.
- Workspace isolation in retrieval is `WHERE workspace_id = $1` (+ HNSW index), not one-collection-per-workspace — same isolation, one table, simpler SQL. (Difference from the Python spec is intentional.)
- `current_workspace_id` has a forward reference problem in DDL order — create `workspaces` before `users`, or add the column via `ALTER TABLE` at the end of the migration.

---

## 7. Configuration

Load once into a struct at startup; **fail fast** on anything missing/invalid.

| var | example | notes |
|---|---|---|
| `DATABASE_URL` | `postgres://rag:rag@localhost:5432/rag` | pgx format |
| `GROQ_API_KEY` | | |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | |
| `EMBEDDING_DIM` | `384` | must match schema |
| `TELEGRAM_BOT_TOKEN` | | from @BotFather |
| `TELEGRAM_ADMIN_ID` | `123456789` | this Telegram user becomes platform admin |
| `TELEGRAM_WEBHOOK_SECRET` | 32 random bytes | also set via setWebhook |
| `BOT_MODE` | `polling` \| `webhook` | default `polling` |
| `WEBHOOK_PUBLIC_URL` | `https://app.example.com` | required when `webhook` |
| `API_KEYS` | `key1,key2` | REST auth, comma-separated |
| `PARSER_URL` | `http://parser:8000` | sidecar base URL |
| `UPLOAD_DIR` | `data/uploads` | originals kept per workspace |
| `OFF_TOPIC_DISTANCE` | `0.95` | cosine-distance guard threshold |
| `HISTORY_WINDOW` | `8` | past messages fed to the LLM |
| `HTTP_ADDR` | `:8080` | |

Provide `.env.example` listing all of them; never commit `.env`.

---

## 8. Access control

Identity resolution:
- **Telegram:** `telegram_id` from the update → auto-provision user on first sight (open onboarding). `is_platform_admin = (telegram_id == TELEGRAM_ADMIN_ID)`.
- **REST:** `X-API-Key` matched with `subtle.ConstantTimeCompare` against `API_KEYS`. 401 JSON on miss.

Rules (implement as **pure functions** in `internal/auth/access.go`; table-test them):

| action | public workspace | private workspace |
|---|---|---|
| list / see workspace | any registered user | members only |
| chat / read | any registered user | members (any role) |
| ingest documents | any registered user | owner + editor |
| invite members | — | owner + editor |
| toggle public/private | owner | owner |
| platform admin | acts as owner everywhere | same |

Enforcement points: bot handlers check before acting; REST chat/upload endpoints take a `user_ref` and apply the same predicates. Never trust a client-claimed role — derive everything from identity + DB.

---

## 9. RAG query pipeline

### 9.1 Flow

```
message → identity → workspace (user's current_workspace_id) → access check
        → embed(query)
        → retrieve top-k chunks (workspace-filtered)
        → off-topic guard
        → prompt(system + history window + context + question)
        → Groq → reply (+ citations) → persist user+assistant messages
```

### 9.2 Chunking (at ingestion)

Split parsed markdown on `# ` / `## ` headers; attach the nearest header as `section`. Fallback for headerless output: fixed windows of ~1000 runes with 150 overlap (rune-safe iteration — Go strings are UTF-8; never slice mid-rune). Skip chunks < 40 runes.

### 9.3 Retrieval + guard

```sql
SELECT id, source_file, section, content,
       1 - (embedding <=> $1) AS similarity          -- cosine similarity
FROM chunks
WHERE workspace_id = $2
ORDER BY embedding <=> $1
LIMIT $3;                                            -- k = 4
```

- Distance `embedding <=> q` is cosine **distance** (0 = identical, 2 = opposite) for normalized vectors.
- **Guard:** if the best chunk's distance > `OFF_TOPIC_DISTANCE` (0.95 ≈ unrelated), reply "I can only answer questions about this workspace's documents." and skip the LLM call. (This mirrors the score>0.95 guard in the Python prototype.)

### 9.4 Prompt & memory

System prompt (keep the prototype's contract — answer only from context):

```
You answer questions using ONLY the context below. If the context does not
contain the answer, say "I don't know." Cite the source file names you used.

Context:
[1] policy.md — ...chunk text...
[2] zones.pdf — ...

Question: ...
```

History: load the last `HISTORY_WINDOW` (default 8) `chat_messages` for the thread, oldest first, insert between system prompt and question. Thread keys: `tg:<telegram_id>:<slug>` and `api:<user_ref>:<slug>`. Reply ends with `Sources: file1, file2` (unique, ordered by relevance).

### 9.5 Telegram reply mechanics

Split at 4096 chars (code-point safe) across up to a few messages. Use HTML parse mode and escape user/document names (`html.EscapeString`) — never MarkdownV2 (its escaping rules bite everyone). Every handler defers a recover() that logs the stack and sends "Something went wrong — please try again."

---

## 10. Ingestion pipeline

```
acquire bytes          Telegram getFile download | REST multipart
  → save original      UPLOAD_DIR/<slug>/<document_id>.<ext>
  → INSERT documents (status=processing)   ← caller gets id immediately
  → go runPipeline():                        (goroutine, per document)
       1. parse     → Parser sidecar → markdown
       2. chunk     → §9.2 splitter
       3. embed     → batched (32), normalized
       4. INSERT INTO chunks (per chunk, same tx: BEGIN; ...; UPDATE documents
          SET status='ready', chunk_count=N; COMMIT)
       on error: UPDATE documents SET status='failed', error=$msg
  → notify          Telegram uploader ("✅ N chunks from X are searchable" / failure)
```

- Guard against double-processing: only one goroutine per `document_id` (in-process map or `SELECT ... FOR UPDATE SKIP LOCKED` on the row).
- Vector insert retries: 3 attempts, linear backoff 2s→4s→8s (kept from the Python prototype).
- Scale-out path (README note, not v1): same `documents` row as the queue; N workers claim `processing` rows with `FOR UPDATE SKIP LOCKED`.

---

## 11. Interfaces

### 11.1 REST API (mounted under `/api/v1`, `X-API-Key` required unless noted)

Errors are always `{"error": {"code": "...", "message": "..."}}` with proper status (401/403/404/409/422/500). Request IDs in every response header (`X-Request-ID`) and log line.

| method & path | request | success response |
|---|---|---|
| `GET /health` (no auth) | — | `200 {"status":"ok","db":"ok"}` |
| `GET /workspaces` | — | `200 [{"slug","name","description","is_private","owner","document_count","chunk_count"}]` |
| `POST /workspaces` | `{"name","description"?,"is_private"?}` | `201 {"slug",...}` · 409 slug collision |
| `GET /workspaces/{slug}/documents` | — | `200 [{id,filename,status,chunk_count,error,source,created_at}]` |
| `POST /workspaces/{slug}/documents` | multipart `file`, optional `uploaded_by` | `202 {"document_id","status":"processing"}` |
| `POST /workspaces/{slug}/chat` | `{"message","user_ref"}` | `200 {"thread_key","answer","sources":["a.md"]}` · 403 if no access |
| `POST /telegram/webhook` | Telegram Update JSON | `200 {}` — reject unless `X-Telegram-Bot-Api-Secret-Token` matches |

Slug rules: lowercase `[a-z0-9-]`, 3–40 chars, derived from name, suffix `-2`, `-3`… on collision.

### 11.2 Telegram bot surface

| trigger | behavior |
|---|---|
| `/start` | auto-register; welcome; inline keyboard of accessible workspaces |
| `/help` | command list |
| `/workspaces` | re-list accessible workspaces as inline buttons (public ∪ private memberships) |
| `/newworkspace <name> [private]` | create; caller = owner |
| `/workspace` | current workspace + doc/chunk stats |
| `/invite @username` | owner/editor adds a *registered* user to the current private workspace as viewer |
| `/public`, `/private` | owner toggles privacy of current workspace |
| callback query (workspace button) | set current workspace; confirm + stats |
| document message | access-checked → "📥 Processing X…" → pipeline (§10) → ✅/❌ |
| plain text | access-checked → RAG answer (§9) with sources |

Access failures always explain: "This workspace is private — ask its owner for access" / "Pick a workspace first (/workspaces)".

---

## 12. Seed (`cmd/seed`)

Idempotent (skip anything whose slug exists). Creates: admin user from `TELEGRAM_ADMIN_ID`, three **public** workspaces from the prototype's corpus:

| slug | files (from `my_docs_folder/`) |
|---|---|
| `delivery-policy` | delivery_policy.md, delivery_zones_and_timeframes.pdf, order_tracking_and_status.pptx, public_user_delivery_terms.html, special_goods_delivery.html |
| `returns-and-refunds` | returns_and_refunds.md |
| `shipping-charges` | shipping_charges_matrix.docx, delivery_sla_matrix.xlsx |

Runs the normal ingestion pipeline (source=`seed`) and prints per-workspace chunk counts. Run once after migrations: `go run ./cmd/seed ./data/my_docs_folder` (corpus path as arg).

---

## 13. Deployment

### docker-compose.yml (three services)

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment: [POSTGRES_USER=rag, POSTGRES_PASSWORD=rag, POSTGRES_DB=rag]
    volumes: ["pgdata:/var/lib/postgresql/data"]
    healthcheck: {test: ["CMD-SHELL","pg_isready -U rag"], interval: 5s, retries: 10}
  parser:
    build: parser-sidecar          # docling; ~2 GB image, cold first parse ~seconds
    healthcheck: {test: ["CMD","python","-c","import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"], interval: 5s, retries: 10}
  app:
    build: {context: ., dockerfile: deployments/Dockerfile}
    depends_on: {db: {condition: service_healthy}, parser: {condition: service_healthy}}
    env_file: .env
    ports: ["8080:8080"]
volumes: {pgdata: {}}
```

### Dockerfile (multi-stage)

```
FROM golang:1.23 AS build
WORKDIR /src; COPY go.mod go.sum ./; RUN go mod download
COPY . .; RUN CGO_ENABLED=0 go build -o /tgrag ./cmd/server
FROM debian:bookworm-slim        # slim, not scratch: fastembed needs onnxruntime libs
COPY --from=build /tgrag /usr/local/bin/tgrag
USER 1000:1000
ENTRYPOINT ["tgrag"]
```

### Go live checklist (webhook mode)

1. Deploy with `BOT_MODE=webhook`, `WEBHOOK_PUBLIC_URL=https://your-host`.
2. `curl "https://api.telegram.org/bot<TOKEN>/setWebhook?url=https://your-host/telegram/webhook&secret_token=<TELEGRAM_WEBHOOK_SECRET>"`.
3. Verify `GET /health` → `{"db":"ok"}`; send `/start` to the bot; run seed if not yet.

---

## 14. Observability & error handling

- `slog` JSON to stdout: one line per HTTP request (method, path, status, ms, request-id) and per Telegram update (user, kind, workspace).
- Panic-recovery middleware → 500 JSON + logged stack; bot handlers → friendly user message.
- Ingestion failures live in `documents.error` — queryable (`GET /workspaces/{slug}/documents` shows `failed` + reason).
- Timeouts everywhere: HTTP server read/write 60s, Groq 60s, parser 120s (big PDFs), Postgres statement 30s.

---

## 15. Testing strategy

| layer | what | how |
|---|---|---|
| `auth` | full §8 matrix incl. platform-admin override | pure-function table tests |
| `rag/chunk` | header splits, headerless fallback, rune-safety, min-length skip | table tests |
| `rag/retrieve` | SQL + guard threshold | integration vs. test DB (docker pg), seeded vectors |
| `rag/answer` | prompt build, citations, history window | mock `LLM`, `Embedder` |
| `httpapi` | auth 401, 403 private workspace, 404 slug, 202 upload, chat shape | `httptest` + fake services |
| `bot` | command routing, workspace switch, upload ack, 4096 split | fabricated `models.Update` JSON, mocked sender |
| `ingest` | status transitions incl. failed-path | mock Parser failing |

Run integration tests against the compose db or a throwaway `pgvector` container; never prod data.

---

## 16. Build order (each step leaves a working system)

1. **Skeleton:** module, config, slog, chi, `/health`, migrations runner, compose db. → `docker compose up` healthy.
2. **Data layer:** migration 0001, repositories, users auto-provision, workspaces CRUD + §8 predicates with table tests.
3. **REST core:** api-key middleware; workspace list/create; document upload → 202 + `processing`; documents list.
4. **Ingestion:** parser sidecar; chunker; fastembed; chunks insert; status machine. Seed script here (§12) — proves the whole path with real files.
5. **RAG query:** retrieve + guard; Groq client; prompt + history; `/chat` endpoint.
6. **Telegram:** polling mode first (dev loop); commands, inline workspace picker, upload flow, text→RAG; reply splitting.
7. **Production mode:** webhook endpoint + secret validation; Dockerfiles; README runbook; live checklist (§13).

Phase 4 ending with the seed is deliberate: real documents flowing end-to-end before any bot code exists.

---

## 17. Carried-over decisions from the approved design (unchanged)

- Open onboarding; anyone on Telegram gets a profile; workspaces public by default, private optional.
- Ingestion via Telegram upload **and** REST; same pipeline.
- Three themed seed workspaces (§12).
- Off-topic guard with distance threshold 0.95; "I don't know" contract in the prompt.
- Per-(user, workspace) conversation threads; history survives restarts (now via `chat_messages`, not LangGraph checkpoints).
- API-key auth for REST; Telegram secret-token for the webhook; no client-claimed roles.
