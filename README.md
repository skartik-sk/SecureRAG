# Telegram Multi-Workspace RAG

A production-shaped RAG assistant you can talk to on **Telegram** (or over a small **REST API**). Ask questions about a workspace's documents and get answers with **inline citations**. Multiple workspaces (teams / topics) share one Postgres + pgvector backend, each fully isolated. Answers are produced by an **agentic LangGraph pipeline** running **hybrid retrieval** — Postgres full-text (BM25-style lexical) + pgvector dense search fused via **Reciprocal Rank Fusion**, diversified with **MMR**, then **LLM-graded reranking** — that retries with a rewritten query when retrieval comes back empty and refuses when the corpus can't answer.

## What it does

- **Workspaces** — 2..N document collections per deployment. Public or private; private ones gate access via membership (owner / editor / viewer).
- **Ingestion** — send a PDF, DOCX, PPTX, XLSX, HTML or MD file (≤ 20 MB) to the bot or `POST` it to the API. Files are parsed with lightweight parsers (pypdf / python-docx / python-pptx / openpyxl), split on markdown headers (fallback: recursive splitter), embedded via an **OpenAI-compatible embeddings API** (Together / Mistral / OpenAI — configurable) and stored per-workspace in **pgvector** collections (`ws_<slug>`).
- **Chat** — every question runs through the graph below. Answers cite `[1]`, `[2]`… mapped to source file + section.
- **History** — conversations live in Postgres; the LangGraph checkpointer (`PostgresSaver`) restores thread memory so `/resume` continues where you left off.
- **Demo mode** — `/demo` jumps to the seeded *Delivery Policy* workspace with tappable sample questions; `python3 scripts/seed.py` seeds three themed workspaces from `my_docs_folder/`.
- **Deploys to Vercel** — 100% API-driven (LLM + embeddings + managed Postgres), webhook-mode bot, lazy serverless startup. See [Deploy to Vercel](#deploy-to-vercel).

## Architecture

```
Telegram ⇄ python-telegram-bot (polling or webhook)      HTTP clients
        │                                                      │
        ▼                                                      ▼
   bot/handlers ──────────────┐                    FastAPI routers (/api/v1)
                              │                              │
                              ▼                              ▼
                    services (users, workspaces, ingest, chat)
                              │
                              ▼
      LangGraph:  START → guard ──refuse──► END
                              │ (distance ≤ threshold)
                              ▼
                          retrieve (k=8, pgvector)
                              ▼
                           grade (LLM JSON batch → rerank)
                              ▼
                 relevant? ──no + attempt 0──► rewrite ──► retrieve
                              │no + attempt 1          (one retry)
                              ▼yes
                          generate (top-4, [n] citations)
                              ▼
                             END
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
  PGVector (per-slug)   PostgresSaver (memory)   Groq LLM (gpt-oss-20b)
  SQLAlchemy models: users, workspaces, workspace_members, documents, conversations
```

## Setup

Prereqs: Python 3.12, Postgres 16 with pgvector.

```bash
# pgvector into Homebrew Postgres
brew install pgvector
psql -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"

# role + databases
psql -d postgres -c "CREATE ROLE langchain LOGIN PASSWORD 'langchain' SUPERUSER CREATEDB;"
psql -d postgres -c "CREATE DATABASE langchain OWNER langchain;"
psql -d langchain -c "CREATE EXTENSION IF NOT EXISTS vector;"

pip3 install -r requirements.txt
cp .env.example .env   # then edit: GROQ_API_KEY is the only required value
```

Key `.env` values: `DATABASE_URL`, `GROQ_API_KEY`, `EMBEDDINGS_API_KEY`, `GROQ_MODEL` (default `openai/gpt-oss-20b`),
`EMBEDDINGS_BASE_URL` / `EMBEDDING_MODEL` (default Together + `BAAI/bge-large-en-v1.5`),
`TELEGRAM_BOT_TOKEN`, `TELEGRAM_ADMIN_ID`, `BOT_MODE` (`disabled` | `polling` | `webhook`),
`WEBHOOK_URL`, `TELEGRAM_WEBHOOK_SECRET`, `API_KEYS` (comma-separated, protects the REST API).

## Seed the demo workspaces

```bash
python3 scripts/seed.py --corpus my_docs_folder
```

Creates `delivery-policy`, `returns-and-refunds`, `shipping-charges` (idempotent — re-runs skip what's already indexed).

## Run

```bash
# API only (no Telegram token needed) — the default
uvicorn app.main:app --host 0.0.0.0 --port 8080        # BOT_MODE=disabled

# Telegram long-polling (local dev): put the BotFather token in .env, then
BOT_MODE=polling uvicorn app.main:app --port 8080

# Production webhook: set WEBHOOK_URL=https://your.host and BOT_MODE=webhook;
# on boot the app calls setWebhook with your secret token.
```

Docker: `docker compose up --build` (app on :8080 + pgvector:pg16 db; mount `.env` and `my_docs_folder/`).

## REST API

All `/api/v1/*` routes need `X-API-Key: <one of API_KEYS>` (platform-level keys can see every workspace).

```bash
curl -H "X-API-Key: $KEY" localhost:8080/api/v1/workspaces
curl -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"name": "My Team", "is_private": true}' localhost:8080/api/v1/workspaces
curl -H "X-API-Key: $KEY" localhost:8080/api/v1/workspaces/delivery-policy/documents
curl -H "X-API-Key: $KEY" -F "file=@policy.pdf" \
     localhost:8080/api/v1/workspaces/delivery-policy/documents      # 202 Accepted, async ingest
curl -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"message": "What is the delivery SLA?"}' \
     localhost:8080/api/v1/workspaces/delivery-policy/chat
```

`GET /health` needs no auth and reports the bot mode.

## Telegram commands

| Command | What it does |
| --- | --- |
| `/start` | Register + pick a workspace (inline buttons) |
| `/workspaces` | List workspaces you can see; tap to switch |
| `/workspace` | Current workspace info (docs, chunks, privacy) |
| `/newworkspace <name> [private]` | Create a workspace; you become its owner |
| `/new` | New conversation; optionally paste content (120+ chars) to index |
| `/resume` | Your recent conversations; tap to continue one |
| `/demo` | Jump to the seeded policy workspace + sample questions |
| `/invite @user [editor\|viewer]` | Add a member to your private workspace (default viewer) |
| `/promote @user editor` | Change a member's role (owner only) |
| `/kick @user` | Remove a member (owner only) |
| `/public` `/private` | Toggle workspace privacy (owner only) |

Or just send a file to index it, or type a question to chat.

## How citations work

The graph retrieves the top-8 chunks by cosine distance, then asks the LLM to grade them in one batched JSON call ("which of these numbered chunks answer the question?") — chunks that survive are the rerank. The top-4 are passed to the generator, which must cite `[n]` inline or answer "I don't know." Cited numbers map back to `file` + `section` metadata, rendered as a **Sources:** block in Telegram and returned as `sources` in the API. If nothing is relevant, the graph rewrites the query once (resolving pronouns from history) and retries before refusing.

## Deploy to Vercel

Everything heavy is an API call (Groq LLM, embeddings, managed Postgres), so the whole app fits Vercel's serverless Python model. `api/index.py` + `vercel.json` route every URL (bot webhook + REST API) into the one FastAPI app; startup is lazy and idempotent because the platform doesn't reliably run lifespan events.

**Prerequisites** — three external services:

1. **Postgres with pgvector** — [Neon](https://neon.tech) (has pgvector; use the *pooled* connection string) or Supabase (enable the `vector` extension in the dashboard first).
2. **Groq** key for the chat LLM — <https://console.groq.com/keys>.
3. **Embeddings API** key — Together (<https://api.together.xyz/settings>) by default, or Mistral/OpenAI via `EMBEDDINGS_BASE_URL` + `EMBEDDING_MODEL`.

**Steps**

```bash
npm i -g vercel
vercel link
vercel env add DATABASE_URL      # e.g. postgresql+psycopg://user:pass@ep-...neon.tech/db?sslmode=require
vercel env add GROQ_API_KEY
vercel env add EMBEDDINGS_API_KEY
vercel env add TELEGRAM_BOT_TOKEN
vercel env add TELEGRAM_ADMIN_ID
vercel env add TELEGRAM_WEBHOOK_SECRET   # a long random string
vercel env add API_KEYS               # comma-separated keys for the REST API
vercel env add BOT_MODE               # webhook
vercel deploy --prod                  # note the https://<project>.vercel.app URL
vercel env add WEBHOOK_URL            # that same https://<project>.vercel.app URL
vercel deploy --prod                  # re-deploy once so setWebhook registers
```

Seed workspaces against the cloud DB from your machine with the same env values (`python3 scripts/seed.py` with a `.env` pointing at Neon) or via the REST API.

**Serverless behavior worth knowing**

- The Telegram bot must run in `BOT_MODE=webhook`; polling needs a persistent process and is for local dev only.
- Uploads through both the bot and the API are ingested **inline** (Vercel freezes the function after the response, so `INLINE_INGEST` is auto-enabled there); documents under a few MB finish well inside the 60 s Hobby function limit. Ingest is idempotent against Telegram webhook redeliveries (in-flight duplicate guard).
- Cold starts take a few seconds (imports + DB + webhook re-registration). Neon's pooler keeps the small per-instance pools from exhausting connections.
- `vercel.json` sets `maxDuration: 60` (Hobby plan). On Pro, raise it (up to 300) for bigger PDFs.

## Tests

```bash
python3 -m pytest tests/ -q     # needs local Postgres; rag_test DB is created automatically
```

## Troubleshooting

- **`extension "vector" is not available`** — install pgvector for your Postgres (`brew install pgvector`), then `CREATE EXTENSION vector;` in each database. On Supabase, enable it from the dashboard.
- **Chat endpoints 500 / bot doesn't answer** — `GROQ_API_KEY` missing; startup logs a warning and disables the graph.
- **`EMBEDDINGS_API_KEY is not set`** — ingest and chat need embeddings; set the key (and `EMBEDDINGS_BASE_URL` if not using Together).
- **Answers get worse after switching `EMBEDDING_MODEL`** — old chunks were embedded with the previous model; delete the workspace's `ws_<slug>` pgvector collection (or re-upload the files) so vectors and queries use the same model.
- **Bot silent after deploy** — `WEBHOOK_URL` must exactly match the deployment URL and `TELEGRAM_WEBHOOK_SECRET` must be set in both Telegram and Vercel; check `https://api.telegram.org/bot<token>/getWebhookInfo`.
