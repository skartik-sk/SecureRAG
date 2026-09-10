# Telegram Multi-Workspace RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the single-collection RAG into a multi-workspace, Telegram-first RAG assistant with persistent conversations (`/new`, `/resume`, `/demo`), cited answers, and an agentic LangGraph pipeline (retrieve → grade/rerank → rewrite-retry → generate).

**Architecture:** One FastAPI process. Thin `app/` package: config → SQLAlchemy models/services → routers (REST + Telegram webhook) → PTB bot handlers. RAG core is an injectable LangGraph state machine; per-workspace PGVector collections (`ws_<slug>`); `PostgresSaver` for conversation memory; a `conversations` table drives `/resume`.

**Tech Stack:** Python 3.12, FastAPI, LangChain 0.3 / LangGraph 0.2, `langgraph-checkpoint-postgres` (PostgresSaver), `python-telegram-bot` 21.x, SQLAlchemy 2 + psycopg 3, PGVector on Postgres 16, docling, HuggingFaceEmbeddings (`BAAI/bge-small-en-v1.5`), ChatGroq (`openai/gpt-oss-20b`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-telegram-multiworkspace-rag-design.md` — read it alongside this plan. Deltas added since the spec (user request, 2026-08-30), all covered by tasks here:

- **D1** `/resume` — list the user's past conversations as inline buttons; tapping one restores its thread (checkpointer gives back history).
- **D2** `/demo` — switch to the seeded policy workspace and offer tappable sample questions for instant testing.
- **D3** `/new` — new conversation; optional content step: user pastes text (chunked + embedded into the current workspace) or sends a file, then asks questions.
- **D4** Cited answers — generation prompt requires `[n]` inline citations mapped to a rendered `Sources:` block (REST + Telegram).
- **D5** Agentic graph replaces the single node: guard → retrieve (k=8) → grade (LLM filters/reranks) → (rewrite query once if nothing relevant) → generate (top-4).
- **D6** `BOT_MODE=disabled` — API-only mode so the app runs before a bot token exists; PTB starts only when `TELEGRAM_BOT_TOKEN` is set.

## Global Constraints

- **NO git commits or pushes. Ever.** The user explicitly forbade both. Task steps end with "run tests" checkpoints, not commit steps. Do not run `git add`/`git commit`/`git push` at any point.
- `rag_phase1.py` … `rag_phase4.py` are learning artifacts — never modify or delete them.
- Root `main.py` and `rag_engine.py` are deleted only in the final task, after the new system's tests pass.
- All timestamps default to UTC via `server_default=func.now()`.
- Primary keys are UUID strings (`String(36)`, `default=lambda: str(uuid.uuid4())`) except auto-increment `chunks` ids (managed by PGVector itself).
- Chat LLM is Groq `openai/gpt-oss-20b` (project convention; do not introduce another provider).
- Telegram replies use HTML parse mode (`html.escape` everything dynamic); never MarkdownV2.
- Telegram hard limit: 4096 chars per message — all bot replies go through `format.split_message`.
- Supported upload extensions: `.pdf .docx .pptx .xlsx .html .md`; max 20 MB.
- Off-topic distance threshold default `0.95` (cosine distance; 0 = identical).
- Environment (verified 2026-08-30): Python 3.12.0 system-wide (no venv — install with `pip3 install --user` or `pip3 install` as available), Homebrew Postgres 16 on localhost:5432 (superuser `singupallikartik`), Docker daemon not running (do not depend on it), `.env` currently contains only `GROQ_API_KEY`.
- Test DB is a dedicated `rag_test` database on the same server (created in Task 0); unit tests never need it.

---

## File Map (final state)

```
app/
├── __init__.py
├── config.py          # Settings (pydantic-settings), get_settings()
├── db.py              # engine, SessionLocal, init_db()
├── models.py          # User, Workspace, WorkspaceMember, Document, Conversation
├── schemas.py         # REST request/response DTOs
├── security.py        # role_of, can_view/can_ingest/can_manage, visible_workspaces, api_key_ok
├── main.py            # create_app(), lifespan, module-level `app`
├── routers/
│   ├── __init__.py
│   ├── health.py      # GET /health
│   ├── workspaces.py  # GET/POST /api/v1/workspaces
│   ├── documents.py   # GET/POST /api/v1/workspaces/{slug}/documents
│   ├── chat.py        # POST /api/v1/workspaces/{slug}/chat
│   └── telegram_webhook.py  # POST /telegram/webhook
├── services/
│   ├── __init__.py
│   ├── users.py       # get_or_create_user, ensure_system_user
│   ├── workspaces.py  # slugify, unique_slug, create_workspace, workspace_by_slug
│   ├── ingest.py      # parse_file, ingest_document_sync/async, ingest_text_sync
│   └── chat.py        # start_conversation, run_chat, list_conversations
├── bot/
│   ├── __init__.py
│   ├── setup.py       # build_application, start_polling, stop_application, Container
│   ├── handlers.py    # all commands/callbacks/document/text handlers
│   ├── keyboards.py   # inline keyboards, DEMO_QUESTIONS
│   └── format.py      # split_message, esc, format_answer
└── rag/
    ├── __init__.py
    ├── embeddings.py  # get_embeddings() singleton
    ├── llm.py         # get_llm() singleton
    ├── vectorstore.py # collection_name, get_store() per-slug cache
    ├── chunker.py     # chunk_markdown, chunk_plain_text
    ├── prompts.py     # GENERATE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT
    └── graph.py       # RAGState, build_graph(), get_graph() production singleton
seed.py                 # SEED_WORKSPACES, DEMO_SLUG, seed_all()
scripts/seed.py         # CLI wrapper
tests/
├── conftest.py
├── test_config.py
├── test_models_db.py
├── test_security.py
├── test_users_service.py
├── test_workspaces_service.py
├── test_chunker.py
├── test_vectorstore.py
├── test_ingest_service.py
├── test_graph.py
├── test_chat_service.py
├── test_routers.py
├── test_format.py
├── test_keyboards.py
└── test_bot_handlers.py
pyproject.toml          # pytest config (asyncio_mode=auto, markers)
requirements.txt        # extended
requirements-dev.txt    # pytest deps
.env.example
docker-compose.yml      # pgvector db for deployment
Dockerfile
README.md               # rewritten
```

Deleted at the end: `main.py`, `rag_engine.py` (logic superseded; recoverable from git history).

---

### Task 0: Environment setup (no TDD — verification steps only)

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`, `.env.example`, `.env` additions, `pyproject.toml`
- System: pgvector, `langchain` role + databases

**Interfaces:**
- Produces: importable `langgraph.checkpoint.postgres`, `telegram`, working `rag_test` database. Every later task assumes this.

- [ ] **Step 1: Install pgvector into Homebrew Postgres**

```bash
brew install pgvector
psql -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector;"   # smoke-check the .so is found
```

Expected: `CREATE EXTENSION`. If brew links a version mismatch, run `brew services restart postgresql@16` first.

- [ ] **Step 2: Create role and databases**

```bash
psql -d postgres -c "CREATE ROLE langchain LOGIN PASSWORD 'langchain' SUPERUSER CREATEDB;"  # ignore error if exists
psql -d postgres -c "CREATE DATABASE langchain OWNER langchain;"                            # ignore if exists
psql -d postgres -c "CREATE DATABASE rag_test OWNER langchain;"                             # ignore if exists
psql -d langchain  -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -d rag_test  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

(SUPERUSER locally so the role can create extensions; matches the docker-compose image where POSTGRES_USER is superuser.)

- [ ] **Step 3: Extend requirements.txt** — append:

```
python-telegram-bot==21.6
langgraph-checkpoint-postgres>=1.0.2,<2
python-multipart==0.0.9
```

Create `requirements-dev.txt`:

```
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
```

- [ ] **Step 4: Install and verify**

```bash
pip3 install -r requirements.txt -r requirements-dev.txt
python3 -c "import telegram, langgraph.checkpoint.postgres, multipart, httpx; print('ok')"
python3 -c "import pgvector; print('pgvector py ok')"   # if this fails: pip3 install pgvector
```

If pip cannot satisfy `langgraph-checkpoint-postgres<2` alongside `langgraph==0.2.20`, relax to `>=1.0.2,<3` and re-verify `PostgresSaver` imports — record the actually installed version in requirements.txt.

- [ ] **Step 5: Config files**

Append to `.env` (never print existing values):

```
DATABASE_URL=postgresql+psycopg://langchain:langchain@localhost:5432/langchain
TELEGRAM_BOT_TOKEN=
TELEGRAM_ADMIN_ID=0
TELEGRAM_WEBHOOK_SECRET=dev-secret-change-me
BOT_MODE=disabled
API_KEYS=dev-api-key-1
UPLOAD_DIR=data/uploads
```

Create `.env.example` with the same keys (empty values, comments explaining each — bot token from @BotFather; `BOT_MODE`: `disabled|polling|webhook`).

Create `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = ["integration: tests that need the local Postgres"]
```

- [ ] **Step 6: Verify**

```bash
python3 -m pytest --collect-only -q | tail -2   # "no tests ran" is fine; no import errors
```

---

### Task 1: `app/config.py` — typed settings

**Files:**
- Create: `app/__init__.py`, `app/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings` (fields below), `get_settings() -> Settings` (lru_cache), `Settings.database_url_plain -> str` (psycopg conninfo form: `postgresql://...` without `+psycopg`).

Fields: `database_url: str`, `groq_api_key: str = ""`, `groq_model: str = "openai/gpt-oss-20b"`, `embedding_model: str = "BAAI/bge-small-en-v1.5"`, `telegram_bot_token: str = ""`, `telegram_admin_id: int = 0`, `telegram_webhook_secret: str = "dev-secret-change-me"`, `bot_mode: str = "disabled"` (`disabled|polling|webhook`), `webhook_url: str = ""`, `api_keys: str = ""`, `off_topic_distance: float = 0.95`, `history_window: int = 8`, `upload_dir: str = "data/uploads"`, `retrieve_k: int = 8`, `top_n: int = 4`. Property `api_key_set -> frozenset[str]` (split on comma, strip, drop empties).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
from app.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.bot_mode == "disabled"
    assert s.groq_model == "openai/gpt-oss-20b"
    assert s.off_topic_distance == 0.95
    assert s.retrieve_k == 8 and s.top_n == 4


def test_api_key_set_splits_and_strips():
    s = Settings(_env_file=None, api_keys=" a , b ,, ")
    assert s.api_key_set == frozenset({"a", "b"})


def test_database_url_plain_strips_dialect():
    s = Settings(_env_file=None, database_url="postgresql+psycopg://u:p@h:5432/db")
    assert s.database_url_plain == "postgresql://u:p@h:5432/db"


def test_env_vars_are_read(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:x")
    s = Settings(_env_file=None)
    assert s.telegram_bot_token == "123:x"
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: Implement**

```python
# app/config.py
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    telegram_bot_token: str = ""
    telegram_admin_id: int = 0
    telegram_webhook_secret: str = "dev-secret-change-me"
    bot_mode: str = "disabled"  # disabled | polling | webhook
    webhook_url: str = ""

    api_keys: str = ""
    off_topic_distance: float = 0.95
    history_window: int = 8
    upload_dir: str = "data/uploads"
    retrieve_k: int = 8
    top_n: int = 4

    @property
    def api_key_set(self) -> frozenset:
        return frozenset(k.strip() for k in self.api_keys.split(",") if k.strip())

    @property
    def database_url_plain(self) -> str:
        return self.database_url.replace("+psycopg", "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

`app/__init__.py` is empty.

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: 4 PASS

---

### Task 2: `app/models.py` + `app/db.py` — schema

**Files:**
- Create: `app/db.py`, `app/models.py`
- Test: `tests/test_models_db.py`
- Reference: `conftest.py` created here too (later tasks depend on its fixtures)

**Interfaces:**
- Produces: `Base` (DeclarativeBase), models `User`, `Workspace`, `WorkspaceMember`, `Document`, `Conversation`; `db.engine(settings)`, `db.SessionLocal(settings)`, `db.init_db(settings)` (create_all + best-effort `CREATE EXTENSION IF NOT EXISTS vector`).
- `conftest.py` produces fixtures: `test_settings` (db → `rag_test`), `db` (session-scoped: create_all + truncate between tests), `session` (function-scoped Session).

Schema (all FKs with `ondelete="CASCADE"` except `users.current_workspace_id`/`current_conversation_id` → `SET NULL`; datetime columns typed `Mapped[datetime]` / `Mapped[datetime | None]` with `from datetime import datetime`):

- `users`: id pk, telegram_id BigInteger unique nullable index, telegram_username String(64) nullable, display_name String(128) default "", is_platform_admin bool default False, current_workspace_id FK workspaces.id nullable, current_conversation_id FK conversations.id nullable (use `use_alter=True`), pending_action String(32) nullable (None | `"awaiting_content"`), created_at, updated_at (onupdate).
- `workspaces`: id pk, slug String(64) unique not null, name String(128) not null, description Text nullable, is_private bool default False, owner_id FK users.id not null, created_at.
- `workspace_members`: workspace_id FK pk, user_id FK pk, role String(16) not null (`owner|editor|viewer`), created_at.
- `documents`: id pk, workspace_id FK not null, filename String(256) not null, file_ext String(16) not null, byte_size Integer default 0, status String(16) default `"processing"` (`processing|ready|failed`), chunk_count Integer default 0, error Text nullable, source String(16) not null (`telegram|api|seed|paste`), uploaded_by FK users.id nullable, created_at, updated_at.
- `conversations`: id pk, user_id FK not null, workspace_id FK not null, title String(256) default `"New chat"`, is_active bool default True, created_at, last_message_at DateTime nullable.

- [ ] **Step 1: Write conftest + failing test**

```python
# tests/conftest.py
import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import engine, init_db
from app.models import Base

TEST_DB_URL = "postgresql+psycopg://langchain:langchain@localhost:5432/rag_test"


def _make_settings() -> Settings:
    return Settings(_env_file=None, database_url=TEST_DB_URL, upload_dir="data/test_uploads")


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    s = _make_settings()
    try:
        engine(s).connect()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"integration Postgres unavailable: {e}")
    return s


@pytest.fixture(scope="session")
def db(test_settings):
    init_db(test_settings)
    yield
    with engine(test_settings).begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f'DELETE FROM "{table.name}"'))


TABLES = ["documents", "conversations", "workspace_members", "workspaces", "users"]  # FK-safe order


@pytest.fixture
def session(test_settings, db) -> Session:
    s = Session(engine(test_settings))
    yield s
    s.rollback()
    s.close()
    with engine(test_settings).begin() as conn:
        for t in TABLES:
            conn.execute(text(f'DELETE FROM "{t}"'))
```

```python
# tests/test_models_db.py
from sqlalchemy import select

from app.db import init_db
from app.models import Conversation, Document, User, Workspace, WorkspaceMember


def test_init_db_creates_all_tables(test_settings):
    init_db(test_settings)  # idempotent
    from app.db import engine

    with engine(test_settings).connect() as conn:
        names = {r[0] for r in conn.exec_driver_sql(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'")}
    assert {"users", "workspaces", "workspace_members", "documents", "conversations"} <= names


def test_user_roundtrip(session):
    u = User(telegram_id=42, display_name="Kartik")
    session.add(u)
    session.flush()
    got = session.scalar(select(User).where(User.telegram_id == 42))
    assert got is not None and got.id and got.is_platform_admin is False
    assert got.pending_action is None


def test_workspace_member_composite_pk(session):
    u = User(telegram_id=1, display_name="a")
    session.add(u)
    session.flush()
    w = Workspace(slug="s-1", name="S", owner_id=u.id)
    session.add(w)
    session.flush()
    session.add(WorkspaceMember(workspace_id=w.id, user_id=u.id, role="owner"))
    session.flush()  # duplicate insert would raise — composite PK enforced
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_models_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.db'` (integration tests skip cleanly if Postgres down — but it's up)

- [ ] **Step 3: Implement `app/db.py`**

```python
# app/db.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models import Base

_ENGINES: dict[str, object] = {}


def engine(settings: Settings):
    key = settings.database_url
    if key not in _ENGINES:
        _ENGINES[key] = create_engine(settings.database_url, pool_pre_ping=True)
    return _ENGINES[key]


def SessionLocal(settings: Settings):
    return sessionmaker(bind=engine(settings), expire_on_commit=False)


def init_db(settings: Settings) -> None:
    eng = engine(settings)
    try:
        with eng.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception:
        pass  # pre-created by setup; PGVector needs it before first collection use
    Base.metadata.create_all(eng)
```

- [ ] **Step 4: Implement `app/models.py`**

```python
# app/models.py
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger, Boolean, DateTime, ForeignKey, Integer, String, Text, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    telegram_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True, nullable=True)
    telegram_username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    display_name: Mapped[str] = mapped_column(String(128), default="")
    is_platform_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    current_workspace_id: Mapped[str | None] = mapped_column(
        ForeignKey("workspaces.id", ondelete="SET NULL", use_alter=True), nullable=True)
    current_conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL", use_alter=True), nullable=True)
    pending_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_private: Mapped[bool] = mapped_column(Boolean, default=False)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class WorkspaceMember(Base):
    __tablename__ = "workspace_members"

    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[str] = mapped_column(String(16))  # owner | editor | viewer
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(256))
    file_ext: Mapped[str] = mapped_column(String(16))
    byte_size: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16), default="processing")  # processing|ready|failed
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(16))  # telegram | api | seed | paste
    uploaded_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(256), default="New chat")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

- [ ] **Step 5: Run to verify pass**

Run: `python3 -m pytest tests/test_models_db.py tests/test_config.py -v`
Expected: PASS (models) — note `use_alter=True` avoids the forward-reference ordering problem flagged in the spec.

---

### Task 3: `app/security.py` — access rules (pure functions)

**Files:**
- Create: `app/security.py`
- Test: `tests/test_security.py`

**Interfaces:**
- Consumes: models from Task 2.
- Produces: `role_of(session, workspace, user) -> str | None` (`"owner"|"editor"|"viewer"|None`; platform admin → `"owner"`; owner_id match → `"owner"`), `can_view(ws, role) -> bool`, `can_ingest(ws, role) -> bool`, `can_manage(ws, role) -> bool`, `visible_workspaces(session, user) -> list[Workspace]`, `api_key_ok(settings, presented: str | None) -> bool`, `api_key_dependency(request) -> None` (FastAPI dependency raising 401; shared by all three protected routers in Task 11).

Rules: public workspace → everyone may view/chat; private → members only. Ingest on public → any registered user; on private → owner/editor only. Manage (invite/privacy/delete) → owner only. `api_key_ok` uses `secrets.compare_digest`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_security.py
from app.models import User, Workspace
from app.security import (
    api_key_ok, can_ingest, can_manage, can_view, role_of, visible_workspaces,
)
from app.config import Settings


def _ws(private=False, owner_id="owner-1"):
    return Workspace(slug="s", name="S", is_private=private, owner_id=owner_id)


def _user(admin=False, uid="u-1"):
    return User(id=uid, display_name="u", is_platform_admin=admin)


def test_public_viewable_by_anyone():
    assert can_view(_ws(), role_of(None, _ws(), _user()))


def test_private_hidden_from_non_member():
    assert not can_view(_ws(private=True), role_of(None, _ws(private=True), _user()))


def test_owner_role_via_owner_id():
    assert role_of(None, _ws(), _user(uid="owner-1")) == "owner"


def test_platform_admin_is_owner_everywhere():
    ws = _ws(private=True)
    assert role_of(None, ws, _user(admin=True)) == "owner"


def test_member_roles(session, test_settings):
    from app.models import WorkspaceMember

    u, owner = _user(uid="m-1"), _user(uid="owner-1")
    w = _ws(private=True)
    session.add_all([u, owner, w])
    session.flush()
    session.add(WorkspaceMember(workspace_id=w.id, user_id=u.id, role="editor"))
    session.flush()
    assert role_of(session, w, u) == "editor"
    assert can_ingest(w, "editor")
    assert not can_manage(w, "editor")


def test_ingest_matrix():
    assert can_ingest(_ws(), "viewer")            # public: anyone
    assert can_ingest(_ws(private=True), "editor")
    assert not can_ingest(_ws(private=True), "viewer")
    assert not can_ingest(_ws(private=True), None)


def test_manage_owner_only():
    assert can_manage(_ws(), "owner")
    assert not can_manage(_ws(), "editor")
    assert not can_manage(_ws(), None)


def test_visible_workspaces(session, test_settings):
    from app.models import WorkspaceMember

    u, other = _user(uid="vis-1"), _user(uid="vis-2")
    pub, priv_mine, priv_other = _ws(owner_id="vis-1"), _ws(private=True), _ws(private=True)
    priv_mine.slug, priv_other.slug, pub.slug = "pm", "po", "pub"
    session.add_all([u, other, pub, priv_mine, priv_other])
    session.flush()
    session.add(WorkspaceMember(workspace_id=priv_mine.id, user_id=u.id, role="viewer"))
    session.flush()
    seen = {w.slug for w in visible_workspaces(session, u)}
    assert seen == {"pub", "pm"}


def test_api_key_ok():
    s = Settings(_env_file=None, api_keys="k1, k2")
    assert api_key_ok(s, "k1") and api_key_ok(s, "k2")
    assert not api_key_ok(s, "nope") and not api_key_ok(s, None) and not api_key_ok(s, "")
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_security.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.security'`

- [ ] **Step 3: Implement**

```python
# app/security.py
import secrets

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import User, Workspace, WorkspaceMember

MANAGE_ROLES = ("owner",)
INGEST_ROLES = ("owner", "editor")


def role_of(session: Session | None, workspace: Workspace, user: User) -> str | None:
    if user.is_platform_admin or workspace.owner_id == user.id:
        return "owner"
    if session is None:
        return None
    member = session.scalar(select(WorkspaceMember).where(
        WorkspaceMember.workspace_id == workspace.id, WorkspaceMember.user_id == user.id))
    return member.role if member else None


def can_view(workspace: Workspace, role: str | None) -> bool:
    return not workspace.is_private or role is not None


def can_ingest(workspace: Workspace, role: str | None) -> bool:
    return not workspace.is_private or role in INGEST_ROLES


def can_manage(workspace: Workspace, role: str | None) -> bool:
    return role in MANAGE_ROLES


def visible_workspaces(session: Session, user: User) -> list[Workspace]:
    member_ws_ids = select(WorkspaceMember.workspace_id).where(WorkspaceMember.user_id == user.id)
    return list(session.scalars(select(Workspace).where(
        or_(Workspace.is_private.is_(False), Workspace.id.in_(member_ws_ids))
    ).order_by(Workspace.created_at)))


def api_key_ok(settings: Settings, presented: str | None) -> bool:
    if not presented:
        return False
    return any(secrets.compare_digest(presented, k) for k in settings.api_key_set)


def api_key_dependency(request: "Request") -> None:
    from fastapi import HTTPException

    settings = request.app.state.settings
    if not api_key_ok(settings, request.headers.get("X-API-Key")):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

(The `"Request"` type is `fastapi.Request` — import it at the top of the module in the real file: `from fastapi import Request`.)

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_security.py -v`
Expected: 10 PASS

---

### Task 4: `app/services/users.py`

**Files:**
- Create: `app/services/__init__.py`, `app/services/users.py`
- Test: `tests/test_users_service.py`

**Interfaces:**
- Consumes: `User` model, `Settings`.
- Produces: `get_or_create_user(session, settings, telegram_id, username=None, display_name=None) -> User` (creates with `is_platform_admin = telegram_id == settings.telegram_admin_id`; updates username/display_name on later logins; returns existing row otherwise), `ensure_system_user(session) -> User` (telegram_id NULL, display_name `"platform"` — owner of API-created workspaces; exactly one such row).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_users_service.py
from sqlalchemy import select

from app.config import Settings
from app.models import User
from app.services.users import ensure_system_user, get_or_create_user


def _s(admin_id=0):
    return Settings(_env_file=None, telegram_admin_id=admin_id)


def test_creates_user(session):
    u = get_or_create_user(session, _s(), telegram_id=777, username="k", display_name="Kartik")
    session.flush()
    assert u.id and u.telegram_id == 777 and u.telegram_username == "k"
    assert u.is_platform_admin is False


def test_second_contact_returns_same_row(session):
    a = get_or_create_user(session, _s(), telegram_id=777, username="k", display_name="Kartik")
    b = get_or_create_user(session, _s(), telegram_id=777, username="k2", display_name="K Tikk")
    assert a.id == b.id and b.telegram_username == "k2" and b.display_name == "K Tikk"


def test_admin_flag_from_settings(session):
    u = get_or_create_user(session, _s(admin_id=42), telegram_id=42)
    assert u.is_platform_admin is True


def test_ensure_system_user_idempotent(session):
    a = ensure_system_user(session)
    b = ensure_system_user(session)
    assert a.id == b.id and a.telegram_id is None
    assert session.scalar(select(User).where(User.telegram_id.is_(None))).id == a.id
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_users_service.py -v`
Expected: FAIL — `No module named 'app.services'`

- [ ] **Step 3: Implement**

```python
# app/services/users.py
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import User

SYSTEM_DISPLAY_NAME = "platform"


def get_or_create_user(
    session: Session,
    settings: Settings,
    telegram_id: int,
    username: str | None = None,
    display_name: str | None = None,
) -> User:
    user = session.scalar(select(User).where(User.telegram_id == telegram_id))
    if user is None:
        user = User(
            telegram_id=telegram_id,
            telegram_username=username,
            display_name=display_name or f"User {telegram_id}",
            is_platform_admin=(telegram_id == settings.telegram_admin_id),
        )
        session.add(user)
        session.flush()
        return user
    if username:
        user.telegram_username = username
    if display_name:
        user.display_name = display_name
    session.flush()
    return user


def ensure_system_user(session: Session) -> User:
    user = session.scalar(select(User).where(User.telegram_id.is_(None)))
    if user is None:
        user = User(display_name=SYSTEM_DISPLAY_NAME)
        session.add(user)
        session.flush()
    return user
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_users_service.py -v`
Expected: 4 PASS

---

### Task 5: `app/services/workspaces.py`

**Files:**
- Create: `app/services/workspaces.py`
- Test: `tests/test_workspaces_service.py`

**Interfaces:**
- Consumes: `Workspace`, `WorkspaceMember`, `services.users.ensure_system_user`, `security.role_of`.
- Produces: `slugify(name) -> str` (lowercase `[a-z0-9-]`, collapse runs to `-`, strip `-`, max 40, empty → `"workspace"`), `unique_slug(session, base) -> str` (append `-2`, `-3`… on collision), `create_workspace(session, owner, name, description=None, is_private=False) -> Workspace` (adds owner member row), `workspace_by_slug(session, slug) -> Workspace | None`, `workspace_stats(session, workspace) -> dict` (`{"documents": int_ready, "chunks": int_sum}`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_workspaces_service.py
from app.services.users import ensure_system_user
from app.services.workspaces import (
    create_workspace, slugify, unique_slug, workspace_by_slug, workspace_stats,
)


def test_slugify():
    assert slugify("Delivery Policy!") == "delivery-policy"
    assert slugify("  A  B   C ") == "a-b-c"
    assert slugify("héllo *** world") == "hllo-world"
    assert slugify("") == "workspace"
    assert slugify("x" * 100) == "x" * 40


def test_unique_slug(session, test_settings):
    owner = ensure_system_user(session)
    create_workspace(session, owner, "Same Name")
    w2 = create_workspace(session, owner, "Same Name")
    assert w2.slug == "same-name-2"
    w3 = create_workspace(session, owner, "Same Name")
    assert w3.slug == "same-name-3"


def test_create_workspace_adds_owner_member(session, test_settings):
    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Private Stuff", description="d", is_private=True)
    from app.models import WorkspaceMember

    m = session.get(WorkspaceMember, (w.id, owner.id))
    assert m.role == "owner" and w.is_private and w.slug == "private-stuff"


def test_workspace_by_slug(session, test_settings):
    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Findable")
    assert workspace_by_slug(session, "findable").id == w.id
    assert workspace_by_slug(session, "missing") is None


def test_workspace_stats(session, test_settings):
    from app.models import Document

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Stats")
    session.add_all([
        Document(workspace_id=w.id, filename="a.md", file_ext="md", status="ready",
                 chunk_count=3, source="seed"),
        Document(workspace_id=w.id, filename="b.md", file_ext="md", status="failed",
                 chunk_count=0, source="seed"),
    ])
    session.flush()
    assert workspace_stats(session, w) == {"documents": 1, "chunks": 3}
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_workspaces_service.py -v`
Expected: FAIL — `No module named 'app.services.workspaces'`

- [ ] **Step 3: Implement**

```python
# app/services/workspaces.py
import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Document, User, Workspace, WorkspaceMember


def slugify(name: str) -> str:
    ascii_only = "".join(c for c in name.lower() if c.isascii())
    s = re.sub(r"[^a-z0-9]+", "-", ascii_only).strip("-")
    return (s or "workspace")[:40].strip("-") or "workspace"


def unique_slug(session: Session, base: str) -> str:
    slug, n = base, 1
    while session.scalar(select(Workspace).where(Workspace.slug == slug)) is not None:
        n += 1
        slug = f"{base}-{n}"
    return slug


def create_workspace(
    session: Session,
    owner: User,
    name: str,
    description: str | None = None,
    is_private: bool = False,
) -> Workspace:
    ws = Workspace(
        slug=unique_slug(session, slugify(name)),
        name=name, description=description, is_private=is_private, owner_id=owner.id,
    )
    session.add(ws)
    session.flush()
    session.add(WorkspaceMember(workspace_id=ws.id, user_id=owner.id, role="owner"))
    session.flush()
    return ws


def workspace_by_slug(session: Session, slug: str) -> Workspace | None:
    return session.scalar(select(Workspace).where(Workspace.slug == slug))


def workspace_stats(session: Session, workspace: Workspace) -> dict:
    rows = session.execute(
        select(
            Document.status,
            func.count(Document.id),
            func.coalesce(func.sum(Document.chunk_count), 0),
        )
        .where(Document.workspace_id == workspace.id)
        .group_by(Document.status)
    ).all()
    ready_docs = sum(int(count) for status, count, _ in rows if status == "ready")
    total_chunks = sum(int(chunks) for _, _, chunks in rows)
    return {"documents": ready_docs, "chunks": total_chunks}
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_workspaces_service.py -v`
Expected: 5 PASS

Note: `slugify("héllo")` → `"hllo"` because `é` is stripped, not transliterated — the test asserts this exact behavior.

---

### Task 6: `app/rag/chunker.py`

**Files:**
- Create: `app/rag/__init__.py`, `app/rag/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Produces: `MIN_CHUNK_CHARS = 40`, `chunk_markdown(md: str) -> list[Document]` (header split on `#`/`##`; each chunk's metadata: `{"section": <nearest header text or None>}`; chunks < 40 chars dropped; chunks > 1200 chars further split), `chunk_plain_text(text: str) -> list[Document]` (RecursiveCharacterTextSplitter 1000/150, `section: None`).
- Consumes: `langchain_text_splitters.MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter`; `langchain_core.documents.Document`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunker.py
from app.rag.chunker import MIN_CHUNK_CHARS, chunk_markdown, chunk_plain_text

MD = """# Delivery

Main policy text that is definitely long enough to survive the minimum filter.

## Zones

Zone details that are also long enough to survive the minimum length filter here.

## Tiny

no

# Refunds

Refund policy body text, again comfortably above the cutoff for inclusion.
"""


def test_markdown_header_split_with_sections():
    chunks = chunk_markdown(MD)
    sections = [c.metadata["section"] for c in chunks]
    assert sections[0] == "Delivery"
    assert "Zones" in sections
    assert "no" not in [c.page_content for c in chunks]  # tiny chunk dropped


def test_small_header_content_merged_into_oversize_rule():
    chunks = chunk_markdown(MD)
    texts = [c.page_content for c in chunks]
    assert all(len(c.strip()) >= MIN_CHUNK_CHARS for c in texts)


def test_headerless_markdown_falls_back():
    md = "word " * 500  # no headers at all
    chunks = chunk_markdown(md)
    assert len(chunks) >= 2  # 2500 chars → split into >1 pieces
    assert all(c.metadata["section"] is None for c in chunks)


def test_plain_text_chunks():
    chunks = chunk_plain_text("hello world. " * 400)
    assert len(chunks) >= 2
    assert all(c.metadata["section"] is None for c in chunks)


def test_empty_and_tiny_inputs():
    assert chunk_markdown("") == []
    assert chunk_markdown("short") == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_chunker.py -v`
Expected: FAIL — `No module named 'app.rag'`

- [ ] **Step 3: Implement**

```python
# app/rag/chunker.py
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

MIN_CHUNK_CHARS = 40
OVERSIZE_CHARS = 1200
_HEADERS = [("#", "h1"), ("##", "h2")]
_FALLBACK = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _section(md_meta: dict) -> str | None:
    return md_meta.get("h2") or md_meta.get("h1")


def _clean(docs: list[Document]) -> list[Document]:
    return [
        Document(page_content=d.page_content, metadata={"section": _section(d.metadata)})
        for d in docs
        if len(d.page_content.strip()) >= MIN_CHUNK_CHARS
    ]


def chunk_markdown(md: str) -> list[Document]:
    if not md or not md.strip():
        return []
    docs = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False).split_text(md)
    if not docs:
        docs = _FALLBACK.create_documents([md])
    else:
        oversized = [d for d in docs if len(d.page_content) > OVERSIZE_CHARS]
        for d in oversized:
            docs.remove(d)
        for d in oversized:
            docs.extend(_FALLBACK.split_documents([d]))
    return _clean(docs)


def chunk_plain_text(text: str) -> list[Document]:
    if not text or not text.strip():
        return []
    docs = _FALLBACK.create_documents([text])
    return [
        Document(page_content=d.page_content, metadata={"section": None})
        for d in docs
        if len(d.page_content.strip()) >= MIN_CHUNK_CHARS
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_chunker.py -v`
Expected: 5 PASS

---

### Task 7: `app/rag/embeddings.py`, `llm.py`, `vectorstore.py`

**Files:**
- Create: `app/rag/embeddings.py`, `app/rag/llm.py`, `app/rag/vectorstore.py`
- Test: `tests/test_vectorstore.py`

**Interfaces:**
- Produces: `embeddings.get_embeddings(settings=None)` (singleton `HuggingFaceEmbeddings`), `llm.get_llm(settings=None)` (singleton `ChatGroq`, temperature 0), `vectorstore.collection_name(slug) -> "ws_<slug>"`, `vectorstore.get_store(settings, slug, embeddings=None) -> PGVector` (per-slug cache; `use_jsonb=True`), `vectorstore.reset_cache()`.

These singletons are lazily created so importing the module in tests never loads bge-small or touches Groq.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vectorstore.py
from app.rag.vectorstore import collection_name


def test_collection_name():
    assert collection_name("delivery-policy") == "ws_delivery-policy"
    assert collection_name("returns") == "ws_returns"


def test_get_store_is_cached_per_slug(test_settings):
    from app.rag import vectorstore as vs

    calls = []

    class FakeEmbeddings:
        pass

    def fake_factory(**kwargs):
        calls.append(kwargs["collection_name"])
        return object()

    import app.rag.vectorstore as mod

    original = mod._new_store
    mod._new_store = fake_factory
    try:
        a = mod.get_store(test_settings, "s1", embeddings=FakeEmbeddings())
        b = mod.get_store(test_settings, "s1", embeddings=FakeEmbeddings())
        c = mod.get_store(test_settings, "s2", embeddings=FakeEmbeddings())
        assert a is b and a is not c
        assert calls == ["ws_s1", "ws_s2"]
    finally:
        mod._new_store = original
        mod.reset_cache()
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_vectorstore.py -v`
Expected: FAIL — `No module named 'app.rag.vectorstore'`

- [ ] **Step 3: Implement the three modules**

```python
# app/rag/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Settings, get_settings

_instance = None


def get_embeddings(settings: Settings | None = None) -> HuggingFaceEmbeddings:
    global _instance
    if _instance is None:
        s = settings or get_settings()
        _instance = HuggingFaceEmbeddings(model_name=s.embedding_model)
    return _instance
```

```python
# app/rag/llm.py
from langchain_groq import ChatGroq

from app.config import Settings, get_settings

_instance = None


def get_llm(settings: Settings | None = None) -> ChatGroq:
    global _instance
    if _instance is None:
        s = settings or get_settings()
        _instance = ChatGroq(model=s.groq_model, temperature=0, api_key=s.groq_api_key or None)
    return _instance
```

```python
# app/rag/vectorstore.py
from langchain_postgres.vectorstores import PGVector

from app.config import Settings


def collection_name(slug: str) -> str:
    return f"ws_{slug}"


_cache: dict[str, PGVector] = {}


def _new_store(**kwargs) -> PGVector:
    return PGVector(**kwargs)


def get_store(settings: Settings, slug: str, embeddings=None) -> PGVector:
    if slug not in _cache:
        from app.rag.embeddings import get_embeddings

        _cache[slug] = _new_store(
            embeddings=embeddings or get_embeddings(settings),
            collection_name=collection_name(slug),
            connection=settings.database_url,
            use_jsonb=True,
        )
    return _cache[slug]


def reset_cache() -> None:
    _cache.clear()
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_vectorstore.py tests/test_chunker.py -v`
Expected: 7 PASS

---

### Task 8: `app/services/ingest.py`

**Files:**
- Create: `app/services/ingest.py`
- Test: `tests/test_ingest_service.py`

**Interfaces:**
- Consumes: `Document` model, `chunker`, a store object duck-typed as `add_documents(docs) -> None`.
- Produces:
  - `SUPPORTED_EXTS: frozenset[str]` = `{pdf, docx, pptx, xlsx, html, md}`
  - `parse_file(data: bytes, ext: str) -> list[Document]` — `md` decoded as UTF-8 → `chunk_markdown`; everything else → docling in a temp file → markdown → `chunk_markdown`. Raises `ValueError` on unsupported ext, `RuntimeError` on parse failure.
  - `create_document_row(session, workspace, filename, source, uploaded_by=None, byte_size=0) -> Document` (status `processing`; ext derived from filename).
  - `ingest_document_sync(session_factory, document_id, data, ext, store, settings=None) -> int` — parse → `store.add_documents` (3 retries, 2s/4s/8s backoff) → row `ready` + `chunk_count`; on failure row `failed` + `error`. Uses its own session per update. Returns chunk count (raises after final retry failure).
  - `ingest_text_sync(session_factory, workspace, title, text, store, source="paste", uploaded_by=None) -> Document` — one row, `chunk_plain_text`, add to store, `ready` immediately. Returns the row.
  - `save_original(settings, slug, document_id, ext, data) -> Path` (mkdir -p `UPLOAD_DIR/<slug>/`).

Markdown files never touch docling (fast path, and makes seeding `.md` testable without the heavy converter).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ingest_service.py
import pytest

from app.models import Document
from app.services.ingest import (
    SUPPORTED_EXTS, create_document_row, ingest_document_sync,
    ingest_text_sync, parse_file,
)

MD_DOC = b"# Title\n\nBody text that is long enough to pass the chunk minimum length filter.\n"


class FakeStore:
    def __init__(self):
        self.added = []

    def add_documents(self, docs):
        self.added.extend(docs)


class FailingStore(FakeStore):
    def __init__(self, fail_times=1):
        super().__init__()
        self.fail_times = fail_times

    def add_documents(self, docs):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("db down")
        self.added.extend(docs)


@pytest.fixture
def fast_retry(monkeypatch):
    monkeypatch.setattr("app.services.ingest._RETRY_DELAYS", [0, 0, 0])


def test_supported_exts():
    assert SUPPORTED_EXTS == frozenset({"pdf", "docx", "pptx", "xlsx", "html", "md"})


def test_parse_markdown_file(fast_retry):
    chunks = parse_file(MD_DOC, "md")
    assert len(chunks) == 1 and "Title" in chunks[0].page_content


def test_parse_unsupported_ext():
    with pytest.raises(ValueError):
        parse_file(b"x", "exe")


def test_create_document_row(session, test_settings):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Ingest WS")
    doc = create_document_row(session, w, "policy.PDF", source="telegram", uploaded_by=owner.id, byte_size=10)
    session.flush()
    assert doc.file_ext == "pdf" and doc.status == "processing" and doc.id


def test_ingest_document_success(fast_retry, test_settings, session):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Ingest OK")
    doc = create_document_row(session, w, "a.md", source="seed")
    session.commit()
    store = FakeStore()
    n = ingest_document_sync(None, doc.id, MD_DOC, "md", store, settings=test_settings, _session=session)
    assert n == 1
    session.refresh(doc)
    assert doc.status == "ready" and doc.chunk_count == 1
    assert len(store.added) == 1
    assert store.added[0].metadata["document_id"] == doc.id


def test_ingest_document_retries_then_succeeds(fast_retry, test_settings, session):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Ingest Retry")
    doc = create_document_row(session, w, "a.md", source="seed")
    session.commit()
    store = FailingStore(fail_times=2)
    n = ingest_document_sync(None, doc.id, MD_DOC, "md", store, settings=test_settings, _session=session)
    assert n == 1 and len(store.added) == 1


def test_ingest_document_marks_failed(fast_retry, test_settings, session):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Ingest Fail")
    doc = create_document_row(session, w, "a.md", source="seed")
    session.commit()
    with pytest.raises(RuntimeError):
        ingest_document_sync(None, doc.id, MD_DOC, "md", FailingStore(fail_times=99),
                             settings=test_settings, _session=session)
    session.refresh(doc)
    assert doc.status == "failed" and "db down" in doc.error


def test_ingest_text_sync(fast_retry, test_settings, session):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Paste WS")
    store = FakeStore()
    doc = ingest_text_sync(None, w, "Meeting Notes", "point one " * 100, store,
                           uploaded_by=owner.id, _session=session)
    assert doc.status == "ready" and doc.source == "paste" and doc.filename == "Meeting Notes"
    assert len(store.added) >= 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_ingest_service.py -v`
Expected: FAIL — `No module named 'app.services.ingest'`

- [ ] **Step 3: Implement**

```python
# app/services/ingest.py
import asyncio
import tempfile
import time
from pathlib import Path

from langchain_core.documents import Document
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import Document as DocRow
from app.models import Workspace, Workspace as Ws
from app.rag.chunker import chunk_markdown, chunk_plain_text

SUPPORTED_EXTS = frozenset({"pdf", "docx", "pptx", "xlsx", "html", "md"})
_RETRY_DELAYS = [2, 4, 8]

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from docling.document_converter import DocumentConverter

        _converter = DocumentConverter()
    return _converter


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def parse_file(data: bytes, ext: str) -> list[Document]:
    ext = ext.lower().lstrip(".")
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type: .{ext}")
    if ext == "md":
        return chunk_markdown(data.decode("utf-8", errors="replace"))
    with tempfile.NamedTemporaryFile(suffix=f".{ext}") as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            result = _get_converter().convert(tmp.name)
            markdown = result.document.export_to_markdown()
        except Exception as e:
            raise RuntimeError(f"parse failed: {e}") from e
    return chunk_markdown(markdown)


def create_document_row(
    session: Session, workspace: Ws, filename: str, source: str,
    uploaded_by: str | None = None, byte_size: int = 0,
) -> DocRow:
    row = DocRow(
        workspace_id=workspace.id, filename=filename, file_ext=_ext(filename),
        byte_size=byte_size, source=source, uploaded_by=uploaded_by,
    )
    session.add(row)
    session.flush()
    return row


def save_original(settings: Settings, slug: str, document_id: str, ext: str, data: bytes) -> Path:
    folder = Path(settings.upload_dir) / slug
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{document_id}.{ext}"
    path.write_bytes(data)
    return path


def _add_with_retry(store, docs: list[Document]) -> None:
    last: Exception | None = None
    for delay in (*_RETRY_DELAYS, None):
        try:
            store.add_documents(docs)
            return
        except Exception as e:  # noqa: BLE001 — retry any store failure
            last = e
            if delay is not None:
                time.sleep(delay)
    raise last  # type: ignore[misc]


def ingest_document_sync(
    session_factory, document_id: str, data: bytes, ext: str, store,
    settings: Settings | None = None, _session: Session | None = None,
    source_file: str | None = None,
) -> int:
    if _session is not None:  # tests pass the live session directly
        row = _session.get(DocRow, document_id)
        try:
            chunks = parse_file(data, ext)
            for c in chunks:
                c.metadata["document_id"] = document_id
                if source_file:
                    c.metadata["source_file"] = source_file
            _add_with_retry(store, chunks)
            row.status, row.chunk_count, row.error = "ready", len(chunks), None
            _session.commit()
            return len(chunks)
        except Exception as e:
            _session.rollback()
            row = _session.get(DocRow, document_id)
            row.status, row.error = "failed", str(e)[:500]
            _session.commit()
            raise

    def _run(session: Session) -> int:
        row = session.get(DocRow, document_id)
        chunks = parse_file(data, ext)
        for c in chunks:
            c.metadata["document_id"] = document_id
            if source_file:
                c.metadata["source_file"] = source_file
        _add_with_retry(store, chunks)
        row.status, row.chunk_count, row.error = "ready", len(chunks), None
        session.commit()
        return len(chunks)

    try:
        with _open_session(session_factory) as session:
            return _run(session)
    except Exception as e:
        with _open_session(session_factory) as session:
            row = session.get(DocRow, document_id)
            row.status, row.error = "failed", str(e)[:500]
            session.commit()
        raise


async def ingest_document_async(session_factory, document_id: str, data: bytes, ext: str, store,
                                settings: Settings | None = None) -> int:
    return await asyncio.to_thread(
        ingest_document_sync, session_factory, document_id, data, ext, store, settings)


def ingest_text_sync(
    session_factory, workspace: Workspace, title: str, text: str, store,
    source: str = "paste", uploaded_by: str | None = None,
    settings: Settings | None = None, _session: Session | None = None,
) -> DocRow:
    chunks = chunk_plain_text(text)
    if not chunks:
        raise ValueError("Content is too short to index.")
    for c in chunks:
        c.metadata["document_id"] = None  # set after row exists
    if _session is not None:
        row = create_document_row(_session, workspace, title, source, uploaded_by, len(text.encode()))
        for c in chunks:
            c.metadata["document_id"] = row.id
        _add_with_retry(store, chunks)
        row.status, row.chunk_count = "ready", len(chunks)
        _session.commit()
        return row

    with _open_session(session_factory) as session:
        row = create_document_row(session, workspace, title, source, uploaded_by, len(text.encode()))
        for c in chunks:
            c.metadata["document_id"] = row.id
        _add_with_retry(store, chunks)
        row.status, row.chunk_count = "ready", len(chunks)
        session.commit()
        return row


class _open_session:
    def __init__(self, factory):
        self.factory = factory
        self.session = None

    def __enter__(self):
        self.session = self.factory() if not isinstance(self.factory, sessionmaker) else self.factory()
        return self.session

    def __exit__(self, *exc):
        self.session.close()
        return False
```

Note the two paths: the `_session=` backdoor exists purely so tests can assert row state on their live session; production callers pass a `sessionmaker`. The `paste` source value covers D3.

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_ingest_service.py -v`
Expected: 8 PASS

---

### Task 9: `app/rag/prompts.py` + `app/rag/graph.py` — the agentic RAG graph

**Files:**
- Create: `app/rag/prompts.py`, `app/rag/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Produces (`graph.py`):
  - `RAGState(TypedDict)`: `question: str`, `rewritten: str`, `attempt: int`, `retrieved: list`, `distances: list[float]`, `relevant: list`, `answer: str`, `sources: list[dict]`, `refused: str | None`, `history: Annotated[list, add_messages]`
  - `build_graph(llm, store_for_slug, settings=None, checkpointer=None)` — store_for_slug: `Callable[[str], store]` where store has `similarity_search_with_score(query, k) -> list[tuple[Document, float]]`
  - `get_graph(settings=None)` — production singleton: real ChatGroq + real per-slug stores + PostgresSaver on a `psycopg_pool.ConnectionPool` (created lazily; expose `get_graph` only).
  - `thread_id_for(conversation_id: str) -> str` → `f"conv-{conversation_id}"`
- Graph topology: `START → guard → retrieve → grade → (generate | rewrite | refuse) ; rewrite → retrieve ; generate/refuse → END`

Node behavior:
- **guard**: best-match check on the raw store (`k=1`); no docs at all → `refused="This workspace has no documents yet..."`; best distance > `off_topic_distance` → `refused="I can only answer questions about this workspace's documents."`; also appends `HumanMessage(question)` to `history`.
- **retrieve**: `similarity_search_with_score(rewritten, k=retrieve_k)` → `retrieved`, `distances`.
- **grade**: one LLM call; prompt lists numbered chunks, expects strict JSON array of relevant indices (e.g. `[0,2]`); parse with `json.loads` + regex-extract fallback; drop chunks whose distance > `off_topic_distance` regardless; `relevant` keeps original (distance) order — that is the rerank.
- **route_after_grade**: `relevant` → `generate`; elif `attempt == 0` → `rewrite`; else → `refuse` (`refused="I couldn't find anything about that in this workspace's documents."`).
- **rewrite**: LLM turns the conversation + question into a standalone search query → `rewritten`, `attempt=1`.
- **generate**: numbered context from `relevant[:top_n]`; prompt requires `[n]` citations and "I don't know." when context is insufficient; `sources` = cited chunks (fallback: all relevant) deduplicated by `(source_file, section)`, each `{"file": metadata["source_file"], "section": metadata.get("section")}`; appends `AIMessage(answer)` to `history`.

Chunk metadata contract: every stored chunk carries `source_file` (set by ingestion callers). `history` persists via the checkpointer, giving `PostgresSaver`-backed multi-turn memory; with `MemorySaver` in tests it works identically.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph.py
import json

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.config import Settings
from app.rag.graph import build_graph, thread_id_for

Q = "What is the delivery SLA?"


class FakeStore:
    def __init__(self, results):
        self.results = results  # list[(Document, distance)]
        self.queries = []

    def similarity_search_with_score(self, query, k=4):
        self.queries.append(query)
        return self.results[:k]


class FakeLLM:
    """Scripted responses. `grades` is a list of per-grade-call results (popped in
    order). Prompt-type detection uses distinctive markers from prompts.py:
    GRADE_PROMPT contains "JSON array", REWRITE_PROMPT contains "standalone search
    query", GENERATE_PROMPT contains neither."""

    def __init__(self, grades=(), answer="The SLA is 48 hours [1].", rewritten="SLA time"):
        self.grades = list(grades)
        self.answer = answer
        self.rewritten = rewritten
        self.prompts = []

    def invoke(self, prompt):
        text = str(prompt)
        self.prompts.append(text)
        if "JSON array" in text:
            idx = self.grades.pop(0) if self.grades else []
            return AIMessage(content=json.dumps(idx))
        if "standalone search query" in text:
            return AIMessage(content=self.rewritten)
        return AIMessage(content=self.answer)


def _doc(name, section=None):
    return Document(page_content=f"content of {name} " + "x " * 30,
                    metadata={"source_file": name, "section": section})


def _mk(store, llm, settings=None):
    s = settings or Settings(_env_file=None)
    return build_graph(llm, lambda slug: store, settings=s, checkpointer=MemorySaver())


def test_happy_path_with_citations():
    store = FakeStore([(_doc("sla.md", "Zones"), 0.2), (_doc("other.md"), 0.8)])
    llm = FakeLLM(grades=[[0]])
    out = _mk(store, llm).invoke({"question": Q, "rewritten": Q, "attempt": 0,
                                  "workspace_slug": "ws1", "history": []})
    assert out["answer"] == "The SLA is 48 hours [1]."
    assert out["sources"] == [{"file": "sla.md", "section": "Zones"}]
    assert out["refused"] is None
    assert isinstance(out["history"][-1], AIMessage)
    assert any(str(m.content) == Q for m in out["history"] if isinstance(m, HumanMessage))


def test_grading_filters_irrelevant_chunks():
    store = FakeStore([(_doc("a.md"), 0.1), (_doc("b.md"), 0.3), (_doc("c.md"), 0.5)])
    llm = FakeLLM(grades=[[0, 2]], answer="Use [1] and [2].")
    out = _mk(store, llm).invoke({"question": Q, "rewritten": Q, "attempt": 0,
                                  "workspace_slug": "ws1", "history": []})
    ctx = llm.prompts[-1]
    assert "content of a.md" in ctx and "content of c.md" in ctx
    assert "content of b.md" not in ctx  # graded out — the rerank
    assert {s["file"] for s in out["sources"]} == {"a.md", "c.md"}


def test_off_topic_guard_refuses_before_llm():
    store = FakeStore([(_doc("a.md"), 0.99)])
    llm = FakeLLM(grades=[[0]])
    out = _mk(store, llm).invoke({"question": "who won the world cup", "rewritten": "who won the world cup",
                                  "attempt": 0, "workspace_slug": "ws1", "history": []})
    assert out["refused"] and out["answer"] == ""
    assert llm.prompts == []  # never called


def test_empty_workspace_refuses():
    store = FakeStore([])
    llm = FakeLLM(grades=[[0]])
    out = _mk(store, llm).invoke({"question": Q, "rewritten": Q, "attempt": 0,
                                  "workspace_slug": "ws1", "history": []})
    assert "no documents" in out["refused"]


def test_rewrite_retry_recovers():
    store = FakeStore([(_doc("sla.md"), 0.2)])
    llm = FakeLLM(grades=[[], [0]], answer="placeholder")  # grade#1 empty, grade#2 relevant
    graph = _mk(store, llm)
    state = graph.invoke({"question": "tell me the sla", "rewritten": "tell me the sla",
                          "attempt": 0, "workspace_slug": "ws1", "history": []})
    # guard + retrieve#1 use the raw question; retrieve#2 uses the rewritten query
    assert store.queries[0] == "tell me the sla"
    assert store.queries[1] == "tell me the sla"
    assert store.queries[2] == "SLA time"
    assert state["answer"] == "placeholder"
    assert state["sources"] == [{"file": "sla.md", "section": None}]


def test_second_empty_grade_refuses():
    store = FakeStore([(_doc("sla.md"), 0.2)])
    llm = FakeLLM(grades=[[], []])
    out = _mk(store, llm).invoke({"question": Q, "rewritten": Q, "attempt": 0,
                                  "workspace_slug": "ws1", "history": []})
    assert out["refused"] and out["answer"] == ""


def test_grade_prompt_is_batch_json():
    store = FakeStore([(_doc("a.md"), 0.1), (_doc("b.md"), 0.2)])
    llm = FakeLLM(grades=[[0, 1]])
    _mk(store, llm).invoke({"question": Q, "rewritten": Q, "attempt": 0,
                            "workspace_slug": "ws1", "history": []})
    grade_prompt = next(p for p in llm.prompts if "JSON array" in p)
    assert "0)" in grade_prompt and "1)" in grade_prompt  # numbered chunks


def test_memory_persists_across_invocations():
    store = FakeStore([(_doc("sla.md"), 0.2)])
    llm = FakeLLM(grades=[[0], [0]])
    graph = _mk(store, llm)
    cfg = {"configurable": {"thread_id": thread_id_for("c1")}}
    graph.invoke({"question": Q, "rewritten": Q, "attempt": 0,
                  "workspace_slug": "ws1", "history": []}, config=cfg)
    out = graph.invoke({"question": "and for zone 2?", "rewritten": "and for zone 2?",
                        "attempt": 0, "workspace_slug": "ws1", "history": []}, config=cfg)
    history = out["history"]
    contents = [m.content for m in history]
    assert Q in contents and "and for zone 2?" in contents  # same thread accumulated
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_graph.py -v`
Expected: FAIL — `No module named 'app.rag.graph'`

- [ ] **Step 3: Implement `app/rag/prompts.py`**

```python
# app/rag/prompts.py
GENERATE_PROMPT = """You are a documentation assistant. Answer the question using ONLY the context below.

Rules:
- Cite the context numbers you used inline, like [1] or [2].
- If the context does not contain the answer, reply exactly: I don't know.
- Be concise.

{history}

Context:
{context}

Question: {question}
"""

GRADE_PROMPT = """You are grading retrieved document chunks for relevance.

Question: {question}

Chunks:
{chunks}

Which chunk numbers contain information needed to answer the question?
Reply with ONLY a JSON array of numbers, e.g. [0,2]. Reply [] if none are relevant.
"""

REWRITE_PROMPT = """Rewrite the user's latest message as one standalone search query.
Resolve pronouns using the conversation. Reply with the query only, no quotes.

{history}

Latest message: {question}
"""


def format_history(messages, max_chars=1500) -> str:
    """Render recent chat turns for prompt injection (oldest → newest, truncated)."""
    lines = [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
             for m in messages[-8:] if isinstance(m, (HumanMessage, AIMessage))]
    out = "\n".join(lines)
    return "Conversation so far:\n" + out if out else ""


from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402  (used above)
```

(Move the import to the top of the file when writing it — the trailing import is shown here only to flag that `HumanMessage`/`AIMessage` are needed.)

- [ ] **Step 4: Implement `app/rag/graph.py`**

```python
# app/rag/graph.py
import json
import re
from typing import Annotated, Callable, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import Settings
from app.rag.prompts import GENERATE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT, format_history


class RAGState(TypedDict):
    workspace_slug: str
    question: str
    rewritten: str
    attempt: int
    retrieved: list
    distances: list[float]
    relevant: list
    answer: str
    sources: list[dict]
    refused: str | None
    history: Annotated[list, add_messages]


def thread_id_for(conversation_id: str) -> str:
    return f"conv-{conversation_id}"


def _parse_grade(raw: str, n: int) -> list[int]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[[\d\s,]*\]", raw)
        data = json.loads(match.group()) if match else []
    return [i for i in data if isinstance(i, int) and 0 <= i < n]


def _numbered(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


def _text(reply) -> str:
    """Unwrap a chat-model reply (AIMessage) to plain text."""
    content = getattr(reply, "content", reply)
    return content if isinstance(content, str) else str(content)


def build_graph(llm, store_for_slug: Callable, settings: Settings | None = None,
                checkpointer=None):
    s = settings or Settings(_env_file=None)

    def guard(state: RAGState) -> dict:
        store = store_for_slug(state["workspace_slug"])
        best = store.similarity_search_with_score(state["question"], k=1)
        update = {"history": [HumanMessage(content=state["question"])],
                  "answer": "", "sources": [], "refused": None}
        if not best:
            update["refused"] = ("This workspace has no documents yet. Send me a file or "
                                 "paste some content first (/new).")
            return update
        if best[0][1] > s.off_topic_distance:
            update["refused"] = "I can only answer questions about this workspace's documents."
            return update
        return update

    def retrieve(state: RAGState) -> dict:
        store = store_for_slug(state["workspace_slug"])
        results = store.similarity_search_with_score(state["rewritten"], k=s.retrieve_k)
        docs = [d for d, _ in results]
        dists = [dist for _, dist in results]
        return {"retrieved": docs, "distances": dists}

    def grade(state: RAGState) -> dict:
        docs = state["retrieved"]
        listing = "\n\n".join(f"{i}) {d.page_content[:600]}" for i, d in enumerate(docs))
        prompt = GRADE_PROMPT.format(question=state["rewritten"], chunks=listing)
        idx = set(_parse_grade(_text(llm.invoke(prompt)), len(docs)))
        relevant = [d for i, (d, dist) in enumerate(zip(docs, state["distances"]))
                    if i in idx and dist <= s.off_topic_distance]
        return {"relevant": relevant}

    def route_after_grade(state: RAGState) -> str:
        if state["relevant"]:
            return "generate"
        if state["attempt"] == 0:
            return "rewrite"
        return "refuse"

    def rewrite(state: RAGState) -> dict:
        prompt = REWRITE_PROMPT.format(history=format_history(state["history"]),
                                       question=state["question"])
        new_query = _text(llm.invoke(prompt)).strip().strip('"')
        return {"rewritten": new_query or state["question"], "attempt": 1}

    def refuse(state: RAGState) -> dict:
        return {"refused": state["refused"] or "I couldn't find anything about that in this "
                                                   "workspace's documents.",
                "answer": "", "sources": []}

    def route_after_guard(state: RAGState) -> str:
        return "refuse" if state["refused"] else "retrieve"

    def generate(state: RAGState) -> dict:
        docs = state["relevant"][: s.top_n]
        prompt = GENERATE_PROMPT.format(
            history=format_history(state["history"]),
            context=_numbered(docs),
            question=state["rewritten"],
        )
        answer = _text(llm.invoke(prompt)).strip()
        cited = {int(n) - 1 for n in re.findall(r"\[(\d+)\]", answer)
                 if 0 < int(n) <= len(docs)}
        used = [docs[i] for i in sorted(cited)] or docs
        sources, seen = [], set()
        for d in used:
            key = (d.metadata.get("source_file"), d.metadata.get("section"))
            if key not in seen:
                seen.add(key)
                sources.append({"file": key[0], "section": key[1]})
        return {"answer": answer, "sources": sources,
                "history": [AIMessage(content=answer)]}

    g = StateGraph(RAGState)
    g.add_node("guard", guard)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("refuse", refuse)
    g.add_node("generate", generate)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", route_after_guard,
                            {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile(checkpointer=checkpointer)


_production = None


def get_graph(settings: Settings | None = None):
    """Production singleton: real Groq, real stores, Postgres-backed checkpointer."""
    global _production
    if _production is None:
        from psycopg_pool import ConnectionPool

        from app.config import get_settings
        from app.rag.embeddings import get_embeddings
        from app.rag.llm import get_llm
        from app.rag.vectorstore import get_store

        s = settings or get_settings()
        pool = ConnectionPool(s.database_url_plain, open=True, kwargs={"autocommit": True})
        from langgraph.checkpoint.postgres import PostgresSaver

        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        _production = build_graph(
            get_llm(s),
            lambda slug: get_store(s, slug, get_embeddings(s)),
            settings=s,
            checkpointer=checkpointer,
        )
    return _production
```

- [ ] **Step 5: Run to verify pass**

Run: `python3 -m pytest tests/test_graph.py -v`
Expected: 8 PASS

Note the `grade → route` conditional returning three targets: this is what makes the graph agentic (grade/rerank + one rewrite-retry) rather than the old single node.

---

### Task 10: `app/services/chat.py` — conversations + run_chat

**Files:**
- Create: `app/services/chat.py`
- Test: `tests/test_chat_service.py`

**Interfaces:**
- Consumes: `Conversation` model, `graph.build_graph` (via injected graph), `thread_id_for`.
- Produces:
  - `@dataclass ChatResult: answer: str, sources: list[dict], conversation_id: str, refused: bool`
  - `start_conversation(session, user, workspace, title="New chat") -> Conversation` — sets `user.current_conversation_id`, clears `user.pending_action`, sets `user.current_workspace_id`.
  - `run_chat(session, user, workspace, message, graph, settings=None) -> ChatResult` — conversation = user's current (if same workspace) else create; invoke graph with `thread_id_for`; on reply: set title from first user message (64 chars) if still `"New chat"`, bump `last_message_at`, commit; `refused = bool(state["refused"])`; `answer = state["refused"] or state["answer"]`.
  - `list_conversations(session, user, limit=8) -> list[Conversation]` — most recent first by `last_message_at`/`created_at`.
  - `resume_conversation(session, user, conversation_id) -> Conversation | None` — must belong to user; sets `current_conversation_id`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chat_service.py
from app.models import Conversation, User, Workspace
from app.services.chat import (
    ChatResult, list_conversations, resume_conversation, run_chat, start_conversation,
)
from app.services.users import ensure_system_user


class FakeGraph:
    def __init__(self, answer="A [1]", sources=None, refused=None):
        self.answer, self.sources, self.refused = answer, sources or [], refused
        self.invocations = []

    def invoke(self, state, config=None):
        self.invocations.append((state, config))
        return {"answer": self.answer, "sources": self.sources,
                "refused": self.refused, "history": []}


def _user_and_ws(session):
    user = User(telegram_id=5, display_name="T")
    session.add(user)
    session.flush()
    ws = Workspace(slug="cws", name="C", owner_id=user.id)
    session.add(ws)
    session.flush()
    return user, ws


def test_start_conversation_sets_current(session):
    user, ws = _user_and_ws(session)
    conv = start_conversation(session, user, ws, title="Hello policy")
    assert user.current_conversation_id == conv.id
    assert user.current_workspace_id == ws.id
    assert user.pending_action is None
    assert conv.title == "Hello policy" and conv.is_active


def test_run_chat_creates_and_titles_conversation(session):
    user, ws = _user_and_ws(session)
    graph = FakeGraph()
    result = run_chat(session, user, ws, "What is the refund window?", graph)
    assert isinstance(result, ChatResult)
    conv_id = result.conversation_id
    conv = session.get(Conversation, conv_id)
    assert conv.title == "What is the refund window?"[:64]
    assert conv.last_message_at is not None
    assert result.answer == "A [1]"
    state, config = graph.invocations[0]
    assert config["configurable"]["thread_id"] == f"conv-{conv_id}"
    assert state["question"] == "What is the refund window?"


def test_run_chat_reuses_current_conversation(session):
    user, ws = _user_and_ws(session)
    graph = FakeGraph()
    r1 = run_chat(session, user, ws, "first question", graph)
    r2 = run_chat(session, user, ws, "second question", graph)
    assert r1.conversation_id == r2.conversation_id
    assert len(graph.invocations) == 2
    # different workspace → new conversation
    ws2 = Workspace(slug="cws2", name="C2", owner_id=user.id)
    session.add(ws2)
    session.flush()
    r3 = run_chat(session, user, ws2, "other workspace now", graph)
    assert r3.conversation_id != r1.conversation_id


def test_run_chat_refused_answer(session):
    user, ws = _user_and_ws(session)
    result = run_chat(session, user, ws, "gibberish", FakeGraph(refused="Nope."))
    assert result.answer == "Nope." and result.refused is True


def test_list_and_resume_conversations(session):
    user, ws = _user_and_ws(session)
    c1 = start_conversation(session, user, ws, "one")
    c2 = start_conversation(session, user, ws, "two")
    assert {c.id for c in list_conversations(session, user)} == {c1.id, c2.id}
    assert resume_conversation(session, user, c1.id).id == c1.id
    assert user.current_conversation_id == c1.id
    other = User(telegram_id=6, display_name="X")
    session.add(other)
    session.flush()
    assert resume_conversation(session, other, c1.id) is None  # not theirs
    assert resume_conversation(session, user, "no-such-id") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_chat_service.py -v`
Expected: FAIL — `No module named 'app.services.chat'`

- [ ] **Step 3: Implement**

```python
# app/services/chat.py
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Conversation, User, Workspace


@dataclass
class ChatResult:
    answer: str
    sources: list[dict]
    conversation_id: str
    refused: bool


def start_conversation(session: Session, user: User, workspace: Workspace,
                      title: str = "New chat") -> Conversation:
    conv = Conversation(user_id=user.id, workspace_id=workspace.id, title=title)
    session.add(conv)
    session.flush()
    user.current_conversation_id = conv.id
    user.current_workspace_id = workspace.id
    user.pending_action = None
    session.flush()
    return conv


def list_conversations(session: Session, user: User, limit: int = 8) -> list[Conversation]:
    return list(session.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id, Conversation.is_active.is_(True))
        .order_by(Conversation.last_message_at.desc().nullslast(), Conversation.created_at.desc())
        .limit(limit)
    ))


def resume_conversation(session: Session, user: User, conversation_id: str) -> Conversation | None:
    conv = session.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        return None
    user.current_conversation_id = conv.id
    user.current_workspace_id = conv.workspace_id
    user.pending_action = None
    session.flush()
    return conv


def _current_conversation(session: Session, user: User, workspace: Workspace) -> Conversation:
    if user.current_conversation_id:
        conv = session.get(Conversation, user.current_conversation_id)
        if conv is not None and conv.workspace_id == workspace.id and conv.is_active:
            return conv
    return start_conversation(session, user, workspace)


def run_chat(session: Session, user: User, workspace: Workspace, message: str,
             graph, settings: Settings | None = None) -> ChatResult:
    from app.rag.graph import thread_id_for

    conv = _current_conversation(session, user, workspace)
    state = graph.invoke(
        {"question": message, "rewritten": message, "attempt": 0,
         "workspace_slug": workspace.slug, "history": []},
        config={"configurable": {"thread_id": thread_id_for(conv.id)}},
    )
    refused = bool(state.get("refused"))
    answer = state["refused"] or state["answer"]
    if conv.title == "New chat":
        conv.title = message.strip()[:64]
    conv.last_message_at = datetime.now(timezone.utc)
    session.commit()
    return ChatResult(answer=answer, sources=state.get("sources", []),
                      conversation_id=conv.id, refused=refused)
```

(Imports at the top of the file: `from datetime import datetime, timezone`.)

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_chat_service.py tests/test_graph.py -v`
Expected: 13 PASS

---

### Task 11: `app/schemas.py` + routers — REST API

**Files:**
- Create: `app/schemas.py`, `app/routers/__init__.py`, `app/routers/health.py`, `app/routers/workspaces.py`, `app/routers/documents.py`, `app/routers/chat.py`, `app/routers/telegram_webhook.py`
- Test: `tests/test_routers.py`

**Interfaces:**
- Consumes: everything from Tasks 1–10; `services.chat.run_chat` with production graph passed in via `app.state.graph`.
- Produces (mounted by `create_app` in Task 14):
  - `GET /health` → `{"status": "ok", "bot_mode": <mode>}` (no auth)
  - `GET /api/v1/workspaces` → `[{"slug","name","description","is_private","documents","chunks"}]`
  - `POST /api/v1/workspaces` `{"name","description"?,"is_private"?}` → `201` same shape; 409 never happens (unique_slug)
  - `GET /api/v1/workspaces/{slug}/documents` → `[{"id","filename","status","chunk_count","error","source"}]`; 404 unknown slug
  - `POST /api/v1/workspaces/{slug}/documents` multipart `file` → `202 {"document_id","status":"processing"}`; 400 unsupported type / too big; 404 unknown slug; 422 empty parse
  - `POST /api/v1/workspaces/{slug}/chat` `{"message","conversation_id"?}` → `200 {"answer","sources","conversation_id","refused"}`
  - `POST /telegram/webhook` — validates `X-Telegram-Bot-Api-Secret-Token` against `settings.telegram_webhook_secret`; mismatch → 403; then `Update.de_json` → `app.state.ptb_app.process_update(update)` → `{}`
- `require_api_key` dependency: `X-API-Key` header → `security.api_key_ok` else 401 `{"detail": "Invalid or missing API key"}`. API keys are platform-level: REST can access any workspace (documented in README).
- REST-created workspaces are owned by the system user. REST chat ignores membership (platform-level key).
- Upload flow: save `UploadFile` bytes (≤ 20 MB), create row, `asyncio.create_task(ingest_document_async(SessionLocal(settings), doc.id, data, ext, get_store(settings, slug), settings))`.
- Chat endpoint calls `run_chat` via `asyncio.to_thread` (Groq + embedding calls are blocking under the hood).
- 20 MB guard: read bytes, `len(data) > 20 * 1024 * 1024` → 400.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_routers.py
import io

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import User
from app.services.ingest import create_document_row


class FakeGraph:
    def invoke(self, state, config=None):
        return {"answer": "Answer [1]", "sources": [{"file": "a.md", "section": None}],
                "refused": None, "history": []}


@pytest.fixture
def client(test_settings, session):
    from app.services.users import ensure_system_user

    ensure_system_user(session)
    session.commit()
    app = create_app(test_settings)
    app.state.graph = FakeGraph()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, test_settings


HEADERS = {"X-API-Key": "dev-api-key-1"}


def test_health_no_auth(client):
    c, s = client
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_api_requires_key(client):
    c, s = client
    assert c.get("/api/v1/workspaces").status_code == 401
    assert c.get("/api/v1/workspaces", headers={"X-API-Key": "wrong"}).status_code == 401


def test_workspace_crud(client):
    c, s = client
    r = c.post("/api/v1/workspaces", headers=HEADERS, json={"name": "API Made", "is_private": True})
    assert r.status_code == 201
    body = r.json()
    assert body["slug"] == "api-made" and body["is_private"] is True
    listed = c.get("/api/v1/workspaces", headers=HEADERS).json()
    assert any(w["slug"] == "api-made" for w in listed)


def test_documents_404_and_upload(client, session):
    c, s = client
    assert c.get("/api/v1/workspaces/missing/documents", headers=HEADERS).status_code == 404
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    ws_session = session  # reuse
    owner = ensure_system_user(ws_session)
    ws = create_workspace(ws_session, owner, "Upload Target")
    ws_session.commit()
    r = c.post(f"/api/v1/workspaces/upload-target/documents", headers=HEADERS,
               files={"file": ("notes.md", io.BytesIO(b"# H\n\nSome long enough content here.\n"), "text/markdown")})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "processing" and body["document_id"]
    listed = c.get("/api/v1/workspaces/upload-target/documents", headers=HEADERS).json()
    assert listed[0]["filename"] == "notes.md"


def test_upload_rejects_bad_type_and_oversize(client, session):
    c, s = client
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    create_workspace(session, owner, "Bad Uploads")
    session.commit()
    r = c.post("/api/v1/workspaces/bad-uploads/documents", headers=HEADERS,
               files={"file": ("virus.exe", io.BytesIO(b"x"), "application/x-msdownload")})
    assert r.status_code == 400


def test_chat_endpoint(client, session):
    c, s = client
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    create_workspace(session, owner, "Chat WS")
    session.commit()
    r = c.post("/api/v1/workspaces/chat-ws/chat", headers=HEADERS, json={"message": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Answer [1]"
    assert body["sources"] == [{"file": "a.md", "section": None}]
    assert body["conversation_id"] and body["refused"] is False


def test_webhook_rejects_bad_secret(client):
    c, s = client
    r = c.post("/telegram/webhook", json={}, headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"})
    assert r.status_code == 403
```

Note: the webhook 403 test works in `disabled` mode (secret check precedes any PTB usage — see Step 3).

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_routers.py -v`
Expected: FAIL — `cannot import name 'create_app'`

- [ ] **Step 3: Implement schemas + routers**

```python
# app/schemas.py
from pydantic import BaseModel, Field


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    is_private: bool = False


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    conversation_id: str
    refused: bool


class WorkspaceOut(BaseModel):
    slug: str
    name: str
    description: str | None
    is_private: bool
    documents: int
    chunks: int


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    chunk_count: int
    error: str | None
    source: str
```

```python
# app/routers/health.py
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    return {"status": "ok", "bot_mode": request.app.state.settings.bot_mode}
```

```python
# app/routers/workspaces.py
from fastapi import APIRouter, Depends, Request
from sqlalchemy import select

from app.schemas import WorkspaceCreate, WorkspaceOut
from app.security import api_key_dependency
from app.services.users import ensure_system_user
from app.services.workspaces import create_workspace, workspace_stats

router = APIRouter(prefix="/api/v1", dependencies=[Depends(api_key_dependency)])


@router.get("/workspaces", response_model=list[WorkspaceOut])
def list_workspaces(request: Request):
    from app.db import SessionLocal
    from app.models import Workspace

    settings = request.app.state.settings
    # API keys are platform-level: REST sees all workspaces.
    with SessionLocal(settings)() as session:
        all_ws = session.scalars(select(Workspace).order_by(Workspace.created_at))
        return [WorkspaceOut(
            slug=w.slug, name=w.name, description=w.description, is_private=w.is_private,
            **workspace_stats(session, w)) for w in all_ws]


@router.post("/workspaces", response_model=WorkspaceOut, status_code=201)
def create_workspace_ep(request: Request, payload: WorkspaceCreate):
    from app.db import SessionLocal

    settings = request.app.state.settings
    with SessionLocal(settings)() as session:
        system = ensure_system_user(session)
        w = create_workspace(session, system, payload.name,
                             payload.description, payload.is_private)
        session.commit()
        return WorkspaceOut(slug=w.slug, name=w.name, description=w.description,
                            is_private=w.is_private, documents=0, chunks=0)
```

```python
# app/routers/documents.py
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile

from app.schemas import DocumentOut
from app.security import api_key_dependency

router = APIRouter(prefix="/api/v1/workspaces/{slug}/documents",
                   dependencies=[Depends(api_key_dependency)])
MAX_BYTES = 20 * 1024 * 1024


@router.get("", response_model=list[DocumentOut])
def list_documents(request: Request, slug: str):
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import Document

    settings = request.app.state.settings
    with SessionLocal(settings)() as session:
        from app.services.workspaces import workspace_by_slug

        ws = workspace_by_slug(session, slug)
        if ws is None:
            raise HTTPException(status_code=404, detail="Unknown workspace")
        rows = session.scalars(select(Document).where(Document.workspace_id == ws.id)
                               .order_by(Document.created_at.desc()))
        return [DocumentOut(id=d.id, filename=d.filename, status=d.status,
                            chunk_count=d.chunk_count, error=d.error, source=d.source)
                for d in rows]


@router.post("", status_code=202)
async def upload_document(request: Request, slug: str, file: UploadFile):
    from app.db import SessionLocal
    from app.models import Document
    from app.rag.vectorstore import get_store
    from app.services.ingest import SUPPORTED_EXTS, ingest_document_async, save_original
    from app.services.workspaces import workspace_by_slug

    settings = request.app.state.settings
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 20 MB limit")

    session_factory = SessionLocal(settings)
    with session_factory() as session:
        ws = workspace_by_slug(session, slug)
        if ws is None:
            raise HTTPException(status_code=404, detail="Unknown workspace")
        from app.services.ingest import create_document_row

        doc = create_document_row(session, ws, file.filename, source="api", byte_size=len(data))
        session.commit()
        doc_id, slug_value = doc.id, ws.slug
    save_original(settings, slug_value, doc_id, ext, data)
    store = get_store(settings, slug_value)
    asyncio.create_task(ingest_document_async(session_factory, doc_id, data, ext, store, settings))
    return {"document_id": doc_id, "status": "processing"}
```

```python
# app/routers/chat.py
import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas import ChatRequest, ChatResponse
from app.security import api_key_dependency

router = APIRouter(prefix="/api/v1/workspaces/{slug}/chat",
                   dependencies=[Depends(api_key_dependency)])


@router.post("", response_model=ChatResponse)
async def chat(request: Request, slug: str, payload: ChatRequest):
    from app.db import SessionLocal
    from app.services.chat import run_chat
    from app.services.users import ensure_system_user
    from app.services.workspaces import workspace_by_slug

    settings = request.app.state.settings
    session_factory = SessionLocal(settings)
    with session_factory() as session:
        ws = workspace_by_slug(session, slug)
        if ws is None:
            raise HTTPException(status_code=404, detail="Unknown workspace")
        system = ensure_system_user(session)
        result = await asyncio.to_thread(
            run_chat, session, system, ws, payload.message,
            request.app.state.graph, settings)
        session.commit()
        return ChatResponse(answer=result.answer, sources=result.sources,
                            conversation_id=result.conversation_id, refused=result.refused)
```

```python
# app/routers/telegram_webhook.py
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    settings = request.app.state.settings
    if request.headers.get("X-Telegram-Bot-Api-Secret-Token") != settings.telegram_webhook_secret:
        raise HTTPException(status_code=403, detail="Bad webhook secret")
    ptb_app = getattr(request.app.state, "ptb_app", None)
    if ptb_app is None:
        raise HTTPException(status_code=503, detail="Bot disabled")
    from telegram import Update

    update = Update.de_json(await request.json(), ptb_app.bot)
    await ptb_app.process_update(update)
    return {}
```

- [ ] **Step 4: Implement minimal `app/main.py`** (full lifespan lands in Task 14 — here just enough to serve tests)

```python
# app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.settings
    init_db(settings)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(title="Telegram Multi-Workspace RAG", lifespan=lifespan)
    app.state.settings = s

    from app.routers import chat, documents, health, telegram_webhook, workspaces

    app.include_router(health.router)
    app.include_router(workspaces.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(telegram_webhook.router)
    return app


app = create_app()
```

- [ ] **Step 5: Run to verify pass**

Run: `python3 -m pytest tests/test_routers.py -v`
Expected: 7 PASS

---

### Task 12: `app/bot/format.py` + `app/bot/keyboards.py`

**Files:**
- Create: `app/bot/__init__.py`, `app/bot/format.py`, `app/bot/keyboards.py`
- Test: `tests/test_format.py`, `tests/test_keyboards.py`

**Interfaces:**
- Produces (`format.py`): `esc(s) -> str` (html.escape), `split_message(text, limit=4096) -> list[str]` (split at newline boundaries; hard-slice oversized lines; empty → `[""]` never), `format_answer(answer, sources) -> str` (answer, then optional `\n\n<b>Sources:</b>` block: `[n] file — section` with esc; duplicates already removed upstream).
- Produces (`keyboards.py`): `DEMO_QUESTIONS: list[str]` (3 fixed questions about the seeded corpus), `workspace_keyboard(workspaces, current_id=None) -> InlineKeyboardMarkup` (callback `ws:<workspace_id>`; ✓ prefix on current), `conversations_keyboard(conversations) -> InlineKeyboardMarkup` (callback `cv:<conversation_id>`; label `title[:32]`), `demo_keyboard() -> InlineKeyboardMarkup` (callback `dq:<index>`), `new_chat_keyboard() -> InlineKeyboardMarkup` (buttons: "📝 I'll paste content" → `nc:content`, "❓ Just chat" → `nc:skip`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_format.py
from app.bot.format import esc, format_answer, split_message


def test_escapes_html():
    assert esc("<b> & 'x'") == "&lt;b&gt; &amp; &#x27;x&#x27;"


def test_split_short_message_unchanged():
    assert split_message("hello") == ["hello"]


def test_split_long_message_on_newlines():
    para = "word " * 300
    text = "\n\n".join([para] * 6)  # ~11k chars
    parts = split_message(text)
    assert len(parts) >= 3
    assert all(len(p) <= 4096 for p in parts)
    assert "\n\n".join(parts).count("word") == text.count("word")


def test_split_hard_slices_when_no_newlines():
    text = "x" * 10_000
    parts = split_message(text)
    assert sum(len(p) for p in parts) == 10_000
    assert all(len(p) <= 4096 for p in parts)


def test_format_answer_with_sources():
    out = format_answer("SLA is 48h [1].", [{"file": "sla.md", "section": "Zones"}])
    assert "SLA is 48h [1]." in out
    assert "<b>Sources:</b>" in out
    assert "[1] sla.md — Zones" in out


def test_format_answer_escapes_source_names():
    out = format_answer("a", [{"file": "<script>.md", "section": None}])
    assert "<script>" not in out.split("Sources:")[-1]


def test_format_answer_no_sources_block_when_empty():
    assert "Sources" not in format_answer("plain", [])
```

```python
# tests/test_keyboards.py
from app.bot.keyboards import (
    DEMO_QUESTIONS, conversations_keyboard, demo_keyboard, new_chat_keyboard,
    workspace_keyboard,
)


class FakeWS:
    def __init__(self, wid, slug):
        self.id, self.slug, self.name = wid, slug, slug


def test_demo_questions_exist():
    assert len(DEMO_QUESTIONS) == 3 and all(len(q) > 10 for q in DEMO_QUESTIONS)


def test_workspace_keyboard_marks_current():
    kb = workspace_keyboard([FakeWS("1", "alpha"), FakeWS("2", "beta")], current_id="2")
    labels = [b.text for row in kb.inline_keyboard for b in row]
    cbs = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert labels == ["alpha", "✓ beta"]
    assert cbs == ["ws:1", "ws:2"]


def test_conversations_keyboard():
    class C:
        id, title = "c1", "x" * 50

    kb = conversations_keyboard([C()])
    btn = kb.inline_keyboard[0][0]
    assert btn.callback_data == "cv:c1" and len(btn.text) <= 34


def test_demo_and_new_keyboards():
    assert demo_keyboard().inline_keyboard[0][0].callback_data == "dq:0"
    labels = [b.callback_data for b in new_chat_keyboard().inline_keyboard[0]]
    assert labels == ["nc:content", "nc:skip"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_format.py tests/test_keyboards.py -v`
Expected: FAIL — `No module named 'app.bot'`

- [ ] **Step 3: Implement**

```python
# app/bot/format.py
import html

LIMIT = 4096


def esc(s: str) -> str:
    return html.escape(str(s or ""))


def split_message(text: str, limit: int = LIMIT) -> list[str]:
    text = text or ""
    if len(text) <= limit:
        return [text] if text else [""]
    parts, current = [], ""
    for para in text.split("\n"):
        while len(para) > limit:  # single oversized line: hard slice
            if current:
                parts.append(current)
                current = ""
            parts.append(para[:limit])
            para = para[limit:]
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) > limit:
            parts.append(current)
            current = para
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def format_answer(answer: str, sources: list[dict]) -> str:
    out = answer or ""
    if sources:
        lines = []
        for i, s in enumerate(sources, 1):
            line = f"[{i}] {esc(s.get('file') or 'unknown')}"
            if s.get("section"):
                line += f" — {esc(s['section'])}"
            lines.append(line)
        out += "\n\n<b>Sources:</b>\n" + "\n".join(lines)
    return out
```

```python
# app/bot/keyboards.py
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

DEMO_QUESTIONS = [
    "What is the delivery SLA for each zone?",
    "Which goods are classified as special goods?",
    "What is the refund window for returned items?",
]


def workspace_keyboard(workspaces, current_id=None) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(
        text=f"{'✓ ' if w.id == current_id else ''}{w.name}",
        callback_data=f"ws:{w.id}")] for w in workspaces]
    return InlineKeyboardMarkup(rows)


def conversations_keyboard(conversations) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text=c.title[:32], callback_data=f"cv:{c.id}")]
            for c in conversations]
    return InlineKeyboardMarkup(rows)


def demo_keyboard() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text=q, callback_data=f"dq:{i}")]
            for i, q in enumerate(DEMO_QUESTIONS)]
    return InlineKeyboardMarkup(rows)


def new_chat_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📝 I'll paste content", callback_data="nc:content"),
        InlineKeyboardButton("❓ Just chat", callback_data="nc:skip"),
    ]])
```

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_format.py tests/test_keyboards.py -v`
Expected: 11 PASS

---

### Task 13: `app/bot/handlers.py`

**Files:**
- Create: `app/bot/handlers.py`
- Test: `tests/test_bot_handlers.py`

**Interfaces:**
- Consumes: services (users, workspaces, ingest, chat), security, keyboards, format. Every handler resolves dependencies from `context.application.bot_data["container"]` — a dataclass `Container(settings, session_factory, graph)` created in Task 14. Sessions: `with container.session_factory() as session: ...` (commit at the end of each handler that writes).
- Produces (async, `(update, context)` signature — PTB contract):
  - `cmd_start` — register user; if they have a current workspace show it, else prompt; inline keyboard: workspaces list (if any) + demo button.
  - `cmd_help`, `cmd_workspaces` (list buttons; empty → "No workspaces yet — create one with /newworkspace"), `cmd_workspace` (current workspace + stats), `cmd_newworkspace` (`/newworkspace <name> [private]`; max 40 chars name; caller becomes owner; switches them there + starts new conversation), `cmd_invite` (`/invite @username` — owner/editor on private ws; target must exist as a registered user), `cmd_public`/`cmd_private` (owner toggles), `cmd_new` (starts new conversation + `new_chat_keyboard`; if `nc:content` → `user.pending_action="awaiting_content"`), `cmd_resume` (buttons of last 8 conversations), `cmd_demo` (switch to `seed.DEMO_SLUG` workspace, new conversation, text + `demo_keyboard()`).
  - `on_callback` — routes `callback_data` prefixes: `ws:` switch workspace + new conversation in it; `cv:` resume; `dq:` answer the demo question via `run_chat`; `nc:content`/`nc:skip`. Always `callback_query.answer()`.
  - `on_document` — access check (`can_ingest`); ext check; 20 MB check; "📥 Processing <name>…"; row + `asyncio.to_thread(ingest_document_sync, ...)` awaited inline (bot user waits for the ✅ — simpler than task tracking); reply "✅ N chunks from <name> are searchable in <workspace>" or ❌ with short reason. Sets `document_id` metadata via the sync path.
  - `on_text` — if `pending_action == "awaiting_content"` and len(text) ≥ 120 → `ingest_text_sync` into current workspace → "✅ N chunks indexed from your notes — ask away." and clear pending; else → `run_chat` (access check `can_view`); reply `format_answer(...)` split into chunks; refused answers reply without sources block.
  - Access refusals reply: private workspace → "🔒 This workspace is private — ask its owner for access."; no workspace selected → "Pick a workspace first — /workspaces".
  - Every handler wrapped: `try/except` logs `logger.exception` and replies "Something went wrong — please try again." (helper decorator `safe_handler`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bot_handlers.py
import pytest
from types import SimpleNamespace

from app.config import Settings
from app.models import User, Workspace
from app.services.chat import start_conversation


class Recorder:
    def __init__(self):
        self.texts = []
        self.markups = []

    async def __call__(self, text, parse_mode=None, reply_markup=None):
        self.texts.append(text)
        self.markups.append(reply_markup)


def fake_update(*, text=None, document=None, callback_data=None, tg_id=42,
                username="kartik", first="Kartik"):
    msg = SimpleNamespace(text=text, reply_text=Recorder(),
                          document=SimpleNamespace(
                              file_name=document["file_name"], file_size=document.get("file_size", 10))
                          if document else None,
                          chat=SimpleNamespace(id=tg_id))
    effective_message = msg
    cq = SimpleNamespace(data=callback_data, answer=lambda: None,
                         message=SimpleNamespace(reply_text=Recorder())) if callback_data else None
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=tg_id, username=username, first_name=first),
        effective_message=effective_message, effective_chat=SimpleNamespace(id=tg_id),
        callback_query=cq, message=msg)


class FakeGraph:
    def invoke(self, state, config=None):
        return {"answer": "Answer [1]", "sources": [{"file": "a.md", "section": None}],
                "refused": None, "history": []}


@pytest.fixture
def env(session, test_settings):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    ws = create_workspace(session, owner, "Bot WS", is_private=False)
    session.commit()
    container = SimpleNamespace(settings=test_settings, session_factory=None, graph=FakeGraph())
    ctx = SimpleNamespace(bot_data={"container": container}, user_data={},
                          args=[], application=SimpleNamespace(bot_data={"container": container}))
    # handlers open sessions via context; monkeypatch factory to return the test session
    container.session_factory = lambda: _CtxSession(session)
    return session, ws, ctx


class _CtxSession:
    def __init__(self, session):
        self._s = session

    def __enter__(self):
        return self._s

    def __exit__(self, *a):
        self._s.rollback()  # don't leak test writes between handlers
        return False


@pytest.mark.asyncio
async def test_start_registers_user_and_lists(env):
    from app.bot.handlers import cmd_start

    session, ws, ctx = env
    await cmd_start(fake_update(text="/start"), ctx)
    user = session.query(User).filter_by(telegram_id=42).one()
    assert user.display_name == "Kartik"


@pytest.mark.asyncio
async def test_newworkspace_command(env):
    from app.bot.handlers import cmd_newworkspace

    session, ws, ctx = env
    ctx.args = ["My", "Team", "Space", "private"]
    upd = fake_update(text="/newworkspace My Team Space private")
    await cmd_newworkspace(upd, ctx)
    created = session.query(Workspace).filter_by(slug="my-team-space").one()
    assert created.is_private is True
    replies = " ".join(upd.effective_message.reply_text.texts)
    assert "created" in replies.lower() and "My Team Space" in replies


@pytest.mark.asyncio
async def test_text_question_answered_with_sources(env):
    from app.bot.handlers import on_text
    from app.services.users import get_or_create_user
    from app.services.chat import start_conversation

    session, ws, ctx = env
    user = get_or_create_user(session, ctx.bot_data["container"].settings, telegram_id=42)
    start_conversation(session, user, ws)
    session.commit()
    upd = fake_update(text="What is the policy?")
    await on_text(upd, ctx)
    assert "Answer [1]" in upd.effective_message.reply_text.texts[0]
    assert "Sources:" in upd.effective_message.reply_text.texts[0]


@pytest.mark.asyncio
async def test_private_workspace_hides_text(env):
    from app.bot.handlers import on_text
    from app.services.workspaces import create_workspace
    from app.services.users import get_or_create_user, ensure_system_user

    session, ws, ctx = env
    owner = ensure_system_user(session)
    priv = create_workspace(session, owner, "Secret", is_private=True)
    user = get_or_create_user(session, ctx.bot_data["container"].settings, telegram_id=42)
    from app.services.chat import start_conversation

    start_conversation(session, user, priv)
    session.commit()
    upd = fake_update(text="hello")
    await on_text(upd, ctx)
    assert "private" in upd.effective_message.reply_text.texts[0].lower()


@pytest.mark.asyncio
async def test_demo_switches_workspace(env):
    from app.bot import handlers

    session, ws, ctx = env
    # create the demo workspace the command looks for
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace
    from app.models import Workspace as W

    demo = create_workspace(session, ensure_system_user(session), "Demo Delivery Policy")
    demo.slug = handlers.DEMO_SLUG
    session.commit()
    upd = fake_update(text="/demo")
    await handlers.cmd_demo(upd, ctx)
    assert upd.effective_message.reply_text.markups[0] is not None  # demo question buttons
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bot_handlers.py -v`
Expected: FAIL — `No module named 'app.bot.handlers'`

- [ ] **Step 3: Implement**

```python
# app/bot/handlers.py
import asyncio
import logging

from telegram import Update
from telegram.ext import ContextTypes

from app.bot.format import esc, format_answer, split_message
from app.bot.keyboards import (
    DEMO_QUESTIONS, conversations_keyboard, demo_keyboard, new_chat_keyboard,
    workspace_keyboard,
)
from app.models import Document, User as UserModel, WorkspaceMember
from app.security import can_ingest, can_manage, can_view, visible_workspaces
from app.services.chat import list_conversations, resume_conversation, run_chat, start_conversation
from app.services.ingest import SUPPORTED_EXTS, ingest_document_sync, ingest_text_sync
from app.services.users import get_or_create_user
from app.services.workspaces import create_workspace, workspace_by_slug, workspace_stats

logger = logging.getLogger(__name__)
DEMO_SLUG = "delivery-policy"  # matches seed.SEED_WORKSPACES
PASTE_MIN_CHARS = 120
MAX_BYTES = 20 * 1024 * 1024


def _container(context: ContextTypes.DEFAULT_TYPE):
    return context.application.bot_data["container"]


def safe_handler(fn):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            return await fn(update, context)
        except Exception:  # noqa: BLE001
            logger.exception("handler %s failed", fn.__name__)
            target = update.effective_message
            if target:
                await target.reply_text("Something went wrong — please try again.")

    wrapper.__name__ = fn.__name__
    return wrapper


def _user_and_settings(context, session, update):
    container = _container(context)
    tg = update.effective_user
    user = get_or_create_user(session, container.settings, telegram_id=tg.id,
                              username=tg.username, display_name=tg.first_name)
    return user, container.settings


def _current_workspace(session, user):
    if user.current_workspace_id:
        return session.get(Workspace, user.current_workspace_id)
    return None
```

```python
# continued app/bot/handlers.py
@safe_handler
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with _container(context).session_factory() as session:
        user, _settings = _user_and_settings(context, session, update)
        session.commit()
        wss = visible_workspaces(session, user)
        current = _current_workspace(session, user)
        if current:
            text = (f"Hi {esc(user.display_name)}! I answer questions from document workspaces.\n"
                    f"Current workspace: <b>{esc(current.name)}</b>")
        else:
            text = f"Hi {esc(user.display_name)}! Pick a workspace below, or /newworkspace to create one."
        await update.effective_message.reply_text(
            text, parse_mode="HTML",
            reply_markup=workspace_keyboard(wss, current.id if current else None))
```

Remaining handlers (same file; each `@safe_handler`, each opens `with _container(context).session_factory() as session:`):

```python
@safe_handler
async def cmd_help(update, context):
    await update.effective_message.reply_text(
        "<b>Commands</b>\n"
        "/workspaces — list workspaces\n"
        "/workspace — current workspace info\n"
        "/newworkspace &lt;name&gt; [private] — create a workspace\n"
        "/new — new conversation (optionally add content)\n"
        "/resume — past conversations\n"
        "/demo — try the seeded policy workspace\n"
        "/invite @user — add a member (private workspaces)\n"
        "/public /private — toggle privacy (owner)\n"
        "Send a PDF/DOCX/PPTX/XLSX/HTML/MD file to index it. Or just ask a question.",
        parse_mode="HTML")


@safe_handler
async def cmd_workspaces(update, context):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        session.commit()
        wss = visible_workspaces(session, user)
        if not wss:
            await update.effective_message.reply_text("No workspaces yet — /newworkspace to create one.")
            return
        await update.effective_message.reply_text(
            "Your workspaces:", reply_markup=workspace_keyboard(wss, user.current_workspace_id))


@safe_handler
async def cmd_workspace(update, context):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        if ws is None:
            await update.effective_message.reply_text("Pick a workspace first — /workspaces")
            return
        stats = workspace_stats(session, ws)
        await update.effective_message.reply_text(
            f"<b>{esc(ws.name)}</b> (<code>{esc(ws.slug)}</code>)\n"
            f"Documents: {stats['documents']} · Chunks: {stats['chunks']} · "
            f"{'🔒 private' if ws.is_private else '🌍 public'}", parse_mode="HTML")


@safe_handler
async def cmd_newworkspace(update, context):
    parts = list(context.args or [])
    is_private = bool(parts) and parts[-1].lower() == "private"
    if is_private:
        parts = parts[:-1]
    name = " ".join(parts).strip()
    if not name or len(name) > 40:
        await update.effective_message.reply_text("Usage: /newworkspace <name> [private] (name ≤ 40 chars)")
        return
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = create_workspace(session, user, name, is_private=is_private)
        start_conversation(session, user, ws)
        session.commit()
        await update.effective_message.reply_text(
            f"✅ Workspace <b>{esc(ws.name)}</b> created ({'🔒' if ws.is_private else '🌍'}). "
            "You're in — send a file or /new.", parse_mode="HTML")


@safe_handler
async def cmd_new(update, context):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        if ws is None:
            await update.effective_message.reply_text("Pick a workspace first — /workspaces")
            return
        conv = start_conversation(session, user, ws)
        session.commit()
        await update.effective_message.reply_text(
            f"🆕 New conversation in <b>{esc(ws.name)}</b>.\n"
            "Add knowledge? Paste content (120+ chars) or send a file — or just ask questions.",
            parse_mode="HTML", reply_markup=new_chat_keyboard())


@safe_handler
async def cmd_resume(update, context):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        convos = list_conversations(session, user)
        session.commit()
        if not convos:
            await update.effective_message.reply_text("No past conversations yet.")
            return
        await update.effective_message.reply_text(
            "📜 Your recent conversations:", reply_markup=conversations_keyboard(convos))


@safe_handler
async def cmd_demo(update, context):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = workspace_by_slug(session, DEMO_SLUG)
        if ws is None:
            await update.effective_message.reply_text(
                "Demo workspace isn't seeded yet. Run: python3 scripts/seed.py")
            return
        start_conversation(session, user, ws)
        session.commit()
        await update.effective_message.reply_text(
            f"🧪 Demo workspace <b>{esc(ws.name)}</b> ready. Tap a question:", parse_mode="HTML",
            reply_markup=demo_keyboard())


@safe_handler
async def cmd_invite(update, context):
    username = ((context.args or [""])[0]).lstrip("@").strip()
    if not username:
        await update.effective_message.reply_text("Usage: /invite @username")
        return
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        target = session.query(UserModel).filter(UserModel.telegram_username == username).one_or_none()
        if ws is None or target is None:
            await update.effective_message.reply_text("They need to /start the bot first, and you must be in a workspace.")
            return
        from app.security import role_of

        role = role_of(session, ws, user)
        if not ws.is_private:
            await update.effective_message.reply_text("This workspace is public — everyone can already use it.")
            return
        if role not in ("owner", "editor"):
            await update.effective_message.reply_text("Only owners/editors can invite.")
            return
        exists = session.get(WorkspaceMember, (ws.id, target.id))
        if exists:
            await update.effective_message.reply_text(f"@{username} is already a member.")
            return
        session.add(WorkspaceMember(workspace_id=ws.id, user_id=target.id, role="viewer"))
        session.commit()
        await update.effective_message.reply_text(f"✅ @{username} added to <b>{esc(ws.name)}</b>.", parse_mode="HTML")


@safe_handler
async def cmd_public(update, context):
    await _set_privacy(update, context, False)


@safe_handler
async def cmd_private(update, context):
    await _set_privacy(update, context, True)


async def _set_privacy(update, context, private: bool):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        from app.security import role_of

        if ws is None or not can_manage(ws, role_of(session, ws, user)):
            await update.effective_message.reply_text("Only the workspace owner can do that.")
            return
        ws.is_private = private
        session.commit()
        await update.effective_message.reply_text(
            f"<b>{esc(ws.name)}</b> is now {'🔒 private' if private else '🌍 public'}.", parse_mode="HTML")


@safe_handler
async def on_callback(update, context):
    query = update.callback_query
    await query.answer()
    prefix, _, value = query.data.partition(":")
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        if prefix == "ws":
            ws = session.get(Workspace, value)
            if ws is None or not can_view(ws, _role(session, ws, user)):
                await query.message.reply_text("🔒 That workspace is private — ask its owner for access.")
                return
            start_conversation(session, user, ws)
            session.commit()
            stats = workspace_stats(session, ws)
            await query.message.reply_text(
                f"✅ Switched to <b>{esc(ws.name)}</b> ({stats['documents']} docs). Ask away!",
                parse_mode="HTML")
        elif prefix == "cv":
            conv = resume_conversation(session, user, value)
            session.commit()
            if conv is None:
                await query.message.reply_text("That conversation isn't available.")
                return
            await query.message.reply_text(f"📜 Resumed <b>{esc(conv.title)}</b> — continue asking.",
                                           parse_mode="HTML")
        elif prefix == "nc":
            if value == "content":
                user.pending_action = "awaiting_content"
                session.commit()
                await query.message.reply_text(
                    f"Send me the content as your next message ({PASTE_MIN_CHARS}+ chars) — "
                    "I'll chunk and index it.")
            else:
                await query.message.reply_text("OK — just type your question.")
        elif prefix == "dq":
            ws = workspace_by_slug(session, DEMO_SLUG)
            question = DEMO_QUESTIONS[int(value)]
            start_conversation(session, user, ws)
            result = await asyncio.to_thread(run_chat, session, user, ws, question,
                                             _container(context).graph)
            session.commit()
            for part in split_message(format_answer(result.answer, result.sources)):
                await query.message.reply_text(part, parse_mode="HTML")


def _role(session, ws, user):
    from app.security import role_of

    return role_of(session, ws, user)


@safe_handler
async def on_document(update, context):
    doc = update.effective_message.document
    name = doc.file_name or "file"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    with _container(context).session_factory() as session:
        user, settings = _user_and_settings(context, session, update)
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        if ws is None:
            await update.effective_message.reply_text("Pick a workspace first — /workspaces")
            return
        if not can_ingest(ws, _role(session, ws, user)):
            await update.effective_message.reply_text("🔒 You can't add files to this private workspace.")
            return
        if ext not in SUPPORTED_EXTS:
            await update.effective_message.reply_text(
                f"Unsupported type .{ext or '?'} — send one of: {', '.join(sorted(SUPPORTED_EXTS))}")
            return
        if (doc.file_size or 0) > MAX_BYTES:
            await update.effective_message.reply_text("File is over the 20 MB Telegram limit.")
            return
        from app.services.ingest import create_document_row

        row = create_document_row(session, ws, name, source="telegram", uploaded_by=user.id,
                                  byte_size=doc.file_size or 0)
        session.commit()
        slug, doc_id = ws.slug, row.id
    await update.effective_message.reply_text(f"📥 Processing {esc(name)}…", parse_mode="HTML")
    tg_file = await context.bot.get_file(doc.file_id)
    data = await tg_file.download_as_bytearray()
    from app.rag.vectorstore import get_store

    store = get_store(settings, slug)
    try:
        def _work():
            return ingest_document_sync(_container(context).session_factory, doc_id,
                                        bytes(data), ext, store, settings)
        chunks = await asyncio.to_thread(_work)
    except Exception as e:  # noqa: BLE001
        logger.exception("ingest failed for %s", name)
        await update.effective_message.reply_text(
            f"❌ Couldn't index {esc(name)}: {esc(str(e)[:200])}", parse_mode="HTML")
        return
    await update.effective_message.reply_text(
        f"✅ {chunks} chunks from {esc(name)} are searchable.", parse_mode="HTML")


@safe_handler
async def on_text(update, context):
    text = (update.effective_message.text or "").strip()
    if not text:
        return
    with _container(context).session_factory() as session:
        user, settings = _user_and_settings(context, session, update)
        if user.pending_action == "awaiting_content":
            ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
            if ws is None:
                await update.effective_message.reply_text("Pick a workspace first — /workspaces")
                return
            if len(text) < PASTE_MIN_CHARS:
                user.pending_action = None
                session.commit()
                await update.effective_message.reply_text(
                    f"That was under {PASTE_MIN_CHARS} chars, so I treated it as a question.")
            else:
                from app.rag.vectorstore import get_store

                store = get_store(settings, ws.slug)

                def _work():
                    return ingest_text_sync(_container(context).session_factory, ws,
                                            f"Notes {update.effective_message.message_id}", text,
                                            store, uploaded_by=user.id)
                try:
                    row = await asyncio.to_thread(_work)
                except ValueError as e:
                    await update.effective_message.reply_text(str(e))
                    return
                user.pending_action = None
                session.commit()
                await update.effective_message.reply_text(
                    f"✅ {row.chunk_count} chunks indexed from your notes — ask away.")
                return
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        if ws is None:
            await update.effective_message.reply_text("Pick a workspace first — /workspaces")
            return
        if not can_view(ws, _role(session, ws, user)):
            await update.effective_message.reply_text("🔒 This workspace is private — ask its owner for access.")
            return
        result = await asyncio.to_thread(run_chat, session, user, ws, text,
                                         _container(context).graph)
        session.commit()
    sources = [] if result.refused else result.sources
    for part in split_message(format_answer(result.answer, sources)):
        await update.effective_message.reply_text(part, parse_mode="HTML")
```

(`session.query(...)` works because `Session` exposes it; prefer `session.scalar(select(...))` where convenient. The listings show `query` only where brevity helps.)

- [ ] **Step 4: Run to verify pass**

Run: `python3 -m pytest tests/test_bot_handlers.py -v`
Expected: 6 PASS (plus the adjusted creation assertion)

---

### Task 14: `app/bot/setup.py` + full `app/main.py` lifespan

**Files:**
- Create: `app/bot/setup.py`
- Modify: `app/main.py` (replace Task 11's minimal version's lifespan body)
- Test: covered by `tests/test_routers.py` (disabled mode) + `tests/test_bot_handlers.py`; add `test_webhook_passes_update` to `tests/test_routers.py`

**Interfaces:**
- Produces (`setup.py`): `Container` dataclass (`settings`, `session_factory`, `graph`), `build_application(settings, session_factory, graph) -> Application` (registers: CommandHandlers for start/help/workspaces/workspace/newworkspace/new/resume/demo/invite/public/private; `CallbackQueryHandler(on_callback)`; `MessageHandler(filters.Document.ALL, on_document)`; `MessageHandler(filters.TEXT & ~filters.COMMAND, on_text)`), `start_polling(app)`, `stop_application(app)`.
- `main.py` lifespan (final form):
  1. `init_db(settings)`; `ensure_system_user` + commit.
  2. `app.state.graph = get_graph(settings)` **only if `groq_api_key` is set**; otherwise log a warning and leave graph unset (endpoints will 500 with a clear log — same as today's behavior without a key).
  3. `bot_mode`:
     - `disabled` (or token empty): log "bot disabled", `ptb_app = None`.
     - `polling`: build PTB app, `await initialize/start/updater.start_polling()`, store on `app.state.ptb_app`.
     - `webhook`: build PTB app, `await initialize()` (no polling; Telegram POSTs to the router), if `webhook_url` set → `bot.set_webhook(f"{webhook_url}/telegram/webhook", secret_token=settings.telegram_webhook_secret, allowed_updates=["message", "callback_query"])`.
  4. On shutdown: stop PTB (`stop_application`), dispose engine, close graph's psycopg pool if it exists (attribute `graph._production` is module-level; expose `app.rag.graph.close_production()` that closes `_production`'s pool — add that tiny helper in this task).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_routers.py — append
@pytest.mark.asyncio
async def test_webhook_passes_update_to_ptb(client, monkeypatch):
    c, s = client
    from app.main import create_app

    app = create_app(s)

    class FakePTB:
        def __init__(self):
            self.updates = []
            self.bot = None

        async def process_update(self, update):
            self.updates.append(update)

    app.state.ptb_app = FakePTB()
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as c2:
        r = c2.post("/telegram/webhook", json={"update_id": 1},
                    headers={"X-Telegram-Bot-Api-Secret-Token": s.telegram_webhook_secret})
    assert r.status_code == 200
    assert app.state.ptb_app.updates[0].update_id == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_routers.py::test_webhook_passes_update_to_ptb -v`
Expected: FAIL — `Update` object lacks `update_id` (de_json returns None on missing required fields → assertion error), or PTB import paths missing in setup module. Note: if `Update.de_json({"update_id": 1})` already yields a valid Update and the current webhook route works, this test may pass already — then run all router tests and continue to Step 3 (the setup module itself is still needed).

- [ ] **Step 3: Implement `app/bot/setup.py`**

```python
# app/bot/setup.py
import logging
from dataclasses import dataclass

from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters,
)

from app.bot import handlers
from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    session_factory: object
    graph: object


def build_application(settings: Settings, session_factory=None, graph=None) -> Application:
    builder = Application.builder().token(settings.telegram_bot_token)
    if settings.bot_mode == "webhook":
        builder = builder.updater(None)  # webhook mode never polls
    app = builder.build()

    from app.db import SessionLocal

    if session_factory is None:
        session_factory = SessionLocal(settings)
    app.bot_data["container"] = Container(settings, session_factory, graph)

    cmds = [
        ("start", handlers.cmd_start), ("help", handlers.cmd_help),
        ("workspaces", handlers.cmd_workspaces), ("workspace", handlers.cmd_workspace),
        ("newworkspace", handlers.cmd_newworkspace), ("new", handlers.cmd_new),
        ("resume", handlers.cmd_resume), ("demo", handlers.cmd_demo),
        ("invite", handlers.cmd_invite), ("public", handlers.cmd_public),
        ("private", handlers.cmd_private),
    ]
    for name, fn in cmds:
        app.add_handler(CommandHandler(name, fn))
    app.add_handler(CallbackQueryHandler(handlers.on_callback))
    app.add_handler(MessageHandler(filters.Document.ALL, handlers.on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.on_text))
    return app


async def start_polling(app: Application) -> None:
    await app.initialize()
    await app.start()
    await app.updater.start_polling(allowed_updates=["message", "callback_query"])
    logger.info("telegram polling started")


async def stop_application(app: Application) -> None:
    if app.updater:
        await app.updater.stop()
    await app.stop()
    await app.shutdown()
```

- [ ] **Step 4: Final `app/main.py`**

```python
# app/main.py
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.db import SessionLocal, init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    init_db(settings)
    with SessionLocal(settings)() as session:
        from app.services.users import ensure_system_user

        ensure_system_user(session)
        session.commit()

    if settings.groq_api_key:
        from app.rag.graph import get_graph

        app.state.graph = get_graph(settings)
    else:
        logger.warning("GROQ_API_KEY not set — chat endpoints will fail until it is provided")

    app.state.ptb_app = None
    if app.state.graph is None and settings.bot_mode != "disabled":
        logger.warning("bot not started: GROQ_API_KEY missing — the bot needs the RAG graph")
    elif settings.bot_mode != "disabled" and settings.telegram_bot_token:
        from app.bot.setup import build_application, start_polling, stop_application

        ptb = build_application(settings, SessionLocal(settings), app.state.graph)
        if settings.bot_mode == "polling":
            await start_polling(ptb)
        elif settings.bot_mode == "webhook":
            await ptb.initialize()
            if settings.webhook_url:
                await ptb.bot.set_webhook(
                    f"{settings.webhook_url.rstrip('/')}/telegram/webhook",
                    secret_token=settings.telegram_webhook_secret,
                    allowed_updates=["message", "callback_query"])
        else:
            ptb = None
        app.state.ptb_app = ptb
    else:
        logger.info("bot disabled (BOT_MODE=%s, token set=%s)",
                    settings.bot_mode, bool(settings.telegram_bot_token))
    try:
        yield
    finally:
        if app.state.ptb_app is not None:
            from app.bot.setup import stop_application

            await stop_application(app.state.ptb_app)
        from app.rag.graph import close_production

        close_production()
        from app.db import engine

        engine(settings).dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(title="Telegram Multi-Workspace RAG", lifespan=lifespan)
    app.state.settings = s
    from app.security import api_key_ok

    app.state.security_api_key_ok = lambda presented: api_key_ok(s, presented)

    from app.routers import chat, documents, health, telegram_webhook, workspaces

    app.include_router(health.router)
    app.include_router(workspaces.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(telegram_webhook.router)
    return app


app = create_app()
```

Add to `app/rag/graph.py` (the production-singleton section): store the pool in a module global when `get_graph` builds it, and close it on shutdown:

```python
_production_pool = None


def close_production() -> None:
    global _production, _production_pool
    if _production_pool is not None:
        _production_pool.close()
    _production = None
    _production_pool = None
```

And inside `get_graph`, after creating the pool: `_production_pool = pool` (alongside the existing `global _production` declaration — change it to `global _production, _production_pool`).

- [ ] **Step 5: Run everything**

Run: `python3 -m pytest tests/ -v`
Expected: ALL PASS (≈55 tests)

---

### Task 15: Seed, deployment files, README, cleanup, smoke test

**Files:**
- Create: `seed.py`, `scripts/seed.py`, `scripts/__init__.py`, `.env.example` (from Task 0 — finalize), `docker-compose.yml`, `Dockerfile`
- Modify: `README.md` (rewrite)
- Delete: `main.py`, `rag_engine.py`
- Test: `tests/test_seed.py` (integration)

**Interfaces:**
- `seed.py` module: `SEED_WORKSPACES: list[tuple[str, str, list[str]]]` (slug, name, files), `DEMO_SLUG = "delivery-policy"` (imported by bot handlers — keep single source in `seed.py` and have `bot.handlers` import it), `seed_all(corpus_dir: str, settings: Settings, notify=None) -> dict[str, int]` — creates system user; for each workspace: skip if slug exists (but still ingest missing files); for each file: skip if a `ready` document with same filename exists in that workspace; read bytes from `corpus_dir`, `parse_file`, add to per-workspace store, mark rows ready. Returns `{slug: chunk_count}`. Source=`seed`.
- `scripts/seed.py`: argparse (`--corpus` default `my_docs_folder`), prints progress and totals.

Seed data:

```python
SEED_WORKSPACES = [
    ("delivery-policy", "Delivery Policy", [
        "delivery_policy.md", "delivery_zones_and_timeframes.pdf",
        "order_tracking_and_status.pptx", "public_user_delivery_terms.html",
        "special_goods_delivery.html",
    ]),
    ("returns-and-refunds", "Returns and Refunds", ["returns_and_refunds.md"]),
    ("shipping-charges", "Shipping Charges", [
        "shipping_charges_matrix.docx", "delivery_sla_matrix.xlsx",
    ]),
]
DEMO_SLUG = "delivery-policy"
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_seed.py
import pytest

from seed import SEED_WORKSPACES, seed_all

MD = b"# H1\n\n" + b"Some chunk body content long enough. " * 40


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "delivery_policy.md").write_bytes(MD)
    return str(d)


@pytest.fixture
def fake_store(monkeypatch):
    added = []

    class S:
        def add_documents(self, docs):
            added.extend(docs)

    monkeypatch.setattr("seed._store_for", lambda settings, slug: S())
    return added


@pytest.mark.integration
def test_seed_all_creates_workspaces_and_is_idempotent(
        session, test_settings, corpus, fake_store, monkeypatch):
    # use the live test session factory so rows are visible
    from app.db import SessionLocal

    monkeypatch.setattr("seed._session_factory", lambda settings: (lambda: _Fixed(session)))
    counts = seed_all(corpus, test_settings)
    assert set(counts) == {"delivery-policy", "returns-and-refunds", "shipping-charges"}
    from app.services.workspaces import workspace_by_slug

    ws = workspace_by_slug(session, "delivery-policy")
    assert ws is not None
    again = seed_all(corpus, test_settings)
    assert all(v == 0 for v in again.values())  # idempotent: nothing re-ingested


class _Fixed:
    def __init__(self, s):
        self._s = s

    def __enter__(self):
        return self._s

    def __exit__(self, *a):
        return False
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_seed.py -v`
Expected: FAIL — `No module named 'seed'`

- [ ] **Step 3: Implement `seed.py` + `scripts/seed.py`**

```python
# seed.py
import logging
from pathlib import Path

from sqlalchemy import select

from app.config import Settings
from app.models import Document
from app.services.ingest import create_document_row, parse_file
from app.services.users import ensure_system_user
from app.services.workspaces import create_workspace, workspace_by_slug

logger = logging.getLogger(__name__)

SEED_WORKSPACES = [
    ("delivery-policy", "Delivery Policy", [
        "delivery_policy.md", "delivery_zones_and_timeframes.pdf",
        "order_tracking_and_status.pptx", "public_user_delivery_terms.html",
        "special_goods_delivery.html",
    ]),
    ("returns-and-refunds", "Returns and Refunds", ["returns_and_refunds.md"]),
    ("shipping-charges", "Shipping Charges", [
        "shipping_charges_matrix.docx", "delivery_sla_matrix.xlsx",
    ]),
]
DEMO_SLUG = "delivery-policy"


def _store_for(settings: Settings, slug: str):
    from app.rag.vectorstore import get_store

    return get_store(settings, slug)


def _session_factory(settings: Settings):
    from app.db import SessionLocal

    return SessionLocal(settings)


def seed_all(corpus_dir: str, settings: Settings, notify=None) -> dict[str, int]:
    counts: dict[str, int] = {}
    corpus = Path(corpus_dir)
    factory = _session_factory(settings)
    for slug, name, files in SEED_WORKSPACES:
        with factory() as session:
            system = ensure_system_user(session)
            ws = workspace_by_slug(session, slug)
            if ws is None:
                ws = create_workspace(session, system, name)
            done = {d.filename for d in session.scalars(
                select(Document).filter_by(workspace_id=ws.id, status="ready"))}
            total = 0
            for filename in files:
                if filename in done:
                    continue
                path = corpus / filename
                if not path.exists():
                    logger.warning("seed: missing %s", path)
                    continue
                data = path.read_bytes()
                chunks = parse_file(data, path.suffix.lstrip("."))
                row = create_document_row(session, ws, filename, source="seed",
                                          byte_size=len(data))
                for c in chunks:
                    c.metadata["document_id"] = row.id
                    c.metadata["source_file"] = filename
                _store_for(settings, slug).add_documents(chunks)
                row.status, row.chunk_count = "ready", len(chunks)
                session.commit()
                total += len(chunks)
                if notify:
                    notify(f"{slug}: +{len(chunks)} chunks from {filename}")
            session.commit()
        counts[slug] = total
    return counts
```

```python
# scripts/seed.py
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from seed import seed_all  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Seed demo workspaces")
    parser.add_argument("--corpus", default="my_docs_folder")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    if not settings.groq_api_key:
        raise SystemExit("GROQ_API_KEY missing in .env")

    counts = seed_all(args.corpus, settings, notify=print)
    print("\nSeeded chunk totals:")
    for slug, n in counts.items():
        print(f"  {slug}: {n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the real seed**

Run: `python3 scripts/seed.py --corpus my_docs_folder`
Expected: progress lines; totals > 0 for all three workspaces (docling parses the PDF/PPTX/DOCX/XLSX/HTML files — first parse is slow while docling warms up). Verify: `psql -d langchain -c "SELECT slug, (SELECT count(*) FROM documents d WHERE d.workspace_id=w.id) FROM workspaces w;"`

- [ ] **Step 5: Smoke-test the RAG end-to-end against real Groq**

```bash
python3 - <<'EOF'
from app.config import get_settings
from app.db import SessionLocal
from app.services.users import ensure_system_user
from app.services.workspaces import workspace_by_slug
from app.services.chat import run_chat
from app.rag.graph import get_graph

s = get_settings()
with SessionLocal(s)() as session:
    user = ensure_system_user(session)
    ws = workspace_by_slug(session, "delivery-policy")
    result = run_chat(session, user, ws, "What is the delivery SLA?", get_graph(s), s)
    print("ANSWER:", result.answer[:400])
    print("SOURCES:", result.sources)
    assert result.sources, "expected citations"
EOF
```

Expected: an answer with real content + `Sources: [...]` listing files from the delivery-policy workspace.

- [ ] **Step 6: Smoke-test the bot wiring without a token (dry run)**

Run: `timeout 8 python3 -c "import uvicorn, app.main; uvicorn.run(app.main.app, host='127.0.0.1', port=8901, log_level='warning')" 2>&1 | head -5; curl -s localhost:8901/health 2>/dev/null || true`

Expected: server starts, logs `bot disabled`, `/health` returns `{"status":"ok","bot_mode":"disabled"}` (the `timeout` exit code 124 is expected). If a Telegram token is available, add it to `.env`, set `BOT_MODE=polling`, restart, and send `/start` → `/demo` in Telegram as the final manual check.

- [ ] **Step 7: Deployment files**

```yaml
# docker-compose.yml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: langchain
      POSTGRES_PASSWORD: langchain
      POSTGRES_DB: langchain
    volumes: ["pgdata:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U langchain"]
      interval: 5s
      retries: 10
  app:
    build: .
    depends_on:
      db:
        condition: service_healthy
    env_file: .env
    environment:
      DATABASE_URL: postgresql+psycopg://langchain:langchain@db:5432/langchain
    ports: ["8080:8080"]
    volumes: ["./data:/app/data", "./my_docs_folder:/app/my_docs_folder:ro"]
volumes:
  pgdata: {}
```

```dockerfile
# Dockerfile
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 8: README rewrite** — sections: What it is; Architecture diagram (text); Setup (prereqs incl. `brew install pgvector`, DB setup SQL from Task 0, pip install, .env from `.env.example`); Seed; Run dev (`BOT_MODE=disabled` API-only, `polling` with a BotFather token, `webhook` + setWebhook for prod incl. `secret_token`); REST examples (curl for each endpoint with `X-API-Key`); Telegram command reference (all commands incl. `/new`, `/resume`, `/demo`); How citations work; Troubleshooting (pgvector missing, Groq key, docling first-parse slowness).

- [ ] **Step 9: Cleanup + final verification**

```bash
grep -q '^data/' .gitignore || echo 'data/' >> .gitignore
git rm --cached main.py rag_engine.py 2>/dev/null; rm main.py rag_engine.py
python3 -m pytest tests/ -q
python3 scripts/seed.py --corpus my_docs_folder   # second run: all zero (idempotent)
```

Expected: full suite green; seed re-run re-ingests nothing. Confirm `git status` shows staged/unstaged changes only — **do NOT commit**. Also update `bot/handlers.py` to import `DEMO_SLUG` from `seed.py` (single source of truth) and re-run the bot tests.

Final `git status` check only — no `git add`, no `git commit`, no `git push` (Global Constraints).


