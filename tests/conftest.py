import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import engine, init_db
from app.models import Base

TEST_DB_URL = "postgresql+psycopg://langchain:langchain@localhost:5432/rag_test"


def _make_settings() -> Settings:
    return Settings(_env_file=None, database_url=TEST_DB_URL, upload_dir="data/test_uploads",
                    api_keys="dev-api-key-1", telegram_webhook_secret="dev-secret-change-me",
                    embeddings_api_key="test-embeddings-key")


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
