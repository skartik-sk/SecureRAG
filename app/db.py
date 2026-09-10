from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.models import Base

_ENGINES: dict = {}


def engine(settings: Settings):
    key = settings.database_url
    if key not in _ENGINES:
        kwargs: dict = {"pool_pre_ping": True}
        if settings.database_url.startswith("postgresql"):
            # Many serverless instances share one Postgres — keep pools small
            # and recycle before provider idle-timeouts (~5 min) kick in.
            kwargs.update(pool_size=2, max_overflow=3, pool_recycle=280)
        _ENGINES[key] = create_engine(settings.database_url, **kwargs)
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
