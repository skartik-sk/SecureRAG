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
