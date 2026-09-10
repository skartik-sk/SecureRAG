"""Hybrid retrieval: lexical (Postgres full-text search, BM25-style ranking)
fused with pgvector dense search via Reciprocal Rank Fusion, then MMR
diversification before the LLM relevance-grading rerank in the graph."""

import math

from langchain_core.documents import Document
from sqlalchemy import text as sql_text

from app.config import Settings

_RRF_K = 60
_MMR_LAMBDA = 0.7


def _doc_key(doc: Document):
    return doc.page_content


def lexical_search(settings: Settings, slug: str, query: str, k: int) -> list[Document]:
    """BM25-style lexical ranking over chunk text with Postgres full-text search."""
    sql = sql_text("""
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = :collection
          AND to_tsvector('english', e.document) @@ websearch_to_tsquery('english', :q)
        ORDER BY ts_rank_cd(to_tsvector('english', e.document),
                            websearch_to_tsquery('english', :q)) DESC
        LIMIT :k
    """)
    from app.db import engine
    from app.rag.vectorstore import collection_name

    with engine(settings).connect() as conn:
        rows = conn.execute(sql, {"collection": collection_name(slug),
                                  "q": query, "k": k}).all()
    return [Document(page_content=doc, metadata=meta or {})
            for doc, meta in rows]


def rrf_fuse(rankings: list[list]) -> list:
    """Reciprocal Rank Fusion. `rankings` is a list of ordered [(key, doc)];
    returns [(key, doc)] fused across lists, best first."""
    scores: dict = {}
    first: dict = {}
    for ranking in rankings:
        for pos, (key, doc) in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + pos + 1)
            first.setdefault(key, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [(key, first[key]) for key in ordered]


def cosine(a: list[float], b: list[float]) -> float:
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    return dot / math.sqrt(na * nb) if na and nb else 0.0


def mmr_select(query_vec: list[float], candidates: list, k: int,
               lambda_mult: float = _MMR_LAMBDA) -> list:
    """Greedy Maximal Marginal Relevance. `candidates` is [(key, doc, vec)];
    returns up to k candidates balancing relevance to the query against
    similarity to already-selected candidates."""
    selected, remaining = [], list(candidates)
    while len(selected) < k and remaining:
        best = max(
            remaining,
            key=lambda c: lambda_mult * cosine(query_vec, c[2])
            - (1 - lambda_mult) * max((cosine(c[2], s[2]) for s in selected), default=0.0),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def _fetch_embeddings(settings: Settings, slug: str, contents: list[str]) -> dict:
    sql = sql_text("""
        SELECT e.document, e.embedding::text
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON c.uuid = e.collection_id
        WHERE c.name = :collection AND e.document = ANY(:texts)
    """)
    from app.db import engine
    from app.rag.vectorstore import collection_name

    def _parse(v: str) -> list[float]:
        return [float(x) for x in v.strip("[]").split(",")]

    with engine(settings).connect() as conn:
        rows = conn.execute(sql, {"collection": collection_name(slug),
                                  "texts": list(set(contents))}).all()
    return {doc: _parse(vec) for doc, vec in rows}


def hybrid_search_with_score(settings: Settings, slug: str, query: str, k: int
                             ) -> list[tuple[Document, float]]:
    """k_fetch candidates per source → RRF fuse → MMR → top-k. Distances keep
    the dense cosine value where available; lexical-only hits are pre-relevant
    (the graph's dense guard already refused off-topic queries before this)."""
    from app.rag.vectorstore import get_store

    fetch = max(k * 2, 12)
    store = get_store(settings, slug)
    dense = store.similarity_search_with_score(query, k=fetch)
    lex = lexical_search(settings, slug, query, fetch)

    dist_by_key = {_doc_key(d): dist for d, dist in dense}
    fused = rrf_fuse([
        [(_doc_key(d), d) for d, _ in dense],
        [(_doc_key(d), d) for d in lex],
    ][:fetch])

    candidates = [(key, doc, dist_by_key.get(key, 0.0)) for key, doc in fused]
    try:  # MMR needs vectors; on any failure the fused order is already good
        from app.rag.embeddings import get_embeddings

        qvec = get_embeddings(settings).embed_query(query)
        vectors = _fetch_embeddings(settings, slug, [key for key, _, _ in candidates])
        with_vecs = [(key, doc, vectors.get(key, [])) for key, doc, _ in candidates]
        with_vecs = [c for c in with_vecs if c[2]]
        if len(with_vecs) >= 2:
            dist_of = {key: dist for key, _, dist in candidates}
            candidates = [(key, doc, dist_of[key])
                          for key, doc, _ in mmr_select(qvec, with_vecs, k)]
    except Exception:  # noqa: BLE001 — diversification is best-effort
        pass
    return [(doc, dist) for _, doc, dist in candidates[:k]]
