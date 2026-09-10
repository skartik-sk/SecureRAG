"""Hybrid retrieval unit tests (pure functions, no DB)."""
from langchain_core.documents import Document

from app.rag.hybrid import mmr_select, rrf_fuse


def _doc(text):
    return Document(page_content=text, metadata={})


def test_rrf_fuse_merges_and_ranks():
    a = [("sla", _doc("sla")), ("refund", _doc("refund")), ("zones", _doc("zones"))]
    b = [("refund", _doc("refund")), ("sla", _doc("sla"))]
    fused = rrf_fuse([a, b])
    keys = [k for k, _ in fused]
    # appearing high in both lists beats appearing in only one
    assert keys[0] in ("sla", "refund")
    assert set(keys) == {"sla", "refund", "zones"}
    assert len(fused) == 3  # duplicates merged


def test_rrf_fuse_single_list_preserves_order():
    a = [("x", _doc("x")), ("y", _doc("y"))]
    assert [k for k, _ in rrf_fuse([a])] == ["x", "y"]


def test_mmr_select_balances_relevance_and_diversity():
    q = [1.0, 0.0]
    candidates = [
        ("near-a", _doc("a"), [0.9, 0.1]),   # relevant
        ("near-a-dup", _doc("a dup"), [0.89, 0.1]),  # near-identical to first
        ("orthogonal", _doc("o"), [0.0, 1.0]),  # different direction
    ]
    # low lambda (diversity-heavy): the redundant doc loses to the diverse one
    picked = mmr_select(q, candidates, k=2, lambda_mult=0.3)
    keys = [k for k, _, _ in picked]
    assert keys[0] == "near-a"
    assert "orthogonal" in keys and "near-a-dup" not in keys
    # default lambda (relevance-heavy): both top-relevance docs win
    picked = mmr_select(q, candidates, k=2)
    assert [k for k, _, _ in picked][:2] == ["near-a", "near-a-dup"]


def test_mmr_select_k_exceeds_candidates():
    candidates = [("x", _doc("x"), [1.0]), ("y", _doc("y"), [0.5])]
    assert len(mmr_select([1.0], candidates, k=5)) == 2


def test_hybrid_search_full_path(test_settings, db, monkeypatch):
    """Integration: real pgvector tables + FTS, fake embeddings."""
    from sqlalchemy import text as sql

    from app.db import engine

    slug = "hybridws"
    eng = engine(test_settings)
    with eng.begin() as c:
        c.execute(sql("DELETE FROM langchain_pg_embedding WHERE collection_id IN "
                      "(SELECT uuid FROM langchain_pg_collection WHERE name = :n)"),
                  {"n": f"ws_{slug}"})
        c.execute(sql("DELETE FROM langchain_pg_collection WHERE name = :n"),
                  {"n": f"ws_{slug}"})
        c.execute(sql("INSERT INTO langchain_pg_collection (name, uuid, cmetadata) "
                      "VALUES (:n, gen_random_uuid(), CAST('{}' AS json))"), {"n": f"ws_{slug}"})
        cid = c.execute(sql("SELECT uuid FROM langchain_pg_collection WHERE name = :n"),
                        {"n": f"ws_{slug}"}).scalar()
        rows = [
            ("alpha zone delivery cut-off is 10 am", '[1.0, 0.1]'),
            ("refund window is thirty days", '[0.2, 0.9]'),
            ("alpha beta gamma launch code", '[0.95, 0.2]'),
        ]
        for i, (doc, vec) in enumerate(rows):
            c.execute(sql("INSERT INTO langchain_pg_embedding "
                          "(id, collection_id, document, cmetadata, embedding) "
                          "VALUES (:id, :cid, :doc, CAST(:meta AS jsonb), CAST(:vec AS vector))"),
                      {"id": f"hy{i}", "cid": str(cid), "doc": doc,
                       "meta": '{"section": "S"}', "vec": vec})

    class FakeEmb:
        def embed_query(self, q):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    import app.rag.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "get_embeddings", lambda s=None: FakeEmb())

    from app.rag.hybrid import hybrid_search_with_score, lexical_search

    lex = lexical_search(test_settings, slug, "alpha launch", k=5)
    assert len(lex) == 1  # websearch tsquery ANDs terms: only the doc with both
    assert "launch" in lex[0].page_content
    assert len(lexical_search(test_settings, slug, "alpha OR launch", k=5)) == 2

    results = hybrid_search_with_score(test_settings, slug, "alpha launch", k=2)
    assert len(results) == 2
    for doc, dist in results:
        assert isinstance(dist, float)
    assert "alpha" in results[0][0].page_content

    with eng.begin() as c:  # cleanup
        c.execute(sql("DELETE FROM langchain_pg_embedding WHERE collection_id = :cid"),
                  {"cid": str(cid)})
        c.execute(sql("DELETE FROM langchain_pg_collection WHERE uuid = :cid"),
                  {"cid": str(cid)})
