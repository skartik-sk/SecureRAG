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
