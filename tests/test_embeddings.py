import httpx

from app.rag.embeddings import OpenAICompatEmbeddings


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload, self.status_code, self.text = payload, status_code, ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=None)


def _payload(texts):
    return {"data": [{"index": i, "embedding": [float(i)] * 4} for i in range(len(texts))]}


def test_embed_documents_batches(monkeypatch):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json["input"])
        return FakeResponse(_payload(json["input"]))

    monkeypatch.setattr(httpx, "post", fake_post)
    emb = OpenAICompatEmbeddings("https://x.example/v1", "key", "m", batch_size=2)
    out = emb.embed_documents(["a", "b", "c"])
    assert len(calls) == 2 and len(calls[0]) == 2 and len(calls[1]) == 1
    # each fake batch returns index-based vectors: [0,1] then [0]
    assert out == [[0.0] * 4, [1.0] * 4, [0.0] * 4]


def test_embed_query_single(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None):
        return FakeResponse(_payload(json["input"]))

    monkeypatch.setattr(httpx, "post", fake_post)
    emb = OpenAICompatEmbeddings("https://x.example/v1", "key", "m")
    assert emb.embed_query("hi") == [0.0] * 4


def test_retries_on_upstream_error(monkeypatch):
    monkeypatch.setattr("app.rag.embeddings._RETRY_DELAYS", (0, 0, 0))
    state = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None):
        state["n"] += 1
        if state["n"] < 3:
            return FakeResponse({}, status_code=503)
        return FakeResponse(_payload(json["input"]))

    monkeypatch.setattr(httpx, "post", fake_post)
    emb = OpenAICompatEmbeddings("https://x.example/v1", "key", "m")
    assert emb.embed_query("hi") == [0.0] * 4
    assert state["n"] == 3
