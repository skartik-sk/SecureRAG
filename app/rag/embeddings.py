import time

import httpx
from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings

_instance = None

_EMBED_BATCH = 64
_RETRY_DELAYS = (1, 2, 4)


class OpenAICompatEmbeddings(Embeddings):
    """Embeddings via any OpenAI-compatible /v1/embeddings endpoint.

    Serverless targets (Vercel) can't ship torch/sentence-transformers, so
    embeddings are an HTTP call just like the LLM.
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 batch_size: int = _EMBED_BATCH, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout

    def _post(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last: Exception | None = None
        for attempt, delay in enumerate((*_RETRY_DELAYS, None)):
            try:
                r = httpx.post(f"{self.base_url}/embeddings", headers=headers,
                               json={"model": self.model, "input": texts},
                               timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"embeddings upstream {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in data]
            except Exception as e:  # noqa: BLE001 — retry transport/upstream errors
                last = e
                if delay is not None:
                    time.sleep(delay)
        raise last  # type: ignore[misc]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._post(texts[i:i + self.batch_size]))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._post([text])[0]


def get_embeddings(settings: Settings | None = None) -> OpenAICompatEmbeddings:
    global _instance
    if _instance is None:
        s = settings or get_settings()
        if not s.embeddings_api_key:
            raise RuntimeError(
                "EMBEDDINGS_API_KEY is not set — get a key from together.ai (or point "
                "EMBEDDINGS_BASE_URL at any OpenAI-compatible embeddings provider).")
        _instance = OpenAICompatEmbeddings(s.embeddings_base_url, s.embeddings_api_key,
                                           s.embedding_model)
    return _instance
