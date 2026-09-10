import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    # Embeddings come from any OpenAI-compatible /v1/embeddings endpoint
    # (Together, Mistral, OpenAI, ...) — serverless can't run local models.
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embeddings_base_url: str = "https://api.together.xyz/v1"
    embeddings_api_key: str = ""

    telegram_bot_token: str = ""
    telegram_admin_id: int = 0
    telegram_webhook_secret: str = "dev-secret-change-me"
    bot_mode: str = "disabled"  # disabled | polling | webhook
    webhook_url: str = ""

    api_keys: str = ""
    off_topic_distance: float = 0.95
    history_window: int = 8
    # On Vercel the filesystem is read-only except /tmp, and fire-and-forget
    # background tasks are frozen after the response — so default accordingly.
    upload_dir: str = Field(default_factory=lambda: "/tmp/uploads" if os.getenv("VERCEL") else "data/uploads")
    inline_ingest: bool = Field(default_factory=lambda: os.getenv("VERCEL") == "1")
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
