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
