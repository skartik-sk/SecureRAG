import logging

from app.logging_setup import setup_logging


def test_log_level_env_sets_root_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    logging.getLogger().setLevel(logging.WARNING)
    setup_logging()
    assert logging.getLogger().isEnabledFor(logging.INFO)


def test_defaults_to_warning(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    logging.getLogger().setLevel(logging.WARNING)
    setup_logging()
    assert not logging.getLogger().isEnabledFor(logging.INFO)


def test_idempotent_no_duplicate_handlers():
    import logging as l

    l.getLogger().setLevel(l.WARNING)
    setup_logging()
    n = len(l.getLogger().handlers)
    setup_logging()
    assert len(l.getLogger().handlers) == n
