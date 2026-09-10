import pytest

from seed import SEED_WORKSPACES, seed_all

MD = b"# H1\n\n" + b"Some chunk body content long enough. " * 40


@pytest.fixture
def corpus(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "delivery_policy.md").write_bytes(MD)
    return str(d)


@pytest.fixture
def fake_store(monkeypatch):
    added = []

    class S:
        def add_documents(self, docs):
            added.extend(docs)

    monkeypatch.setattr("seed._store_for", lambda settings, slug: S())
    return added


class _Fixed:
    def __init__(self, s):
        self._s = s

    def __enter__(self):
        return self._s

    def __exit__(self, *a):
        return False


@pytest.mark.integration
def test_seed_all_creates_workspaces_and_is_idempotent(
        session, test_settings, corpus, fake_store, monkeypatch):
    # use the live test session factory so rows are visible
    monkeypatch.setattr("seed._session_factory", lambda settings: (lambda: _Fixed(session)))
    counts = seed_all(corpus, test_settings)
    assert set(counts) == {slug for slug, _, _ in SEED_WORKSPACES}
    from app.services.workspaces import workspace_by_slug

    ws = workspace_by_slug(session, "delivery-policy")
    assert ws is not None
    again = seed_all(corpus, test_settings)
    assert all(v == 0 for v in again.values())  # idempotent: nothing re-ingested
