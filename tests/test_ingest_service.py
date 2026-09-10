import pytest

from app.services.ingest import (
    SUPPORTED_EXTS, create_document_row, ingest_document_sync,
    ingest_text_sync, parse_file,
)

MD_DOC = b"# Title\n\nBody text that is long enough to pass the chunk minimum length filter.\n"


class FakeStore:
    def __init__(self):
        self.added = []

    def add_documents(self, docs):
        self.added.extend(docs)


class FailingStore(FakeStore):
    def __init__(self, fail_times=1):
        super().__init__()
        self.fail_times = fail_times

    def add_documents(self, docs):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("db down")
        self.added.extend(docs)


@pytest.fixture
def fast_retry(monkeypatch):
    monkeypatch.setattr("app.services.ingest._RETRY_DELAYS", [0, 0, 0])


def test_supported_exts():
    assert SUPPORTED_EXTS == frozenset({"pdf", "docx", "pptx", "xlsx", "html", "md"})


def test_parse_markdown_file(fast_retry):
    chunks = parse_file(MD_DOC, "md")
    assert len(chunks) == 1 and "Title" in chunks[0].page_content


def test_parse_unsupported_ext():
    with pytest.raises(ValueError):
        parse_file(b"x", "exe")


def test_create_document_row(session, test_settings):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Ingest WS")
    doc = create_document_row(session, w, "policy.PDF", source="telegram", uploaded_by=owner.id, byte_size=10)
    session.flush()
    assert doc.file_ext == "pdf" and doc.status == "processing" and doc.id


def test_ingest_document_success(fast_retry, test_settings, session):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Ingest OK")
    doc = create_document_row(session, w, "a.md", source="seed")
    session.commit()
    store = FakeStore()
    n = ingest_document_sync(None, doc.id, MD_DOC, "md", store, settings=test_settings, _session=session)
    assert n == 1
    session.refresh(doc)
    assert doc.status == "ready" and doc.chunk_count == 1
    assert len(store.added) == 1
    assert store.added[0].metadata["document_id"] == doc.id


def test_ingest_document_retries_then_succeeds(fast_retry, test_settings, session):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Ingest Retry")
    doc = create_document_row(session, w, "a.md", source="seed")
    session.commit()
    store = FailingStore(fail_times=2)
    n = ingest_document_sync(None, doc.id, MD_DOC, "md", store, settings=test_settings, _session=session)
    assert n == 1 and len(store.added) == 1


def test_ingest_document_marks_failed(fast_retry, test_settings, session):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Ingest Fail")
    doc = create_document_row(session, w, "a.md", source="seed")
    session.commit()
    with pytest.raises(RuntimeError):
        ingest_document_sync(None, doc.id, MD_DOC, "md", FailingStore(fail_times=99),
                             settings=test_settings, _session=session)
    session.refresh(doc)
    assert doc.status == "failed" and "db down" in doc.error


def test_ingest_text_sync(fast_retry, test_settings, session):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Paste WS")
    store = FakeStore()
    doc = ingest_text_sync(None, w, "Meeting Notes", "point one " * 100, store,
                           uploaded_by=owner.id, _session=session)
    assert doc.status == "ready" and doc.source == "paste" and doc.filename == "Meeting Notes"
    assert len(store.added) >= 1


def test_find_inflight_duplicate(fast_retry, test_settings, session):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Dupe WS")
    processing = create_document_row(session, w, "report.pdf", source="telegram", byte_size=500)
    ready = create_document_row(session, w, "report.pdf", source="telegram", byte_size=500)
    ready.status = "ready"
    done = create_document_row(session, w, "report.pdf", source="telegram", byte_size=999)
    done.status = "ready"
    session.flush()

    from app.services.ingest import find_inflight_duplicate

    hit = find_inflight_duplicate(session, w.id, "report.pdf", 500)
    assert hit is not None and hit.id == processing.id
    assert find_inflight_duplicate(session, w.id, "report.pdf", 999) is None
    assert find_inflight_duplicate(session, w.id, "other.pdf", 500) is None
