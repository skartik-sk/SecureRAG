import io

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


class FakeGraph:
    def invoke(self, state, config=None):
        return {"answer": "Answer [1]", "sources": [{"file": "a.md", "section": None}],
                "refused": None, "history": []}


@pytest.fixture
def client(test_settings, session):
    from app.services.users import ensure_system_user

    ensure_system_user(session)
    session.commit()
    app = create_app(test_settings)
    app.state.graph = FakeGraph()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, test_settings


HEADERS = {"X-API-Key": "dev-api-key-1"}


def test_health_no_auth(client):
    c, s = client
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_api_requires_key(client):
    c, s = client
    assert c.get("/api/v1/workspaces").status_code == 401
    assert c.get("/api/v1/workspaces", headers={"X-API-Key": "wrong"}).status_code == 401


def test_workspace_crud(client):
    c, s = client
    r = c.post("/api/v1/workspaces", headers=HEADERS, json={"name": "API Made", "is_private": True})
    assert r.status_code == 201
    body = r.json()
    assert body["slug"] == "api-made" and body["is_private"] is True
    listed = c.get("/api/v1/workspaces", headers=HEADERS).json()
    assert any(w["slug"] == "api-made" for w in listed)


def test_documents_404_and_upload(client, session):
    c, s = client
    assert c.get("/api/v1/workspaces/missing/documents", headers=HEADERS).status_code == 404
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    ws = create_workspace(session, owner, "Upload Target")
    session.commit()
    r = c.post(f"/api/v1/workspaces/upload-target/documents", headers=HEADERS,
               files={"file": ("notes.md", io.BytesIO(b"# H\n\nSome long enough content here.\n"), "text/markdown")})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "processing" and body["document_id"]
    listed = c.get("/api/v1/workspaces/upload-target/documents", headers=HEADERS).json()
    assert listed[0]["filename"] == "notes.md"


def test_upload_rejects_bad_type_and_oversize(client, session):
    c, s = client
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    create_workspace(session, owner, "Bad Uploads")
    session.commit()
    r = c.post("/api/v1/workspaces/bad-uploads/documents", headers=HEADERS,
               files={"file": ("virus.exe", io.BytesIO(b"x"), "application/x-msdownload")})
    assert r.status_code == 400


def test_chat_endpoint(client, session):
    c, s = client
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    create_workspace(session, owner, "Chat WS")
    session.commit()
    r = c.post("/api/v1/workspaces/chat-ws/chat", headers=HEADERS, json={"message": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Answer [1]"
    assert body["sources"] == [{"file": "a.md", "section": None}]
    assert body["conversation_id"] and body["refused"] is False


def test_webhook_rejects_bad_secret(client):
    c, s = client
    r = c.post("/telegram/webhook", json={}, headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_webhook_passes_update_to_ptb(client, test_settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(test_settings)

    class FakePTB:
        def __init__(self):
            self.updates = []
            self.bot = None

        async def process_update(self, update):
            self.updates.append(update)

    app.state.ptb_app = FakePTB()
    with TestClient(app, raise_server_exceptions=False) as c2:
        r = c2.post("/telegram/webhook", json={"update_id": 1},
                    headers={"X-Telegram-Bot-Api-Secret-Token": test_settings.telegram_webhook_secret})
    assert r.status_code == 200
    assert app.state.ptb_app.updates[0].update_id == 1


def test_landing_page(client):
    c, s = client
    r = c.get("/")
    assert r.status_code == 200
    body = r.text
    assert "Secure" in body and "RAG" in body
    assert "t.me/SecureRAG_bot" in body and "/docs" in body


def test_demo_chat_answers(client, session):
    from app.routers.demo import _hits
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace
    from seed import DEMO_SLUG

    _hits.clear()
    ensure_system_user(session)
    create_workspace(session, ensure_system_user(session), "Demo Delivery Policy")
    ws = session.query(type(create_workspace(session, ensure_system_user(session), "x"))).first()
    session.query(type(ws)).filter_by(slug="demo-delivery-policy").one().slug = DEMO_SLUG
    session.commit()
    c, s = client
    r = c.post("/api/v1/demo/chat", json={"message": "What is the delivery SLA?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Answer [1]" and body["refused"] is False


def test_demo_chat_validates_and_404s(client):
    from app.routers.demo import _hits
    _hits.clear()
    c, s = client
    assert c.post("/api/v1/demo/chat", json={"message": ""}).status_code == 400
    assert c.post("/api/v1/demo/chat", json={"message": "x" * 501}).status_code == 400
    r = c.post("/api/v1/demo/chat", json={"message": "hello"})
    assert r.status_code == 404 and "not seeded" in r.json()["detail"]


def test_demo_chat_rate_limited(client, session, monkeypatch):
    import app.routers.demo as demo_mod
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace
    from seed import DEMO_SLUG

    _hits = demo_mod._hits
    _hits.clear()
    monkeypatch.setattr(demo_mod, "_MAX_PER_WINDOW", 2)
    ensure_system_user(session)
    ws = create_workspace(session, ensure_system_user(session), "Demo Delivery Policy")
    ws.slug = DEMO_SLUG
    session.commit()
    c, s = client
    assert c.post("/api/v1/demo/chat", json={"message": "q1"}).status_code == 200
    assert c.post("/api/v1/demo/chat", json={"message": "q2"}).status_code == 200
    r = c.post("/api/v1/demo/chat", json={"message": "q3"})
    assert r.status_code == 429 and "Too many" in r.json()["detail"]
    _hits.clear()
