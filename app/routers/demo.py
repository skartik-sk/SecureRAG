"""Public demo chat — lets web visitors try the real pipeline without an API
key. Hard-limited to the seeded demo workspace: read-only (no ingest), fresh
conversation per request, best-effort per-IP rate limiting."""

import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/demo")

MAX_DEMO_CHARS = 500
_WINDOW_SECONDS = 300
_MAX_PER_WINDOW = 8
_hits: dict[str, list[float]] = {}


def _rate_ok(ip: str) -> bool:
    now = time.monotonic()
    hits = [t for t in _hits.get(ip, []) if now - t < _WINDOW_SECONDS]
    _hits[ip] = hits
    if len(hits) >= _MAX_PER_WINDOW:
        return False
    _hits[ip].append(now)
    return True


class DemoQuestion(BaseModel):
    message: str


@router.post("/chat")
async def demo_chat(request: Request, body: DemoQuestion):
    message = body.message.strip()
    if not message or len(message) > MAX_DEMO_CHARS:
        raise HTTPException(status_code=400,
                            detail=f"Message must be 1..{MAX_DEMO_CHARS} characters")
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_ok(client_ip):
        raise HTTPException(status_code=429,
                            detail="Too many demo questions — try again in a few minutes")

    graph = request.app.state.graph
    if graph is None:
        raise HTTPException(status_code=503, detail="Chat backend unavailable")

    from app.db import SessionLocal
    from app.services.chat import run_chat, start_conversation
    from app.services.users import ensure_system_user
    from app.services.workspaces import workspace_by_slug
    from seed import DEMO_SLUG

    settings = request.app.state.settings
    with SessionLocal(settings)() as session:
        ws = workspace_by_slug(session, DEMO_SLUG)
        if ws is None:
            raise HTTPException(status_code=404, detail="Demo workspace not seeded yet")
        user = ensure_system_user(session)
        # fresh thread per request; no intermediate commit so concurrent
        # visitors can't cross threads through the shared system user
        start_conversation(session, user, ws, title="Web demo")
        result = run_chat(session, user, ws, message, graph)
        session.commit()
    return {"answer": result.answer, "sources": [] if result.refused else result.sources,
            "refused": result.refused}
