"""Public demo chat — lets web visitors try the real pipeline without an API
key or login. Fully ephemeral: no database rows are ever created (no users,
no conversations, no LangGraph checkpoints). Conversation memory lives in an
in-process dict keyed by a random session id, with a TTL and hard caps.

Hard-limited to the seeded demo workspace: read-only vector search, no ingest.
"""

import secrets
import time
from dataclasses import dataclass, field

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/demo")

MAX_DEMO_CHARS = 500
_WINDOW_SECONDS = 300
_MAX_PER_WINDOW = 8
_SESSION_TTL_SECONDS = 30 * 60
_MAX_TURNS_PER_SESSION = 10
_MAX_SESSIONS = 1000
_HISTORY_TURNS = 6  # turns passed to the graph for context

_hits: dict[str, list[float]] = {}
_graph = None  # checkpointer-less graph; production graph writes checkpoints


def _rate_ok(ip: str) -> bool:
    now = time.monotonic()
    hits = [t for t in _hits.get(ip, []) if now - t < _WINDOW_SECONDS]
    _hits[ip] = hits
    if len(hits) >= _MAX_PER_WINDOW:
        return False
    _hits[ip].append(now)
    return True


@dataclass
class DemoSession:
    turns: list = field(default_factory=list)  # [(question, answer)]
    expires: float = 0.0


_sessions: dict[str, DemoSession] = {}


def _prune_sessions(now: float) -> None:
    expired = [sid for sid, s in _sessions.items() if s.expires <= now]
    for sid in expired:
        del _sessions[sid]
    while len(_sessions) > _MAX_SESSIONS:  # hard cap: drop soonest-expiring
        oldest = min(_sessions, key=lambda sid: _sessions[sid].expires)
        del _sessions[oldest]


def _get_graph(settings):
    """Demo-only graph without a checkpointer: history is supplied per call,
    so nothing is persisted anywhere."""
    global _graph
    if _graph is None:
        from app.rag.graph import build_graph
        from app.rag.hybrid import hybrid_search_with_score
        from app.rag.embeddings import get_embeddings
        from app.rag.llm import get_llm
        from app.rag.vectorstore import get_store

        _graph = build_graph(
            get_llm(settings),
            lambda slug: get_store(settings, slug, get_embeddings(settings)),
            settings=settings,
            hybrid_for_slug=lambda slug, q, k: hybrid_search_with_score(settings, slug, q, k),
        )
    return _graph


class DemoQuestion(BaseModel):
    message: str
    session_id: str | None = None


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

    graph = request.app.state.graph  # presence signal: chat backend configured
    if graph is None:
        raise HTTPException(status_code=503, detail="Chat backend unavailable")
    graph = _get_graph(request.app.state.settings)

    from seed import DEMO_SLUG

    now = time.monotonic()
    _prune_sessions(now)
    session = _sessions.get(body.session_id or "")
    if session is None:
        session = DemoSession(expires=now + _SESSION_TTL_SECONDS)
        session_id = secrets.token_urlsafe(16)
        _sessions[session_id] = session
    else:
        session_id = body.session_id

    history = []
    for prev_q, prev_a in session.turns[-_HISTORY_TURNS:]:
        history += [HumanMessage(content=prev_q), AIMessage(content=prev_a)]

    state = graph.invoke(
        {"question": message, "rewritten": message, "attempt": 0,
         "workspace_slug": DEMO_SLUG, "history": history})
    refused = bool(state.get("refused"))
    answer = state.get("refused") or state.get("answer", "")

    session.turns.append((message, answer))
    session.turns[:] = session.turns[-_MAX_TURNS_PER_SESSION:]
    session.expires = now + _SESSION_TTL_SECONDS

    return {"answer": answer, "sources": [] if refused else state.get("sources", []),
            "refused": refused, "session_id": session_id}
