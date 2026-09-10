from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Conversation, User, Workspace


@dataclass
class ChatResult:
    answer: str
    sources: list[dict]
    conversation_id: str
    refused: bool


def start_conversation(session: Session, user: User, workspace: Workspace,
                      title: str = "New chat") -> Conversation:
    conv = Conversation(user_id=user.id, workspace_id=workspace.id, title=title)
    session.add(conv)
    session.flush()
    user.current_conversation_id = conv.id
    user.current_workspace_id = workspace.id
    user.pending_action = None
    session.flush()
    return conv


def list_conversations(session: Session, user: User, limit: int = 8) -> list[Conversation]:
    return list(session.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id, Conversation.is_active.is_(True))
        .order_by(Conversation.last_message_at.desc().nullslast(), Conversation.created_at.desc())
        .limit(limit)
    ))


def resume_conversation(session: Session, user: User, conversation_id: str) -> Conversation | None:
    conv = session.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        return None
    user.current_conversation_id = conv.id
    user.current_workspace_id = conv.workspace_id
    user.pending_action = None
    session.flush()
    return conv


def _current_conversation(session: Session, user: User, workspace: Workspace) -> Conversation:
    if user.current_conversation_id:
        conv = session.get(Conversation, user.current_conversation_id)
        if conv is not None and conv.workspace_id == workspace.id and conv.is_active:
            return conv
    return start_conversation(session, user, workspace)


def run_chat(session: Session, user: User, workspace: Workspace, message: str,
             graph, settings: Settings | None = None) -> ChatResult:
    from app.rag.graph import thread_id_for

    conv = _current_conversation(session, user, workspace)
    state = graph.invoke(
        {"question": message, "rewritten": message, "attempt": 0,
         "workspace_slug": workspace.slug, "history": []},
        config={"configurable": {"thread_id": thread_id_for(conv.id)}},
    )
    refused = bool(state.get("refused"))
    answer = state["refused"] or state["answer"]
    if conv.title == "New chat":
        conv.title = message.strip()[:64]
    conv.last_message_at = datetime.now(timezone.utc)
    session.commit()
    return ChatResult(answer=answer, sources=state.get("sources", []),
                      conversation_id=conv.id, refused=refused)
