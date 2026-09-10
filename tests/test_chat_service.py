from app.models import Conversation, User, Workspace
from app.services.chat import (
    ChatResult, list_conversations, resume_conversation, run_chat, start_conversation,
)
from app.services.users import ensure_system_user


class FakeGraph:
    def __init__(self, answer="A [1]", sources=None, refused=None):
        self.answer, self.sources, self.refused = answer, sources or [], refused
        self.invocations = []

    def invoke(self, state, config=None):
        self.invocations.append((state, config))
        return {"answer": self.answer, "sources": self.sources,
                "refused": self.refused, "history": []}


def _user_and_ws(session):
    user = User(telegram_id=5, display_name="T")
    session.add(user)
    session.flush()
    ws = Workspace(slug="cws", name="C", owner_id=user.id)
    session.add(ws)
    session.flush()
    return user, ws


def test_start_conversation_sets_current(session):
    user, ws = _user_and_ws(session)
    conv = start_conversation(session, user, ws, title="Hello policy")
    assert user.current_conversation_id == conv.id
    assert user.current_workspace_id == ws.id
    assert user.pending_action is None
    assert conv.title == "Hello policy" and conv.is_active


def test_run_chat_creates_and_titles_conversation(session):
    user, ws = _user_and_ws(session)
    graph = FakeGraph()
    result = run_chat(session, user, ws, "What is the refund window?", graph)
    assert isinstance(result, ChatResult)
    conv_id = result.conversation_id
    conv = session.get(Conversation, conv_id)
    assert conv.title == "What is the refund window?"[:64]
    assert conv.last_message_at is not None
    assert result.answer == "A [1]"
    state, config = graph.invocations[0]
    assert config["configurable"]["thread_id"] == f"conv-{conv_id}"
    assert state["question"] == "What is the refund window?"


def test_run_chat_reuses_current_conversation(session):
    user, ws = _user_and_ws(session)
    graph = FakeGraph()
    r1 = run_chat(session, user, ws, "first question", graph)
    r2 = run_chat(session, user, ws, "second question", graph)
    assert r1.conversation_id == r2.conversation_id
    assert len(graph.invocations) == 2
    # different workspace → new conversation
    ws2 = Workspace(slug="cws2", name="C2", owner_id=user.id)
    session.add(ws2)
    session.flush()
    r3 = run_chat(session, user, ws2, "other workspace now", graph)
    assert r3.conversation_id != r1.conversation_id


def test_run_chat_refused_answer(session):
    user, ws = _user_and_ws(session)
    result = run_chat(session, user, ws, "gibberish", FakeGraph(refused="Nope."))
    assert result.answer == "Nope." and result.refused is True


def test_list_and_resume_conversations(session):
    user, ws = _user_and_ws(session)
    c1 = start_conversation(session, user, ws, "one")
    c2 = start_conversation(session, user, ws, "two")
    assert {c.id for c in list_conversations(session, user)} == {c1.id, c2.id}
    assert resume_conversation(session, user, c1.id).id == c1.id
    assert user.current_conversation_id == c1.id
    other = User(telegram_id=6, display_name="X")
    session.add(other)
    session.flush()
    assert resume_conversation(session, other, c1.id) is None  # not theirs
    assert resume_conversation(session, user, "no-such-id") is None
