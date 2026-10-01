import pytest

from app.models import AnswerFeedback, Conversation, User, Workspace
from app.services.feedback import rate_feedback, record_answer


@pytest.fixture
def user_and_ws(session):
    user = User(telegram_id=7, display_name="F")
    session.add(user)
    session.flush()
    ws = Workspace(slug="fbws", name="FB", owner_id=user.id)
    session.add(ws)
    session.flush()
    conv = Conversation(user_id=user.id, workspace_id=ws.id)
    session.add(conv)
    session.flush()
    return user, ws, conv


def test_record_answer_stores_unrated(session, user_and_ws):
    user, ws, conv = user_and_ws
    fb = record_answer(session, user, ws, conv.id, "What is the SLA?", "48 hours [1]")
    assert fb.id
    assert fb.rating is None
    assert fb.question == "What is the SLA?"
    assert fb.answer == "48 hours [1]"
    assert fb.user_id == user.id and fb.workspace_id == ws.id
    assert session.get(AnswerFeedback, fb.id) is fb


def test_record_answer_allows_null_conversation(session, user_and_ws):
    user, ws, _conv = user_and_ws
    fb = record_answer(session, user, ws, None, "q", "a")
    assert fb.conversation_id is None


def test_rate_feedback_sets_rating(session, user_and_ws):
    user, ws, _conv = user_and_ws
    fb = record_answer(session, user, ws, None, "q", "a")
    out = rate_feedback(session, user, fb.id, 1)
    assert out.rating == 1


def test_rate_feedback_dislike_is_minus_one(session, user_and_ws):
    user, ws, _conv = user_and_ws
    fb = record_answer(session, user, ws, None, "q", "a")
    assert rate_feedback(session, user, fb.id, 0).rating == -1


def test_cannot_rate_someone_elses_feedback(session, user_and_ws):
    user, ws, _conv = user_and_ws
    other = User(telegram_id=8, display_name="O")
    session.add(other)
    session.flush()
    fb = record_answer(session, user, ws, None, "q", "a")
    assert rate_feedback(session, other, fb.id, 1) is None
    assert fb.rating is None


def test_unknown_feedback_id_returns_none(session, user_and_ws):
    user, _ws, _conv = user_and_ws
    assert rate_feedback(session, user, "no-such-id", 1) is None


def test_invalid_rating_raises(session, user_and_ws):
    user, ws, _conv = user_and_ws
    fb = record_answer(session, user, ws, None, "q", "a")
    with pytest.raises(ValueError):
        rate_feedback(session, user, fb.id, 5)
