from app.bot.keyboards import (
    DEMO_QUESTIONS, conversations_keyboard, demo_keyboard, new_chat_keyboard,
    workspace_keyboard,
)


class FakeWS:
    def __init__(self, wid, slug):
        self.id, self.slug, self.name = wid, slug, slug


def test_demo_questions_exist():
    assert len(DEMO_QUESTIONS) == 3 and all(len(q) > 10 for q in DEMO_QUESTIONS)


def test_workspace_keyboard_marks_current():
    kb = workspace_keyboard([FakeWS("1", "alpha"), FakeWS("2", "beta")], current_id="2")
    labels = [b.text for row in kb.inline_keyboard for b in row]
    cbs = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert labels == ["alpha", "✓ beta"]
    assert cbs == ["ws:1", "ws:2"]


def test_conversations_keyboard():
    class C:
        id, title = "c1", "x" * 50

    kb = conversations_keyboard([C()])
    btn = kb.inline_keyboard[0][0]
    assert btn.callback_data == "cv:c1" and len(btn.text) <= 34


def test_demo_and_new_keyboards():
    assert demo_keyboard().inline_keyboard[0][0].callback_data == "dq:0"
    labels = [b.callback_data for b in new_chat_keyboard().inline_keyboard[0]]
    assert labels == ["nc:content", "nc:skip"]
