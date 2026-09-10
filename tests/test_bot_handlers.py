import pytest
from types import SimpleNamespace

from app.models import User, Workspace
from app.services.chat import start_conversation


class Recorder:
    def __init__(self):
        self.texts = []
        self.markups = []

    async def __call__(self, text, parse_mode=None, reply_markup=None):
        self.texts.append(text)
        self.markups.append(reply_markup)


def fake_update(*, text=None, document=None, callback_data=None, tg_id=42,
                username="kartik", first="Kartik"):
    msg = SimpleNamespace(text=text, reply_text=Recorder(),
                          document=SimpleNamespace(
                              file_name=document["file_name"], file_size=document.get("file_size", 10))
                          if document else None,
                          chat=SimpleNamespace(id=tg_id))
    effective_message = msg
    cq = SimpleNamespace(data=callback_data, answer=lambda: None,
                         message=SimpleNamespace(reply_text=Recorder())) if callback_data else None
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=tg_id, username=username, first_name=first),
        effective_message=effective_message, effective_chat=SimpleNamespace(id=tg_id),
        callback_query=cq, message=msg)


class FakeGraph:
    def invoke(self, state, config=None):
        return {"answer": "Answer [1]", "sources": [{"file": "a.md", "section": None}],
                "refused": None, "history": []}


class _CtxSession:
    def __init__(self, session):
        self._s = session

    def __enter__(self):
        return self._s

    def __exit__(self, *a):
        self._s.rollback()  # don't leak test writes between handlers
        return False


@pytest.fixture
def env(session, test_settings):
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    owner = ensure_system_user(session)
    ws = create_workspace(session, owner, "Bot WS", is_private=False)
    session.commit()
    container = SimpleNamespace(settings=test_settings, session_factory=None, graph=FakeGraph())
    ctx = SimpleNamespace(bot_data={"container": container}, user_data={},
                          args=[], application=SimpleNamespace(bot_data={"container": container}))
    # handlers open sessions via context; monkeypatch factory to return the test session
    container.session_factory = lambda: _CtxSession(session)
    return session, ws, ctx


@pytest.mark.asyncio
async def test_start_registers_user_and_lists(env):
    from app.bot.handlers import cmd_start

    session, ws, ctx = env
    await cmd_start(fake_update(text="/start"), ctx)
    user = session.query(User).filter_by(telegram_id=42).one()
    assert user.display_name == "Kartik"


@pytest.mark.asyncio
async def test_newworkspace_command(env):
    from app.bot.handlers import cmd_newworkspace

    session, ws, ctx = env
    ctx.args = ["My", "Team", "Space", "private"]
    upd = fake_update(text="/newworkspace My Team Space private")
    await cmd_newworkspace(upd, ctx)
    created = session.query(Workspace).filter_by(slug="my-team-space").one()
    assert created.is_private is True
    replies = " ".join(upd.effective_message.reply_text.texts)
    assert "created" in replies.lower() and "My Team Space" in replies


@pytest.mark.asyncio
async def test_text_question_answered_with_sources(env):
    from app.bot.handlers import on_text
    from app.services.users import get_or_create_user

    session, ws, ctx = env
    user = get_or_create_user(session, ctx.bot_data["container"].settings, telegram_id=42)
    start_conversation(session, user, ws)
    session.commit()
    upd = fake_update(text="What is the policy?")
    await on_text(upd, ctx)
    assert "Answer [1]" in upd.effective_message.reply_text.texts[0]
    assert "Sources:" in upd.effective_message.reply_text.texts[0]


@pytest.mark.asyncio
async def test_private_workspace_hides_text(env):
    from app.bot.handlers import on_text
    from app.services.users import ensure_system_user, get_or_create_user
    from app.services.workspaces import create_workspace

    session, ws, ctx = env
    owner = ensure_system_user(session)
    priv = create_workspace(session, owner, "Secret", is_private=True)
    user = get_or_create_user(session, ctx.bot_data["container"].settings, telegram_id=42)
    start_conversation(session, user, priv)
    session.commit()
    upd = fake_update(text="hello")
    await on_text(upd, ctx)
    assert "private" in upd.effective_message.reply_text.texts[0].lower()


@pytest.mark.asyncio
async def test_demo_switches_workspace(env):
    from app.bot import handlers

    session, ws, ctx = env
    # create the demo workspace the command looks for
    from app.models import Workspace as W
    from app.services.users import ensure_system_user
    from app.services.workspaces import create_workspace

    demo = create_workspace(session, ensure_system_user(session), "Demo Delivery Policy")
    demo.slug = handlers.DEMO_SLUG
    session.commit()
    upd = fake_update(text="/demo")
    await handlers.cmd_demo(upd, ctx)
    assert upd.effective_message.reply_text.markups[0] is not None  # demo question buttons
