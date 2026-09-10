import pytest
from types import SimpleNamespace

from app.models import User, WorkspaceMember
from app.services.users import get_or_create_user
from app.services.workspaces import create_workspace
from tests.test_bot_handlers import FakeGraph, _CtxSession, fake_update


@pytest.fixture
def env(session, test_settings):
    """Same harness as test_bot_handlers.env — session-backed container."""
    container = SimpleNamespace(settings=test_settings, session_factory=None, graph=FakeGraph())
    ctx = SimpleNamespace(bot_data={"container": container}, user_data={}, args=[],
                          application=SimpleNamespace(bot_data={"container": container}))
    container.session_factory = lambda: _CtxSession(session)
    return session, ctx


@pytest.fixture
def owner_env(env):
    """Telegram user 42 owns a private workspace and has it selected."""
    from app.services.chat import start_conversation

    session, ctx = env
    owner = get_or_create_user(session, ctx.bot_data["container"].settings,
                               telegram_id=42, username="kartik", display_name="Kartik")
    priv = create_workspace(session, owner, "Members Only", is_private=True)
    start_conversation(session, owner, priv)
    session.commit()
    return session, priv, ctx, owner


def _member(session, ws, username):
    user = session.query(User).filter_by(telegram_username=username).one()
    member = session.get(WorkspaceMember, (ws.id, user.id))
    assert member is not None
    return member


@pytest.mark.asyncio
async def test_invite_with_editor_role(owner_env):
    from app.bot.handlers import cmd_invite

    session, priv, ctx, owner = owner_env
    session.add(User(telegram_id=99, telegram_username="alice", display_name="Alice"))
    session.commit()
    ctx.args = ["@alice", "editor"]
    upd = fake_update(text="/invite @alice editor")
    await cmd_invite(upd, ctx)
    assert _member(session, priv, "alice").role == "editor"
    assert "as editor" in " ".join(upd.effective_message.reply_text.texts)


@pytest.mark.asyncio
async def test_invite_defaults_to_viewer(owner_env):
    from app.bot.handlers import cmd_invite

    session, priv, ctx, owner = owner_env
    session.add(User(telegram_id=99, telegram_username="bob", display_name="Bob"))
    session.commit()
    ctx.args = ["@bob"]
    upd = fake_update(text="/invite @bob")
    await cmd_invite(upd, ctx)
    assert _member(session, priv, "bob").role == "viewer"


@pytest.mark.asyncio
async def test_promote_changes_role(owner_env):
    from app.bot.handlers import cmd_invite, cmd_promote

    session, priv, ctx, owner = owner_env
    session.add(User(telegram_id=99, telegram_username="carol", display_name="Carol"))
    session.commit()
    ctx.args = ["@carol"]
    await cmd_invite(fake_update(text="/invite @carol"), ctx)
    assert _member(session, priv, "carol").role == "viewer"
    ctx.args = ["@carol", "editor"]
    upd = fake_update(text="/promote @carol editor")
    await cmd_promote(upd, ctx)
    assert _member(session, priv, "carol").role == "editor"


@pytest.mark.asyncio
async def test_promote_requires_owner(env):
    from app.bot.handlers import cmd_promote
    from app.models import Workspace
    from app.services.chat import start_conversation

    session, ctx = env  # no workspace owned by tg 42 here
    outsider = get_or_create_user(session, ctx.bot_data["container"].settings,
                                  telegram_id=42, username="kartik")
    member = get_or_create_user(session, ctx.bot_data["container"].settings,
                                telegram_id=99, username="dave")
    not_mine = create_workspace(session, outsider, "Not Mine", is_private=True)
    start_conversation(session, member, not_mine)
    session.commit()
    ctx.args = ["@kartik", "editor"]
    upd = fake_update(text="/promote @kartik editor", tg_id=99, username="dave")
    await cmd_promote(upd, ctx)
    assert "Only the workspace owner" in " ".join(upd.effective_message.reply_text.texts)


@pytest.mark.asyncio
async def test_promote_rejects_invalid_role(owner_env):
    from app.bot.handlers import cmd_promote

    session, priv, ctx, owner = owner_env
    ctx.args = ["@someone", "owner"]
    upd = fake_update(text="/promote @someone owner")
    await cmd_promote(upd, ctx)
    assert "editor or viewer" in " ".join(upd.effective_message.reply_text.texts)
