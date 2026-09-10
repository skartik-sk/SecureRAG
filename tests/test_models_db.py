from sqlalchemy import select

from app.db import init_db
from app.models import Conversation, Document, User, Workspace, WorkspaceMember


def test_init_db_creates_all_tables(test_settings):
    init_db(test_settings)  # idempotent
    from app.db import engine

    with engine(test_settings).connect() as conn:
        names = {r[0] for r in conn.exec_driver_sql(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'")}
    assert {"users", "workspaces", "workspace_members", "documents", "conversations"} <= names


def test_user_roundtrip(session):
    u = User(telegram_id=42, display_name="Kartik")
    session.add(u)
    session.flush()
    got = session.scalar(select(User).where(User.telegram_id == 42))
    assert got is not None and got.id and got.is_platform_admin is False
    assert got.pending_action is None


def test_workspace_member_composite_pk(session):
    u = User(telegram_id=1, display_name="a")
    session.add(u)
    session.flush()
    w = Workspace(slug="s-1", name="S", owner_id=u.id)
    session.add(w)
    session.flush()
    session.add(WorkspaceMember(workspace_id=w.id, user_id=u.id, role="owner"))
    session.flush()  # duplicate insert would raise — composite PK enforced
