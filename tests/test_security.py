from app.config import Settings
from app.models import User, Workspace
from app.security import (
    api_key_ok, can_ingest, can_manage, can_view, role_of, visible_workspaces,
)


def _ws(private=False, owner_id="owner-1"):
    return Workspace(slug="s", name="S", is_private=private, owner_id=owner_id)


def _user(admin=False, uid="u-1"):
    return User(id=uid, display_name="u", is_platform_admin=admin)


def test_public_viewable_by_anyone():
    assert can_view(_ws(), role_of(None, _ws(), _user()))


def test_private_hidden_from_non_member():
    assert not can_view(_ws(private=True), role_of(None, _ws(private=True), _user()))


def test_owner_role_via_owner_id():
    assert role_of(None, _ws(), _user(uid="owner-1")) == "owner"


def test_platform_admin_is_owner_everywhere():
    ws = _ws(private=True)
    assert role_of(None, ws, _user(admin=True)) == "owner"


def test_member_roles(session, test_settings):
    from app.models import WorkspaceMember

    u, owner = _user(uid="m-1"), _user(uid="owner-1")
    w = _ws(private=True)
    session.add_all([u, owner, w])
    session.flush()
    session.add(WorkspaceMember(workspace_id=w.id, user_id=u.id, role="editor"))
    session.flush()
    assert role_of(session, w, u) == "editor"
    assert can_ingest(w, "editor")
    assert not can_manage(w, "editor")


def test_ingest_matrix():
    assert can_ingest(_ws(), "viewer")            # public: anyone
    assert can_ingest(_ws(private=True), "editor")
    assert not can_ingest(_ws(private=True), "viewer")
    assert not can_ingest(_ws(private=True), None)


def test_manage_owner_only():
    assert can_manage(_ws(), "owner")
    assert not can_manage(_ws(), "editor")
    assert not can_manage(_ws(), None)


def test_visible_workspaces(session, test_settings):
    from app.models import WorkspaceMember

    u, other = _user(uid="vis-1"), _user(uid="vis-2")
    pub, priv_mine, priv_other = (_ws(owner_id="vis-1"), _ws(private=True, owner_id="vis-1"),
                                  _ws(private=True, owner_id="vis-1"))
    priv_mine.slug, priv_other.slug, pub.slug = "pm", "po", "pub"
    session.add_all([u, other, pub, priv_mine, priv_other])
    session.flush()
    session.add(WorkspaceMember(workspace_id=priv_mine.id, user_id=u.id, role="viewer"))
    session.flush()
    seen = {w.slug for w in visible_workspaces(session, u)}
    assert seen == {"pub", "pm"}


def test_api_key_ok():
    s = Settings(_env_file=None, api_keys="k1, k2")
    assert api_key_ok(s, "k1") and api_key_ok(s, "k2")
    assert not api_key_ok(s, "nope") and not api_key_ok(s, None) and not api_key_ok(s, "")


def test_api_key_dependency(session, test_settings):
    from types import SimpleNamespace

    from fastapi import HTTPException

    from app.security import api_key_dependency

    app = SimpleNamespace(state=SimpleNamespace(
        settings=Settings(_env_file=None, api_keys="good")))
    request_ok = SimpleNamespace(headers={"X-API-Key": "good"}, app=app)
    request_bad = SimpleNamespace(headers={"X-API-Key": "nope"}, app=app)
    assert api_key_dependency(request_ok) is None
    try:
        api_key_dependency(request_bad)
        raised = False
    except HTTPException as e:
        raised = e.status_code == 401
    assert raised
