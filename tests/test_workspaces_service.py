from app.services.users import ensure_system_user
from app.services.workspaces import (
    create_workspace, slugify, unique_slug, workspace_by_slug, workspace_stats,
)


def test_slugify():
    assert slugify("Delivery Policy!") == "delivery-policy"
    assert slugify("  A  B   C ") == "a-b-c"
    assert slugify("héllo *** world") == "hllo-world"
    assert slugify("") == "workspace"
    assert slugify("x" * 100) == "x" * 40


def test_unique_slug(session, test_settings):
    owner = ensure_system_user(session)
    create_workspace(session, owner, "Same Name")
    w2 = create_workspace(session, owner, "Same Name")
    assert w2.slug == "same-name-2"
    w3 = create_workspace(session, owner, "Same Name")
    assert w3.slug == "same-name-3"


def test_create_workspace_adds_owner_member(session, test_settings):
    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Private Stuff", description="d", is_private=True)
    from app.models import WorkspaceMember

    m = session.get(WorkspaceMember, (w.id, owner.id))
    assert m.role == "owner" and w.is_private and w.slug == "private-stuff"


def test_workspace_by_slug(session, test_settings):
    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Findable")
    assert workspace_by_slug(session, "findable").id == w.id
    assert workspace_by_slug(session, "missing") is None


def test_workspace_stats(session, test_settings):
    from app.models import Document

    owner = ensure_system_user(session)
    w = create_workspace(session, owner, "Stats")
    session.add_all([
        Document(workspace_id=w.id, filename="a.md", file_ext="md", status="ready",
                 chunk_count=3, source="seed"),
        Document(workspace_id=w.id, filename="b.md", file_ext="md", status="failed",
                 chunk_count=0, source="seed"),
    ])
    session.flush()
    assert workspace_stats(session, w) == {"documents": 1, "chunks": 3}
