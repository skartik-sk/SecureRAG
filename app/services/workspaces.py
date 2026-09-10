import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Document, User, Workspace, WorkspaceMember


def slugify(name: str) -> str:
    ascii_only = "".join(c for c in name.lower() if c.isascii())
    s = re.sub(r"[^a-z0-9]+", "-", ascii_only).strip("-")
    return (s or "workspace")[:40].strip("-") or "workspace"


def unique_slug(session: Session, base: str) -> str:
    slug, n = base, 1
    while session.scalar(select(Workspace).where(Workspace.slug == slug)) is not None:
        n += 1
        slug = f"{base}-{n}"
    return slug


def create_workspace(
    session: Session,
    owner: User,
    name: str,
    description: str | None = None,
    is_private: bool = False,
) -> Workspace:
    ws = Workspace(
        slug=unique_slug(session, slugify(name)),
        name=name, description=description, is_private=is_private, owner_id=owner.id,
    )
    session.add(ws)
    session.flush()
    session.add(WorkspaceMember(workspace_id=ws.id, user_id=owner.id, role="owner"))
    session.flush()
    return ws


def workspace_by_slug(session: Session, slug: str) -> Workspace | None:
    return session.scalar(select(Workspace).where(Workspace.slug == slug))


def workspace_stats(session: Session, workspace: Workspace) -> dict:
    rows = session.execute(
        select(
            Document.status,
            func.count(Document.id),
            func.coalesce(func.sum(Document.chunk_count), 0),
        )
        .where(Document.workspace_id == workspace.id)
        .group_by(Document.status)
    ).all()
    ready_docs = sum(int(count) for status, count, _ in rows if status == "ready")
    total_chunks = sum(int(chunks) for _, _, chunks in rows)
    return {"documents": ready_docs, "chunks": total_chunks}
