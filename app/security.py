import secrets

from fastapi import Request
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import User, Workspace, WorkspaceMember

MANAGE_ROLES = ("owner",)
INGEST_ROLES = ("owner", "editor")


def role_of(session: Session | None, workspace: Workspace, user: User) -> str | None:
    if user.is_platform_admin or workspace.owner_id == user.id:
        return "owner"
    if session is None:
        return None
    member = session.scalar(select(WorkspaceMember).where(
        WorkspaceMember.workspace_id == workspace.id, WorkspaceMember.user_id == user.id))
    return member.role if member else None


def can_view(workspace: Workspace, role: str | None) -> bool:
    return not workspace.is_private or role is not None


def can_ingest(workspace: Workspace, role: str | None) -> bool:
    return not workspace.is_private or role in INGEST_ROLES


def can_manage(workspace: Workspace, role: str | None) -> bool:
    return role in MANAGE_ROLES


def visible_workspaces(session: Session, user: User) -> list[Workspace]:
    member_ws_ids = select(WorkspaceMember.workspace_id).where(WorkspaceMember.user_id == user.id)
    return list(session.scalars(select(Workspace).where(
        or_(Workspace.is_private.is_(False), Workspace.id.in_(member_ws_ids))
    ).order_by(Workspace.created_at)))


def api_key_ok(settings: Settings, presented: str | None) -> bool:
    if not presented:
        return False
    return any(secrets.compare_digest(presented, k) for k in settings.api_key_set)


def api_key_dependency(request: Request) -> None:
    settings = request.app.state.settings
    if not api_key_ok(settings, request.headers.get("X-API-Key")):
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Invalid or missing API key")
