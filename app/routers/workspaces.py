from fastapi import APIRouter, Depends, Request
from sqlalchemy import select

from app.schemas import WorkspaceCreate, WorkspaceOut
from app.security import api_key_dependency
from app.services.users import ensure_system_user
from app.services.workspaces import create_workspace, workspace_stats

router = APIRouter(prefix="/api/v1", dependencies=[Depends(api_key_dependency)])


@router.get("/workspaces", response_model=list[WorkspaceOut])
def list_workspaces(request: Request):
    from app.db import SessionLocal
    from app.models import Workspace

    settings = request.app.state.settings
    # API keys are platform-level: REST sees all workspaces.
    with SessionLocal(settings)() as session:
        all_ws = session.scalars(select(Workspace).order_by(Workspace.created_at))
        return [WorkspaceOut(
            slug=w.slug, name=w.name, description=w.description, is_private=w.is_private,
            **workspace_stats(session, w)) for w in all_ws]


@router.post("/workspaces", response_model=WorkspaceOut, status_code=201)
def create_workspace_ep(request: Request, payload: WorkspaceCreate):
    from app.db import SessionLocal

    settings = request.app.state.settings
    with SessionLocal(settings)() as session:
        system = ensure_system_user(session)
        w = create_workspace(session, system, payload.name,
                             payload.description, payload.is_private)
        session.commit()
        return WorkspaceOut(slug=w.slug, name=w.name, description=w.description,
                            is_private=w.is_private, documents=0, chunks=0)
