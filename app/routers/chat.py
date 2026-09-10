import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from app.schemas import ChatRequest, ChatResponse
from app.security import api_key_dependency

router = APIRouter(prefix="/api/v1/workspaces/{slug}/chat",
                   dependencies=[Depends(api_key_dependency)])


@router.post("", response_model=ChatResponse)
async def chat(request: Request, slug: str, payload: ChatRequest):
    from app.db import SessionLocal
    from app.services.chat import run_chat
    from app.services.users import ensure_system_user
    from app.services.workspaces import workspace_by_slug

    settings = request.app.state.settings
    session_factory = SessionLocal(settings)
    with session_factory() as session:
        ws = workspace_by_slug(session, slug)
        if ws is None:
            raise HTTPException(status_code=404, detail="Unknown workspace")
        system = ensure_system_user(session)
        result = await asyncio.to_thread(
            run_chat, session, system, ws, payload.message,
            request.app.state.graph, settings)
        session.commit()
        return ChatResponse(answer=result.answer, sources=result.sources,
                            conversation_id=result.conversation_id, refused=result.refused)
