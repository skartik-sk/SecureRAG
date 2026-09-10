import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.schemas import DocumentOut
from app.security import api_key_dependency

router = APIRouter(prefix="/api/v1/workspaces/{slug}/documents",
                   dependencies=[Depends(api_key_dependency)])
MAX_BYTES = 20 * 1024 * 1024


@router.get("", response_model=list[DocumentOut])
def list_documents(request: Request, slug: str):
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import Document

    settings = request.app.state.settings
    with SessionLocal(settings)() as session:
        from app.services.workspaces import workspace_by_slug

        ws = workspace_by_slug(session, slug)
        if ws is None:
            raise HTTPException(status_code=404, detail="Unknown workspace")
        rows = session.scalars(select(Document).where(Document.workspace_id == ws.id)
                               .order_by(Document.created_at.desc()))
        return [DocumentOut(id=d.id, filename=d.filename, status=d.status,
                            chunk_count=d.chunk_count, error=d.error, source=d.source)
                for d in rows]


@router.post("")
async def upload_document(request: Request, slug: str, file: UploadFile):
    from app.db import SessionLocal
    from app.rag.vectorstore import get_store
    from app.services.ingest import (
        SUPPORTED_EXTS, create_document_row, ingest_document_async, save_original,
    )
    from app.services.workspaces import workspace_by_slug

    settings = request.app.state.settings
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else ""
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 20 MB limit")

    session_factory = SessionLocal(settings)
    with session_factory() as session:
        ws = workspace_by_slug(session, slug)
        if ws is None:
            raise HTTPException(status_code=404, detail="Unknown workspace")
        doc = create_document_row(session, ws, file.filename, source="api", byte_size=len(data))
        session.commit()
        doc_id, slug_value = doc.id, ws.slug
    save_original(settings, slug_value, doc_id, ext, data)
    store = get_store(settings, slug_value)
    task = ingest_document_async(session_factory, doc_id, data, ext, store, settings,
                                 source_file=file.filename)
    if settings.inline_ingest:
        # Serverless freezes the function once the response is sent, so the
        # work has to finish inside this request. Failures are already
        # recorded on the document row by ingest_document_sync.
        try:
            chunks = await task
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"Ingest failed: {str(e)[:200]}")
        return {"document_id": doc_id, "status": "ready", "chunks": chunks}
    asyncio.create_task(task)
    return JSONResponse({"document_id": doc_id, "status": "processing"}, status_code=202)
