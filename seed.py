import logging
from pathlib import Path

from sqlalchemy import select

from app.config import Settings
from app.models import Document
from app.services.ingest import create_document_row, parse_file
from app.services.users import ensure_system_user
from app.services.workspaces import create_workspace, workspace_by_slug

logger = logging.getLogger(__name__)

SEED_WORKSPACES = [
    ("delivery-policy", "Delivery Policy", [
        "delivery_policy.md", "delivery_zones_and_timeframes.pdf",
        "order_tracking_and_status.pptx", "public_user_delivery_terms.html",
        "special_goods_delivery.html",
    ]),
    ("returns-and-refunds", "Returns and Refunds", ["returns_and_refunds.md"]),
    ("shipping-charges", "Shipping Charges", [
        "shipping_charges_matrix.docx", "delivery_sla_matrix.xlsx",
    ]),
]
DEMO_SLUG = "delivery-policy"


def _store_for(settings: Settings, slug: str):
    from app.rag.vectorstore import get_store

    return get_store(settings, slug)


def _session_factory(settings: Settings):
    from app.db import SessionLocal

    return SessionLocal(settings)


def seed_all(corpus_dir: str, settings: Settings, notify=None) -> dict[str, int]:
    counts: dict[str, int] = {}
    corpus = Path(corpus_dir)
    factory = _session_factory(settings)
    for slug, name, files in SEED_WORKSPACES:
        with factory() as session:
            system = ensure_system_user(session)
            ws = workspace_by_slug(session, slug)
            if ws is None:
                ws = create_workspace(session, system, name)
            done = {d.filename for d in session.scalars(
                select(Document).filter_by(workspace_id=ws.id, status="ready"))}
            total = 0
            for filename in files:
                if filename in done:
                    continue
                path = corpus / filename
                if not path.exists():
                    logger.warning("seed: missing %s", path)
                    continue
                data = path.read_bytes()
                chunks = parse_file(data, path.suffix.lstrip("."))
                row = create_document_row(session, ws, filename, source="seed",
                                          byte_size=len(data))
                for c in chunks:
                    c.metadata["document_id"] = row.id
                    c.metadata["source_file"] = filename
                _store_for(settings, slug).add_documents(chunks)
                row.status, row.chunk_count = "ready", len(chunks)
                session.commit()
                total += len(chunks)
                if notify:
                    notify(f"{slug}: +{len(chunks)} chunks from {filename}")
            session.commit()
        counts[slug] = total
    return counts
