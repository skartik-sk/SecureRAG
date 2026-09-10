import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.db import SessionLocal, init_db

logger = logging.getLogger(__name__)


async def ensure_started(app: FastAPI) -> None:
    """Idempotent startup: safe to call from lifespan (docker/local) and from
    the lazy-init middleware — Vercel's Python runtime does not reliably run
    ASGI lifespan events, so the first request triggers initialization there.
    State lives on the app instance so several apps can coexist (tests)."""
    if getattr(app.state, "started", False):
        return
    lock = getattr(app.state, "start_lock", None)
    if lock is None:
        lock = app.state.start_lock = asyncio.Lock()
    async with lock:
        if getattr(app.state, "started", False):
            return
        settings: Settings = app.state.settings
        init_db(settings)
        with SessionLocal(settings)() as session:
            from app.services.users import ensure_system_user

            ensure_system_user(session)
            session.commit()

        # Only inject components that weren't already provided (lets tests
        # pre-assign app.state.graph / app.state.ptb_app before this runs).
        if not hasattr(app.state, "graph"):
            if settings.groq_api_key:
                from app.rag.graph import get_graph

                app.state.graph = get_graph(settings)
            else:
                app.state.graph = None
                logger.warning("GROQ_API_KEY not set — chat endpoints will fail until it is provided")

        if not hasattr(app.state, "ptb_app"):
            app.state.ptb_app = None
            if app.state.graph is None and settings.bot_mode != "disabled":
                logger.warning("bot not started: GROQ_API_KEY missing — the bot needs the RAG graph")
            elif settings.bot_mode != "disabled" and settings.telegram_bot_token:
                from app.bot.setup import build_application, set_bot_commands, start_polling

                ptb = build_application(settings, SessionLocal(settings), app.state.graph)
                if settings.bot_mode == "polling":
                    await start_polling(ptb)
                elif settings.bot_mode == "webhook":
                    await ptb.initialize()
                    if settings.webhook_url:
                        await ptb.bot.set_webhook(
                            f"{settings.webhook_url.rstrip('/')}/telegram/webhook",
                            secret_token=settings.telegram_webhook_secret,
                            allowed_updates=["message", "callback_query"])
                    else:
                        logger.warning("BOT_MODE=webhook but WEBHOOK_URL is empty — "
                                       "register it manually or set WEBHOOK_URL")
                    await set_bot_commands(ptb)
                app.state.ptb_app = ptb
            else:
                logger.info("bot disabled (BOT_MODE=%s, token set=%s)",
                            settings.bot_mode, bool(settings.telegram_bot_token))
        app.state.started = True
        logger.info("app started (vercel=%s)", bool(os.getenv("VERCEL")))

class LazyInitMiddleware:
    """Pure-ASGI middleware that runs ensure_started before the first request."""

    def __init__(self, asgi_app, target: FastAPI):
        self.asgi_app = asgi_app
        self.target = target

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            await ensure_started(self.target)
        await self.asgi_app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_started(app)
    try:
        yield
    finally:
        await shutdown(app)


async def shutdown(app: FastAPI) -> None:
    ptb = getattr(app.state, "ptb_app", None)
    if ptb is not None and hasattr(ptb, "shutdown"):
        from app.bot.setup import stop_application

        await stop_application(ptb)
    from app.rag.graph import close_production

    close_production()
    from app.db import engine

    engine(app.state.settings).dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    app = FastAPI(title="Telegram Multi-Workspace RAG", lifespan=lifespan)
    app.state.settings = s
    app.add_middleware(LazyInitMiddleware, target=app)

    from app.routers import chat, documents, health, telegram_webhook, workspaces

    app.include_router(health.router)
    app.include_router(workspaces.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(telegram_webhook.router)
    return app


app = create_app()
