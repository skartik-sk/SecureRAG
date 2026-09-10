import logging
from dataclasses import dataclass

from telegram.ext import (
    Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters,
)

from app.bot import handlers
from app.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    session_factory: object
    graph: object


def build_application(settings: Settings, session_factory=None, graph=None) -> Application:
    builder = Application.builder().token(settings.telegram_bot_token)
    if settings.bot_mode == "webhook":
        builder = builder.updater(None)  # webhook mode never polls
    app = builder.build()

    from app.db import SessionLocal

    if session_factory is None:
        session_factory = SessionLocal(settings)
    app.bot_data["container"] = Container(settings, session_factory, graph)

    cmds = [
        ("start", handlers.cmd_start), ("help", handlers.cmd_help),
        ("workspaces", handlers.cmd_workspaces), ("workspace", handlers.cmd_workspace),
        ("newworkspace", handlers.cmd_newworkspace), ("new", handlers.cmd_new),
        ("resume", handlers.cmd_resume), ("demo", handlers.cmd_demo),
        ("invite", handlers.cmd_invite), ("promote", handlers.cmd_promote),
        ("kick", handlers.cmd_kick),
        ("public", handlers.cmd_public), ("private", handlers.cmd_private),
    ]
    for name, fn in cmds:
        app.add_handler(CommandHandler(name, fn))
    app.add_handler(CallbackQueryHandler(handlers.on_callback))
    app.add_handler(MessageHandler(filters.Document.ALL, handlers.on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.on_text))
    return app


BOT_COMMANDS = [
    ("start", "Pick a workspace and begin"),
    ("workspaces", "List workspaces"),
    ("workspace", "Current workspace info"),
    ("newworkspace", "Create a workspace — /newworkspace Name [private]"),
    ("new", "New conversation"),
    ("resume", "Past conversations"),
    ("demo", "Try the seeded demo workspace"),
    ("invite", "Add a member — /invite @user [editor|viewer]"),
    ("promote", "Change a member's role — /promote @user editor"),
    ("kick", "Remove a member — /kick @user (owner)"),
    ("public", "Make workspace public (owner)"),
    ("private", "Make workspace private (owner)"),
    ("help", "Show all commands"),
]


async def set_bot_commands(app: Application) -> None:
    from telegram import BotCommand

    await app.bot.set_my_commands([BotCommand(name, desc) for name, desc in BOT_COMMANDS])


async def start_polling(app: Application) -> None:
    await app.initialize()
    await app.start()
    await app.updater.start_polling(allowed_updates=["message", "callback_query"])
    logger.info("telegram polling started")


async def stop_application(app: Application) -> None:
    if app.updater:
        await app.updater.stop()
    await app.stop()
    await app.shutdown()
