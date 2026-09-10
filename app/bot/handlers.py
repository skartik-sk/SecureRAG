import asyncio
import logging

from telegram import Update
from telegram.ext import ContextTypes

from app.bot.format import esc, format_answer, split_message
from app.bot.keyboards import (
    DEMO_QUESTIONS, conversations_keyboard, demo_keyboard, new_chat_keyboard,
    workspace_keyboard,
)
from app.models import User as UserModel
from app.models import Workspace, WorkspaceMember
from app.security import can_ingest, can_manage, can_view, role_of, visible_workspaces
from app.services.chat import list_conversations, resume_conversation, run_chat, start_conversation
from app.services.ingest import SUPPORTED_EXTS, ingest_document_sync, ingest_text_sync
from app.services.users import get_or_create_user
from app.services.workspaces import create_workspace, workspace_by_slug, workspace_stats
from seed import DEMO_SLUG

logger = logging.getLogger(__name__)
PASTE_MIN_CHARS = 120
MAX_BYTES = 20 * 1024 * 1024


def _container(context: ContextTypes.DEFAULT_TYPE):
    return context.application.bot_data["container"]


def safe_handler(fn):
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            return await fn(update, context)
        except Exception:  # noqa: BLE001
            logger.exception("handler %s failed", fn.__name__)
            target = update.effective_message
            if target:
                await target.reply_text("Something went wrong — please try again.")

    wrapper.__name__ = fn.__name__
    return wrapper


def _user_and_settings(context, session, update):
    container = _container(context)
    tg = update.effective_user
    user = get_or_create_user(session, container.settings, telegram_id=tg.id,
                              username=tg.username, display_name=tg.first_name)
    return user, container.settings


def _current_workspace(session, user):
    if user.current_workspace_id:
        return session.get(Workspace, user.current_workspace_id)
    return None


@safe_handler
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with _container(context).session_factory() as session:
        user, _settings = _user_and_settings(context, session, update)
        session.commit()
        wss = visible_workspaces(session, user)
        current = _current_workspace(session, user)
        if current:
            text = (f"Hi {esc(user.display_name)}! I answer questions from document workspaces.\n"
                    f"Current workspace: <b>{esc(current.name)}</b>")
        else:
            text = f"Hi {esc(user.display_name)}! Pick a workspace below, or /newworkspace to create one."
        await update.effective_message.reply_text(
            text, parse_mode="HTML",
            reply_markup=workspace_keyboard(wss, current.id if current else None))


@safe_handler
async def cmd_help(update, context):
    await update.effective_message.reply_text(
        "<b>Commands</b>\n"
        "/workspaces — list workspaces\n"
        "/workspace — current workspace info\n"
        "/newworkspace &lt;name&gt; [private] — create a workspace\n"
        "/new — new conversation (optionally add content)\n"
        "/resume — past conversations\n"
        "/demo — try the seeded policy workspace\n"
        "/invite @user [editor|viewer] — add a member (private workspaces)\n"
        "/promote @user editor — change a member's role (owner)\n"
        "/public /private — toggle privacy (owner)\n"
        "Send a PDF/DOCX/PPTX/XLSX/HTML/MD file to index it. Or just ask a question.",
        parse_mode="HTML")


@safe_handler
async def cmd_workspaces(update, context):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        session.commit()
        wss = visible_workspaces(session, user)
        if not wss:
            await update.effective_message.reply_text("No workspaces yet — /newworkspace to create one.")
            return
        await update.effective_message.reply_text(
            "Your workspaces:", reply_markup=workspace_keyboard(wss, user.current_workspace_id))


@safe_handler
async def cmd_workspace(update, context):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        if ws is None:
            await update.effective_message.reply_text("Pick a workspace first — /workspaces")
            return
        stats = workspace_stats(session, ws)
        await update.effective_message.reply_text(
            f"<b>{esc(ws.name)}</b> (<code>{esc(ws.slug)}</code>)\n"
            f"Documents: {stats['documents']} · Chunks: {stats['chunks']} · "
            f"{'🔒 private' if ws.is_private else '🌍 public'}", parse_mode="HTML")


@safe_handler
async def cmd_newworkspace(update, context):
    parts = list(context.args or [])
    is_private = bool(parts) and parts[-1].lower() == "private"
    if is_private:
        parts = parts[:-1]
    name = " ".join(parts).strip()
    if not name or len(name) > 40:
        await update.effective_message.reply_text("Usage: /newworkspace <name> [private] (name ≤ 40 chars)")
        return
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = create_workspace(session, user, name, is_private=is_private)
        start_conversation(session, user, ws)
        session.commit()
        await update.effective_message.reply_text(
            f"✅ Workspace <b>{esc(ws.name)}</b> created ({'🔒' if ws.is_private else '🌍'}). "
            "You're in — send a file or /new.", parse_mode="HTML")


@safe_handler
async def cmd_new(update, context):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        if ws is None:
            await update.effective_message.reply_text("Pick a workspace first — /workspaces")
            return
        start_conversation(session, user, ws)
        session.commit()
        await update.effective_message.reply_text(
            f"🆕 New conversation in <b>{esc(ws.name)}</b>.\n"
            "Add knowledge? Paste content (120+ chars) or send a file — or just ask questions.",
            parse_mode="HTML", reply_markup=new_chat_keyboard())


@safe_handler
async def cmd_resume(update, context):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        convos = list_conversations(session, user)
        session.commit()
        if not convos:
            await update.effective_message.reply_text("No past conversations yet.")
            return
        await update.effective_message.reply_text(
            "📜 Your recent conversations:", reply_markup=conversations_keyboard(convos))


@safe_handler
async def cmd_demo(update, context):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = workspace_by_slug(session, DEMO_SLUG)
        if ws is None:
            await update.effective_message.reply_text(
                "Demo workspace isn't seeded yet. Run: python3 scripts/seed.py")
            return
        start_conversation(session, user, ws)
        session.commit()
        await update.effective_message.reply_text(
            f"🧪 Demo workspace <b>{esc(ws.name)}</b> ready. Tap a question:", parse_mode="HTML",
            reply_markup=demo_keyboard())


@safe_handler
async def cmd_invite(update, context):
    args = list(context.args or [])
    username = (args[0] if args else "").lstrip("@").strip()
    role = (args[1].strip().lower() if len(args) > 1 else "viewer")
    if role not in ("editor", "viewer"):
        await update.effective_message.reply_text("Role must be editor or viewer.")
        return
    if not username:
        await update.effective_message.reply_text(
            "Usage: /invite @username [editor|viewer]")
        return
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        target = session.query(UserModel).filter(
            UserModel.telegram_username == username).one_or_none()
        if ws is None or target is None:
            await update.effective_message.reply_text(
                "They need to /start the bot first, and you must be in a workspace.")
            return
        role_of_user = role_of(session, ws, user)
        if not ws.is_private:
            await update.effective_message.reply_text(
                "This workspace is public — everyone can already use it.")
            return
        if role_of_user not in ("owner", "editor"):
            await update.effective_message.reply_text("Only owners/editors can invite.")
            return
        exists = session.get(WorkspaceMember, (ws.id, target.id))
        if exists:
            await update.effective_message.reply_text(
                f"@{username} is already a member — /promote @{username} {role} to change their role.")
            return
        session.add(WorkspaceMember(workspace_id=ws.id, user_id=target.id, role=role))
        session.commit()
        await update.effective_message.reply_text(
            f"✅ @{esc(username)} added to <b>{esc(ws.name)}</b> as {role}.", parse_mode="HTML")


@safe_handler
async def cmd_promote(update, context):
    args = list(context.args or [])
    username = (args[0] if args else "").lstrip("@").strip()
    role = (args[1].strip().lower() if len(args) > 1 else "")
    if not username or role not in ("editor", "viewer"):
        await update.effective_message.reply_text("Role must be editor or viewer. "
                                                  "Usage: /promote @username editor")
        return
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        target = session.query(UserModel).filter(
            UserModel.telegram_username == username).one_or_none()
        if ws is None or target is None:
            await update.effective_message.reply_text(
                "They need to /start the bot first, and you must be in a workspace.")
            return
        if not can_manage(ws, role_of(session, ws, user)):
            await update.effective_message.reply_text("Only the workspace owner can do that.")
            return
        member = session.get(WorkspaceMember, (ws.id, target.id))
        if member is None:
            await update.effective_message.reply_text(
                f"@{username} isn't a member — /invite @{username} first.")
            return
        if member.role == "owner":
            await update.effective_message.reply_text("You can't change the owner's role.")
            return
        member.role = role
        session.commit()
        article = "an" if role == "editor" else "a"
        await update.effective_message.reply_text(
            f"✅ @{esc(username)} is now {article} <b>{role}</b> of <b>{esc(ws.name)}</b>.",
            parse_mode="HTML")


@safe_handler
async def cmd_public(update, context):
    await _set_privacy(update, context, False)


@safe_handler
async def cmd_private(update, context):
    await _set_privacy(update, context, True)


async def _set_privacy(update, context, private: bool):
    with _container(context).session_factory() as session:
        user, _ = _user_and_settings(context, session, update)
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        if ws is None or not can_manage(ws, role_of(session, ws, user)):
            await update.effective_message.reply_text("Only the workspace owner can do that.")
            return
        ws.is_private = private
        session.commit()
        await update.effective_message.reply_text(
            f"<b>{esc(ws.name)}</b> is now {'🔒 private' if private else '🌍 public'}.", parse_mode="HTML")


def _role(session, ws, user):
    return role_of(session, ws, user)


@safe_handler
async def on_callback(update, context):
    query = update.callback_query
    await query.answer()
    prefix, _, value = query.data.partition(":")
    with _container(context).session_factory() as session:
        user, settings = _user_and_settings(context, session, update)
        if prefix == "ws":
            ws = session.get(Workspace, value)
            if ws is None or not can_view(ws, _role(session, ws, user)):
                await query.message.reply_text("🔒 That workspace is private — ask its owner for access.")
                return
            start_conversation(session, user, ws)
            session.commit()
            stats = workspace_stats(session, ws)
            await query.message.reply_text(
                f"✅ Switched to <b>{esc(ws.name)}</b> ({stats['documents']} docs). Ask away!",
                parse_mode="HTML")
        elif prefix == "cv":
            conv = resume_conversation(session, user, value)
            session.commit()
            if conv is None:
                await query.message.reply_text("That conversation isn't available.")
                return
            await query.message.reply_text(
                f"📜 Resumed <b>{esc(conv.title)}</b> — continue asking.", parse_mode="HTML")
        elif prefix == "nc":
            if value == "content":
                user.pending_action = "awaiting_content"
                session.commit()
                await query.message.reply_text(
                    f"Send me the content as your next message ({PASTE_MIN_CHARS}+ chars) — "
                    "I'll chunk and index it.")
            else:
                await query.message.reply_text("OK — just type your question.")
        elif prefix == "dq":
            ws = workspace_by_slug(session, DEMO_SLUG)
            question = DEMO_QUESTIONS[int(value)]
            start_conversation(session, user, ws)
            result = await asyncio.to_thread(run_chat, session, user, ws, question,
                                             _container(context).graph)
            session.commit()
            for part in split_message(format_answer(result.answer, result.sources)):
                await query.message.reply_text(part, parse_mode="HTML")


@safe_handler
async def on_document(update, context):
    doc = update.effective_message.document
    name = doc.file_name or "file"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    with _container(context).session_factory() as session:
        user, settings = _user_and_settings(context, session, update)
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        if ws is None:
            await update.effective_message.reply_text("Pick a workspace first — /workspaces")
            return
        if not can_ingest(ws, _role(session, ws, user)):
            await update.effective_message.reply_text("🔒 You can't add files to this private workspace.")
            return
        from app.services.ingest import find_inflight_duplicate

        dupe = find_inflight_duplicate(session, ws.id, name, doc.file_size or 0)
        if dupe is not None:
            await update.effective_message.reply_text(
                f"⏳ {esc(name)} is still being processed — hang on a moment.", parse_mode="HTML")
            return
        if ext not in SUPPORTED_EXTS:
            await update.effective_message.reply_text(
                f"Unsupported type .{ext or '?'} — send one of: {', '.join(sorted(SUPPORTED_EXTS))}")
            return
        if (doc.file_size or 0) > MAX_BYTES:
            await update.effective_message.reply_text("File is over the 20 MB Telegram limit.")
            return
        from app.services.ingest import create_document_row

        row = create_document_row(session, ws, name, source="telegram", uploaded_by=user.id,
                                  byte_size=doc.file_size or 0)
        session.commit()
        slug, doc_id = ws.slug, row.id
    await update.effective_message.reply_text(f"📥 Processing {esc(name)}…", parse_mode="HTML")
    tg_file = await context.bot.get_file(doc.file_id)
    data = await tg_file.download_as_bytearray()
    from app.rag.vectorstore import get_store

    store = get_store(settings, slug)
    try:
        def _work():
            return ingest_document_sync(_container(context).session_factory, doc_id,
                                        bytes(data), ext, store, settings, source_file=name)
        chunks = await asyncio.to_thread(_work)
    except Exception as e:  # noqa: BLE001
        logger.exception("ingest failed for %s", name)
        await update.effective_message.reply_text(
            f"❌ Couldn't index {esc(name)}: {esc(str(e)[:200])}", parse_mode="HTML")
        return
    await update.effective_message.reply_text(
        f"✅ {chunks} chunks from {esc(name)} are searchable.", parse_mode="HTML")


@safe_handler
async def on_text(update, context):
    text = (update.effective_message.text or "").strip()
    if not text:
        return
    with _container(context).session_factory() as session:
        user, settings = _user_and_settings(context, session, update)
        if user.pending_action == "awaiting_content":
            ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
            if ws is None:
                await update.effective_message.reply_text("Pick a workspace first — /workspaces")
                return
            if len(text) < PASTE_MIN_CHARS:
                user.pending_action = None
                session.commit()
                await update.effective_message.reply_text(
                    f"That was under {PASTE_MIN_CHARS} chars, so I treated it as a question.")
            else:
                from app.rag.vectorstore import get_store

                store = get_store(settings, ws.slug)

                def _work():
                    return ingest_text_sync(_container(context).session_factory, ws,
                                            f"Notes {update.effective_message.message_id}", text,
                                            store, uploaded_by=user.id)
                try:
                    row = await asyncio.to_thread(_work)
                except ValueError as e:
                    await update.effective_message.reply_text(str(e))
                    return
                user.pending_action = None
                session.commit()
                await update.effective_message.reply_text(
                    f"✅ {row.chunk_count} chunks indexed from your notes — ask away.")
                return
        ws = session.get(Workspace, user.current_workspace_id) if user.current_workspace_id else None
        if ws is None:
            await update.effective_message.reply_text("Pick a workspace first — /workspaces")
            return
        if not can_view(ws, _role(session, ws, user)):
            await update.effective_message.reply_text(
                "🔒 This workspace is private — ask its owner for access.")
            return
        result = await asyncio.to_thread(run_chat, session, user, ws, text,
                                         _container(context).graph)
        session.commit()
    sources = [] if result.refused else result.sources
    for part in split_message(format_answer(result.answer, sources)):
        await update.effective_message.reply_text(part, parse_mode="HTML")
