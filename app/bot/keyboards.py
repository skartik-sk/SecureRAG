from telegram import InlineKeyboardButton, InlineKeyboardMarkup

DEMO_QUESTIONS = [
    "What is the delivery SLA for each zone?",
    "Which goods are classified as special goods?",
    "What is the refund window for returned items?",
]


def workspace_keyboard(workspaces, current_id=None) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(
        text=f"{'✓ ' if w.id == current_id else ''}{w.name}",
        callback_data=f"ws:{w.id}")] for w in workspaces]
    return InlineKeyboardMarkup(rows)


def conversations_keyboard(conversations) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text=c.title[:32], callback_data=f"cv:{c.id}")]
            for c in conversations]
    return InlineKeyboardMarkup(rows)


def demo_keyboard() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text=q, callback_data=f"dq:{i}")]
            for i, q in enumerate(DEMO_QUESTIONS)]
    return InlineKeyboardMarkup(rows)


def new_chat_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("📝 I'll paste content", callback_data="nc:content"),
        InlineKeyboardButton("❓ Just chat", callback_data="nc:skip"),
    ]])


def feedback_keyboard(feedback_id: str) -> InlineKeyboardMarkup:
    """👍/👎 on a bot answer; the rating lands in answer_feedback."""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("👍", callback_data=f"fb:{feedback_id}:1"),
        InlineKeyboardButton("👎", callback_data=f"fb:{feedback_id}:0"),
    ]])
