from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import User

SYSTEM_DISPLAY_NAME = "platform"


def get_or_create_user(
    session: Session,
    settings: Settings,
    telegram_id: int,
    username: str | None = None,
    display_name: str | None = None,
) -> User:
    user = session.scalar(select(User).where(User.telegram_id == telegram_id))
    if user is None:
        user = User(
            telegram_id=telegram_id,
            telegram_username=username,
            display_name=display_name or f"User {telegram_id}",
            is_platform_admin=(telegram_id == settings.telegram_admin_id),
        )
        session.add(user)
        session.flush()
        return user
    if username:
        user.telegram_username = username
    if display_name:
        user.display_name = display_name
    session.flush()
    return user


def ensure_system_user(session: Session) -> User:
    user = session.scalar(select(User).where(User.telegram_id.is_(None)))
    if user is None:
        user = User(display_name=SYSTEM_DISPLAY_NAME)
        session.add(user)
        session.flush()
    return user
