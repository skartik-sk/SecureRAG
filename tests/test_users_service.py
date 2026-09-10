from sqlalchemy import select

from app.config import Settings
from app.models import User
from app.services.users import ensure_system_user, get_or_create_user


def _s(admin_id=0):
    return Settings(_env_file=None, telegram_admin_id=admin_id)


def test_creates_user(session):
    u = get_or_create_user(session, _s(), telegram_id=777, username="k", display_name="Kartik")
    session.flush()
    assert u.id and u.telegram_id == 777 and u.telegram_username == "k"
    assert u.is_platform_admin is False


def test_second_contact_returns_same_row(session):
    a = get_or_create_user(session, _s(), telegram_id=777, username="k", display_name="Kartik")
    b = get_or_create_user(session, _s(), telegram_id=777, username="k2", display_name="K Tikk")
    assert a.id == b.id and b.telegram_username == "k2" and b.display_name == "K Tikk"


def test_admin_flag_from_settings(session):
    u = get_or_create_user(session, _s(admin_id=42), telegram_id=42)
    assert u.is_platform_admin is True


def test_ensure_system_user_idempotent(session):
    a = ensure_system_user(session)
    b = ensure_system_user(session)
    assert a.id == b.id and a.telegram_id is None
    assert session.scalar(select(User).where(User.telegram_id.is_(None))).id == a.id
