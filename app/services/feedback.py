from sqlalchemy.orm import Session

from app.models import AnswerFeedback, User, Workspace

_RATINGS = {"1": 1, "0": -1}  # callback_data carries "1" (👍) or "0" (👎)


def record_answer(session: Session, user: User, workspace: Workspace,
                  conversation_id: str | None, question: str, answer: str) -> AnswerFeedback:
    """Persist a bot answer as unrated feedback the user can later 👍/👎."""
    fb = AnswerFeedback(user_id=user.id, workspace_id=workspace.id,
                        conversation_id=conversation_id, question=question, answer=answer)
    session.add(fb)
    session.flush()
    return fb


def rate_feedback(session: Session, user: User, feedback_id: str,
                  rating: int | str) -> AnswerFeedback | None:
    """Set 👍 (1 / "1") or 👎 (0 / "0") on the user's own feedback row.

    Returns None when the row doesn't exist or belongs to someone else;
    raises ValueError for a rating that isn't 1 or 0.
    """
    key = str(rating)
    if key not in _RATINGS:
        raise ValueError(f"rating must be 1 or 0, got {rating!r}")
    fb = session.get(AnswerFeedback, feedback_id)
    if fb is None or fb.user_id != user.id:
        return None
    fb.rating = _RATINGS[key]
    session.flush()
    return fb
