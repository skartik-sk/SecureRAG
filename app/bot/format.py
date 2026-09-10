import html

LIMIT = 4096


def esc(s: str) -> str:
    return html.escape(str(s or ""))


def split_message(text: str, limit: int = LIMIT) -> list[str]:
    text = text or ""
    if len(text) <= limit:
        return [text] if text else [""]
    parts, current = [], ""
    for para in text.split("\n"):
        while len(para) > limit:  # single oversized line: hard slice
            if current:
                parts.append(current)
                current = ""
            parts.append(para[:limit])
            para = para[limit:]
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) > limit:
            parts.append(current)
            current = para
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def format_answer(answer: str, sources: list[dict]) -> str:
    out = answer or ""
    if sources:
        lines = []
        for i, s in enumerate(sources, 1):
            line = f"[{i}] {esc(s.get('file') or 'unknown')}"
            if s.get("section"):
                line += f" — {esc(s['section'])}"
            lines.append(line)
        out += "\n\n<b>Sources:</b>\n" + "\n".join(lines)
    return out
