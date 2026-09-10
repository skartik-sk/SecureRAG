import html
import re

LIMIT = 4096


def esc(s: str) -> str:
    return html.escape(str(s or ""))


def md_to_telegram_html(text: str) -> str:
    """Convert the markdown subset the LLM emits (**bold**, *italic*, `code`,
    # headings, - bullets, ``` fences) to Telegram HTML, escaping everything
    else. Without this, parse_mode=HTML renders markdown literally — or fails
    to send when the answer contains stray < or &."""
    text = text or ""
    parts = re.split(r"```(\w*)\n?(.*?)```", text, flags=re.S)
    # parts: [text, lang, code, text, lang, code, ..., text]
    out = []
    for i, part in enumerate(parts):
        if i % 3 == 2:  # fenced code block
            out.append(f"<pre>{esc(part.rstrip())}</pre>")
            continue
        chunk = esc(part)
        chunk = re.sub(r"`([^`\n]+)`", lambda m: f"<code>{m.group(1)}</code>", chunk)
        chunk = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", chunk)
        chunk = re.sub(r"__([^_\n]+)__", r"<b>\1</b>", chunk)
        chunk = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", chunk)
        chunk = re.sub(r"(?m)^#{1,6}\s+(.+)$", r"<b>\1</b>", chunk)
        chunk = re.sub(r"(?m)^(\s*)[-*]\s+", r"\1• ", chunk)
        out.append(chunk)
    return "".join(out)


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
    """Raw markdown answer + sources block, ready for split_message() then
    md_to_telegram_html() per part (splitting after conversion could cut an
    HTML tag in half)."""
    out = answer or ""
    if sources:
        lines = []
        for i, s in enumerate(sources, 1):
            line = f"[{i}] {s.get('file') or 'unknown'}"
            if s.get("section"):
                line += f" — {s['section']}"
            lines.append(line)
        out += "\n\n**Sources:**\n" + "\n".join(lines)
    return out


def format_answer_parts(answer: str, sources: list[dict], limit: int = LIMIT) -> list[str]:
    """Split raw text to the Telegram limit, then render each part as HTML."""
    return [md_to_telegram_html(p) for p in split_message(format_answer(answer, sources), limit)]
