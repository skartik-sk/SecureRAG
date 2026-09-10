from app.bot.format import (
    esc, format_answer, format_answer_parts, md_to_telegram_html, split_message,
)


def test_escapes_html():
    assert esc("<b> & 'x'") == "&lt;b&gt; &amp; &#x27;x&#x27;"


def test_split_short_message_unchanged():
    assert split_message("hello") == ["hello"]


def test_split_long_message_on_newlines():
    para = "word " * 300
    text = "\n\n".join([para] * 6)  # ~11k chars
    parts = split_message(text)
    assert len(parts) >= 3
    assert all(len(p) <= 4096 for p in parts)
    assert "\n\n".join(parts).count("word") == text.count("word")


def test_split_hard_slices_when_no_newlines():
    text = "x" * 10_000
    parts = split_message(text)
    assert sum(len(p) for p in parts) == 10_000
    assert all(len(p) <= 4096 for p in parts)


def test_md_converter_bold_italic_headings_bullets():
    out = md_to_telegram_html("# Policy\n- **Zones**: two days\n- *same-day* before 11:00")
    assert "<b>Policy</b>" in out
    assert "• <b>Zones</b>: two days" in out
    assert "• <i>same-day</i> before 11:00" in out
    assert "**" not in out and "&#x27;" not in out.split("<b>")[0]


def test_md_converter_escapes_html_and_keeps_citations():
    out = md_to_telegram_html("a < b & [1]")
    assert "a &lt; b &amp; [1]" in out


def test_md_converter_code():
    out = md_to_telegram_html("run `npm i` or:\n```\n<p>raw</p>\n```")
    assert "<code>npm i</code>" in out
    assert "<pre>&lt;p&gt;raw&lt;/p&gt;</pre>" in out


def test_format_answer_parts_renders_html_sources():
    parts = format_answer_parts("SLA is 48h [1].", [{"file": "sla.md", "section": "Zones"}])
    assert len(parts) == 1
    assert "SLA is 48h [1]." in parts[0]
    assert "<b>Sources:</b>" in parts[0]
    assert "[1] sla.md — Zones" in parts[0]


def test_format_answer_parts_never_leak_raw_markdown():
    parts = format_answer_parts("**Bold** and *ital*.", [])
    assert parts == ["<b>Bold</b> and <i>ital</i>."]


def test_format_answer_escapes_source_names():
    parts = format_answer_parts("a", [{"file": "<script>.md", "section": None}])
    assert "<script>" not in parts[0]


def test_format_answer_no_sources_block_when_empty():
    assert "Sources" not in format_answer("plain", [])
