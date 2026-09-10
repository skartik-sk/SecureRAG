from app.bot.format import esc, format_answer, split_message


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


def test_format_answer_with_sources():
    out = format_answer("SLA is 48h [1].", [{"file": "sla.md", "section": "Zones"}])
    assert "SLA is 48h [1]." in out
    assert "<b>Sources:</b>" in out
    assert "[1] sla.md — Zones" in out


def test_format_answer_escapes_source_names():
    out = format_answer("a", [{"file": "<script>.md", "section": None}])
    assert "<script>" not in out.split("Sources:")[-1]


def test_format_answer_no_sources_block_when_empty():
    assert "Sources" not in format_answer("plain", [])
