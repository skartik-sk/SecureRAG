from app.rag.chunker import MIN_CHUNK_CHARS, chunk_markdown, chunk_plain_text

MD = """# Delivery

Main policy text that is definitely long enough to survive the minimum filter.

## Zones

Zone details that are also long enough to survive the minimum length filter here.

## Tiny

no

# Refunds

Refund policy body text, again comfortably above the cutoff for inclusion.
"""


def test_markdown_header_split_with_sections():
    chunks = chunk_markdown(MD)
    sections = [c.metadata["section"] for c in chunks]
    assert sections[0] == "Delivery"
    assert "Zones" in sections
    assert "no" not in [c.page_content for c in chunks]  # tiny chunk dropped


def test_small_header_content_merged_into_oversize_rule():
    chunks = chunk_markdown(MD)
    texts = [c.page_content for c in chunks]
    assert all(len(c.strip()) >= MIN_CHUNK_CHARS for c in texts)


def test_headerless_markdown_falls_back():
    md = "word " * 500  # no headers at all
    chunks = chunk_markdown(md)
    assert len(chunks) >= 2  # 2500 chars → split into >1 pieces
    assert all(c.metadata["section"] is None for c in chunks)


def test_plain_text_chunks():
    chunks = chunk_plain_text("hello world. " * 400)
    assert len(chunks) >= 2
    assert all(c.metadata["section"] is None for c in chunks)


def test_empty_and_tiny_inputs():
    assert chunk_markdown("") == []
    assert chunk_markdown("short") == []
