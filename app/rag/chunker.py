from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

MIN_CHUNK_CHARS = 40
OVERSIZE_CHARS = 1200
_HEADERS = [("#", "h1"), ("##", "h2")]
_FALLBACK = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def _section(md_meta: dict) -> str | None:
    return md_meta.get("h2") or md_meta.get("h1")


def _clean(docs: list[Document]) -> list[Document]:
    return [
        Document(page_content=d.page_content, metadata={"section": _section(d.metadata)})
        for d in docs
        if len(d.page_content.strip()) >= MIN_CHUNK_CHARS
    ]


def chunk_markdown(md: str) -> list[Document]:
    if not md or not md.strip():
        return []
    docs = MarkdownHeaderTextSplitter(_HEADERS, strip_headers=False).split_text(md)
    if not docs:
        docs = _FALLBACK.create_documents([md])
    else:
        oversized = [d for d in docs if len(d.page_content) > OVERSIZE_CHARS]
        for d in oversized:
            docs.remove(d)
        for d in oversized:
            docs.extend(_FALLBACK.split_documents([d]))
    return _clean(docs)


def chunk_plain_text(text: str) -> list[Document]:
    if not text or not text.strip():
        return []
    docs = _FALLBACK.create_documents([text])
    return [
        Document(page_content=d.page_content, metadata={"section": None})
        for d in docs
        if len(d.page_content.strip()) >= MIN_CHUNK_CHARS
    ]
