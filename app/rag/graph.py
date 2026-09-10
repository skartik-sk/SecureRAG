import json
import re
from typing import Annotated, Callable, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import Settings
from app.rag.prompts import GENERATE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT, format_history


class RAGState(TypedDict):
    workspace_slug: str
    question: str
    rewritten: str
    attempt: int
    retrieved: list
    distances: list[float]
    relevant: list
    answer: str
    sources: list[dict]
    refused: str | None
    history: Annotated[list, add_messages]


def thread_id_for(conversation_id: str) -> str:
    return f"conv-{conversation_id}"


def _parse_grade(raw: str, n: int) -> list[int]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[[\d\s,]*\]", raw)
        data = json.loads(match.group()) if match else []
    return [i for i in data if isinstance(i, int) and 0 <= i < n]


def _numbered(docs: list[Document]) -> str:
    return "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


def _text(reply) -> str:
    """Unwrap a chat-model reply (AIMessage) to plain text."""
    content = getattr(reply, "content", reply)
    return content if isinstance(content, str) else str(content)


def _normalize_citations(text: str) -> str:
    """Models occasionally emit fullwidth 【1】 citations — canonicalize to [1]
    so answer text and source mapping agree."""
    return text.replace("【", "[").replace("】", "]")


def build_graph(llm, store_for_slug: Callable, settings: Settings | None = None,
                checkpointer=None):
    s = settings or Settings(_env_file=None)

    def guard(state: RAGState) -> dict:
        store = store_for_slug(state["workspace_slug"])
        best = store.similarity_search_with_score(state["question"], k=1)
        update = {"history": [HumanMessage(content=state["question"])],
                  "answer": "", "sources": [], "refused": None}
        if not best:
            update["refused"] = ("This workspace has no documents yet. Send me a file or "
                                 "paste some content first (/new).")
            return update
        if best[0][1] > s.off_topic_distance:
            update["refused"] = "I can only answer questions about this workspace's documents."
            return update
        return update

    def retrieve(state: RAGState) -> dict:
        store = store_for_slug(state["workspace_slug"])
        results = store.similarity_search_with_score(state["rewritten"], k=s.retrieve_k)
        docs = [d for d, _ in results]
        dists = [dist for _, dist in results]
        return {"retrieved": docs, "distances": dists}

    def grade(state: RAGState) -> dict:
        docs = state["retrieved"]
        listing = "\n\n".join(f"{i}) {d.page_content[:600]}" for i, d in enumerate(docs))
        prompt = GRADE_PROMPT.format(question=state["rewritten"], chunks=listing)
        idx = set(_parse_grade(_text(llm.invoke(prompt)), len(docs)))
        relevant = [d for i, (d, dist) in enumerate(zip(docs, state["distances"]))
                    if i in idx and dist <= s.off_topic_distance]
        return {"relevant": relevant}

    def route_after_grade(state: RAGState) -> str:
        if state["relevant"]:
            return "generate"
        if state["attempt"] == 0:
            return "rewrite"
        return "refuse"

    def rewrite(state: RAGState) -> dict:
        prompt = REWRITE_PROMPT.format(history=format_history(state["history"]),
                                       question=state["question"])
        new_query = _text(llm.invoke(prompt)).strip().strip('"')
        return {"rewritten": new_query or state["question"], "attempt": 1}

    def refuse(state: RAGState) -> dict:
        return {"refused": state["refused"] or "I couldn't find anything about that in this "
                                                   "workspace's documents.",
                "answer": "", "sources": []}

    def route_after_guard(state: RAGState) -> str:
        return "refuse" if state["refused"] else "retrieve"

    def generate(state: RAGState) -> dict:
        docs = state["relevant"][: s.top_n]
        prompt = GENERATE_PROMPT.format(
            history=format_history(state["history"]),
            context=_numbered(docs),
            question=state["rewritten"],
        )
        answer = _normalize_citations(_text(llm.invoke(prompt)).strip())
        cited = {int(n) - 1 for n in re.findall(r"\[(\d+)\]", answer)
                 if 0 < int(n) <= len(docs)}
        used = [docs[i] for i in sorted(cited)] or docs
        sources, seen = [], set()
        for d in used:
            key = (d.metadata.get("source_file"), d.metadata.get("section"))
            if key not in seen:
                seen.add(key)
                sources.append({"file": key[0], "section": key[1]})
        return {"answer": answer, "sources": sources,
                "history": [AIMessage(content=answer)]}

    g = StateGraph(RAGState)
    g.add_node("guard", guard)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("refuse", refuse)
    g.add_node("generate", generate)
    g.add_edge(START, "guard")
    g.add_conditional_edges("guard", route_after_guard,
                            {"retrieve": "retrieve", "refuse": "refuse"})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"generate": "generate", "rewrite": "rewrite", "refuse": "refuse"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile(checkpointer=checkpointer)


_production = None
_production_pool = None


def close_production() -> None:
    global _production, _production_pool
    if _production_pool is not None:
        _production_pool.close()
    _production = None
    _production_pool = None


def get_graph(settings: Settings | None = None):
    """Production singleton: real Groq, real stores, Postgres-backed checkpointer."""
    global _production, _production_pool
    if _production is None:
        from psycopg_pool import ConnectionPool

        from app.config import get_settings
        from app.rag.embeddings import get_embeddings
        from app.rag.llm import get_llm
        from app.rag.vectorstore import get_store

        s = settings or get_settings()
        pool = ConnectionPool(s.database_url_plain, open=True, check=False, timeout=10,
                              min_size=1, max_size=3, kwargs={"autocommit": True})
        from langgraph.checkpoint.postgres import PostgresSaver

        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        _production_pool = pool
        _production = build_graph(
            get_llm(s),
            lambda slug: get_store(s, slug, get_embeddings(s)),
            settings=s,
            checkpointer=checkpointer,
        )
    return _production
