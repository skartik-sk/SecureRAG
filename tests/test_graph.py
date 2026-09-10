import json

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.config import Settings
from app.rag.graph import build_graph, thread_id_for

Q = "What is the delivery SLA?"


class FakeStore:
    def __init__(self, results):
        self.results = results  # list[(Document, distance)]
        self.queries = []

    def similarity_search_with_score(self, query, k=4):
        self.queries.append(query)
        return self.results[:k]


class FakeLLM:
    """Scripted responses. `grades` is a list of per-grade-call results (popped in
    order). Prompt-type detection uses distinctive markers from prompts.py:
    GRADE_PROMPT contains "JSON array", REWRITE_PROMPT contains "standalone search
    query", GENERATE_PROMPT contains neither."""

    def __init__(self, grades=(), answer="The SLA is 48 hours [1].", rewritten="SLA time"):
        self.grades = list(grades)
        self.answer = answer
        self.rewritten = rewritten
        self.prompts = []

    def invoke(self, prompt):
        text = str(prompt)
        self.prompts.append(text)
        if "JSON array" in text:
            idx = self.grades.pop(0) if self.grades else []
            return AIMessage(content=json.dumps(idx))
        if "standalone search query" in text:
            return AIMessage(content=self.rewritten)
        return AIMessage(content=self.answer)


def _doc(name, section=None):
    return Document(page_content=f"content of {name} " + "x " * 30,
                    metadata={"source_file": name, "section": section})


def _mk(store, llm, settings=None):
    s = settings or Settings(_env_file=None)
    return build_graph(llm, lambda slug: store, settings=s, checkpointer=MemorySaver())


CFG = {"configurable": {"thread_id": "t1"}}


def test_happy_path_with_citations():
    store = FakeStore([(_doc("sla.md", "Zones"), 0.2), (_doc("other.md"), 0.8)])
    llm = FakeLLM(grades=[[0]])
    out = _mk(store, llm).invoke({"question": Q, "rewritten": Q, "attempt": 0,
                                  "workspace_slug": "ws1", "history": []}, config=CFG)
    assert out["answer"] == "The SLA is 48 hours [1]."
    assert out["sources"] == [{"file": "sla.md", "section": "Zones"}]
    assert out["refused"] is None
    assert isinstance(out["history"][-1], AIMessage)
    assert any(str(m.content) == Q for m in out["history"] if isinstance(m, HumanMessage))


def test_grading_reranks_relevant_chunks_first():
    store = FakeStore([(_doc("a.md"), 0.1), (_doc("b.md"), 0.3), (_doc("c.md"), 0.5)])
    llm = FakeLLM(grades=[[0, 2]], answer="Use [1] and [2].")
    out = _mk(store, llm).invoke({"question": Q, "rewritten": Q, "attempt": 0,
                                  "workspace_slug": "ws1", "history": []}, config=CFG)
    ctx = llm.prompts[-1]
    # soft rerank: graded chunks (a, c) lead the context; ungraded b trails
    assert "content of a.md" in ctx and "content of c.md" in ctx and "content of b.md" in ctx
    assert ctx.index("content of a.md") < ctx.index("content of b.md")
    assert ctx.index("content of c.md") < ctx.index("content of b.md")
    assert {s["file"] for s in out["sources"]} == {"a.md", "c.md"}


def test_off_topic_guard_refuses_before_llm():
    store = FakeStore([(_doc("a.md"), 0.99)])
    llm = FakeLLM(grades=[[0]])
    out = _mk(store, llm).invoke({"question": "who won the world cup", "rewritten": "who won the world cup",
                                  "attempt": 0, "workspace_slug": "ws1", "history": []}, config=CFG)
    assert out["refused"] and out["answer"] == ""
    assert llm.prompts == []  # never called


def test_empty_workspace_refuses():
    store = FakeStore([])
    llm = FakeLLM(grades=[[0]])
    out = _mk(store, llm).invoke({"question": Q, "rewritten": Q, "attempt": 0,
                                  "workspace_slug": "ws1", "history": []}, config=CFG)
    assert "no documents" in out["refused"]


def test_rewrite_retry_recovers():
    store = FakeStore([(_doc("sla.md"), 0.2)])
    llm = FakeLLM(grades=[[], [0]], answer="placeholder")  # grade#1 empty, grade#2 relevant
    graph = _mk(store, llm)
    state = graph.invoke({"question": "tell me the sla", "rewritten": "tell me the sla",
                          "attempt": 0, "workspace_slug": "ws1", "history": []}, config=CFG)
    # guard + retrieve#1 use the raw question; retrieve#2 uses the rewritten query
    assert store.queries[0] == "tell me the sla"
    assert store.queries[1] == "tell me the sla"
    assert store.queries[2] == "SLA time"
    assert state["answer"] == "placeholder"
    assert state["sources"] == [{"file": "sla.md", "section": None}]


def test_second_empty_grade_refuses():
    store = FakeStore([(_doc("sla.md"), 0.2)])
    llm = FakeLLM(grades=[[], []])
    out = _mk(store, llm).invoke({"question": Q, "rewritten": Q, "attempt": 0,
                                  "workspace_slug": "ws1", "history": []}, config=CFG)
    assert out["refused"] and out["answer"] == ""


def test_grade_prompt_is_batch_json():
    store = FakeStore([(_doc("a.md"), 0.1), (_doc("b.md"), 0.2)])
    llm = FakeLLM(grades=[[0, 1]])
    _mk(store, llm).invoke({"question": Q, "rewritten": Q, "attempt": 0,
                            "workspace_slug": "ws1", "history": []}, config=CFG)
    grade_prompt = next(p for p in llm.prompts if "JSON array" in p)
    assert "0)" in grade_prompt and "1)" in grade_prompt  # numbered chunks


def test_memory_persists_across_invocations():
    store = FakeStore([(_doc("sla.md"), 0.2)])
    llm = FakeLLM(grades=[[0], [0]])
    graph = _mk(store, llm)
    cfg = {"configurable": {"thread_id": thread_id_for("c1")}}
    graph.invoke({"question": Q, "rewritten": Q, "attempt": 0,
                  "workspace_slug": "ws1", "history": []}, config=cfg)
    out = graph.invoke({"question": "and for zone 2?", "rewritten": "and for zone 2?",
                        "attempt": 0, "workspace_slug": "ws1", "history": []}, config=cfg)
    history = out["history"]
    contents = [m.content for m in history]
    assert Q in contents and "and for zone 2?" in contents  # same thread accumulated


def test_normalize_citations_fullwidth():
    from app.rag.graph import _normalize_citations

    assert _normalize_citations("Zones A and B【1】 take 2 days【12】.") == \
        "Zones A and B[1] take 2 days[12]."
    assert _normalize_citations("plain [3] stays") == "plain [3] stays"
