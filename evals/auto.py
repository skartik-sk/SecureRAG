"""Auto mode: generate an eval set for ANY workspace from its own chunks.

Answerable questions are written by the LLM from real chunks (reliable
labels — the chunk they came from answers them). Refusal coverage comes
from a fixed universal off-topic list that is safe for every corpus.
Near-miss cases are deliberately NOT auto-generated: their ground truth
depends on corpus content and LLM-generated labels there are noisy.
"""

import json
import re
from collections import OrderedDict

from evals.dataset import Case

# General-knowledge questions no NovaCart-style corpus answers.
OFF_TOPIC_QUESTIONS = [
    "Who won the FIFA World Cup 2022?",
    "What is the capital of Australia?",
    "How do I make sourdough bread at home?",
    "Explain quantum entanglement in simple terms.",
    "Who is the CEO of Amazon?",
    "What's the best programming language to learn first?",
    "What's the weather like in Tokyo today?",
    "Recommend a good mystery novel.",
]

QUESTION_PROMPT = """Below is an excerpt from this workspace's documents.

Excerpt:
{excerpt}

Write {n} distinct customer questions that this excerpt answers. Each question must
be answerable using ONLY this excerpt. Reply as a numbered list, one question per
line, no preamble."""


def pick_chunks(docs, want: int) -> list:
    """Spread chunk sampling across files: dedupe (file, section), then
    round-robin one chunk per file per round until `want` is reached."""
    seen, by_file = set(), OrderedDict()
    for d in docs:
        key = (d.metadata.get("source_file"), d.metadata.get("section"))
        if key in seen or not (d.page_content or "").strip():
            continue
        seen.add(key)
        by_file.setdefault(key[0], []).append(d)
    picked: list = []
    rounds = max((len(v) for v in by_file.values()), default=0)
    for i in range(rounds):
        for lst in by_file.values():
            if len(picked) >= want:
                return picked
            if i < len(lst):
                picked.append(lst[i])
    return picked


def parse_questions(raw: str) -> list[str]:
    """Parse an LLM reply into clean question strings (numbered list,
    bullets, or a JSON array). Anything unparsable yields []."""
    raw = raw.strip()
    if raw.startswith("["):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        items = [q for q in data if isinstance(q, str)] if isinstance(data, list) else []
    else:
        items = re.findall(r"^[ \t]*(?:\d+[.)]|[-*])[ \t]*(.+?)[ \t]*$", raw, re.M)
    return [q.strip().strip('"') for q in items if q.strip()]


def generate_cases(llm, sample_fn, slug: str, per_chunk: int = 2,
                   want_chunks: int = 10, off_topic: int = 8) -> tuple[list[Case], list[dict]]:
    """Build (cases, provenance) for one workspace. `sample_fn(slug)` must
    return that workspace's chunks; `llm` is any chat-model with .invoke."""
    picked = pick_chunks(sample_fn(slug), want=want_chunks)
    cases: list[Case] = []
    provenance: list[dict] = []
    for d in picked:
        prompt = QUESTION_PROMPT.format(excerpt=d.page_content[:1200], n=per_chunk)
        questions = parse_questions(_text(llm.invoke(prompt)))[:per_chunk]
        meta = {"source_file": d.metadata.get("source_file"),
                "section": d.metadata.get("section")}
        for q in questions:
            cases.append(Case(question=q, workspace=slug, expect="answer",
                              category="answerable"))
            provenance.append({"question": q, **meta})
    for q in OFF_TOPIC_QUESTIONS[:off_topic]:
        cases.append(Case(question=q, workspace=slug, expect="refuse",
                          category="off_topic"))
    return cases, provenance


def _text(reply) -> str:
    content = getattr(reply, "content", reply)
    return content if isinstance(content, str) else str(content)
