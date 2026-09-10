from langchain_core.messages import AIMessage, HumanMessage

GENERATE_PROMPT = """You are a documentation assistant. Answer the question using ONLY the context below.

Rules:
- Cite the context numbers you used inline, like [1] or [2].
- If the context does not contain the answer, reply exactly: I don't know.
- Be concise.

{history}

Context:
{context}

Question: {question}
"""

GRADE_PROMPT = """You are grading retrieved document chunks for relevance.

Question: {question}

Chunks:
{chunks}

Which chunk numbers contain information needed to answer the question?
Reply with ONLY a JSON array of numbers, e.g. [0,2]. Reply [] if none are relevant.
"""

REWRITE_PROMPT = """Rewrite the user's latest message as one standalone search query.
Resolve pronouns using the conversation. Reply with the query only, no quotes.

{history}

Latest message: {question}
"""


def format_history(messages, max_chars=1500) -> str:
    """Render recent chat turns for prompt injection (oldest → newest, truncated)."""
    lines = [f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
             for m in messages[-8:] if isinstance(m, (HumanMessage, AIMessage))]
    out = "\n".join(lines)
    return "Conversation so far:\n" + out if out else ""
