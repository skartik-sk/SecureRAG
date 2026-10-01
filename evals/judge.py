"""LLM-judged faithfulness for answered eval cases (--judge flag).

For every ANSWERED_CITED record, one judge call asks whether every claim in the
answer is supported by the chunk texts that were actually given to the
generator. Pure parsing/aggregation lives here; the runner wires the LLM.
"""

import json
import re

from evals.scoring import Outcome

JUDGE_PROMPT = """You are grading a RAG answer for faithfulness: every claim in \
the answer must be supported by the numbered context snippets, which are the \
only source of truth.

Question: {question}

Context:
{context}

Answer:
{answer}

Reply with ONLY a JSON object: {{"faithful": true, "reason": "<one sentence>"}}
Set faithful=false if any claim is missing from, contradicted by, or goes \
beyond the context. Do not use outside knowledge to judge content true."""

_JSON = re.compile(r"\{.*\}", re.S)


def parse_verdict(raw: str) -> tuple[bool | None, str]:
    """(faithful, reason) from a judge reply; (None, "") when unparseable."""
    match = _JSON.search(raw)
    if not match:
        return None, ""
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None, ""
    if not isinstance(data, dict) or not isinstance(data.get("faithful"), bool):
        return None, ""
    return data["faithful"], str(data.get("reason", ""))


def judge_records(llm, records: list[dict]) -> list[dict]:
    """Attach `faithful` / `judge_reason` to each cited answer, in place.

    Returns one error entry per judge call that raised; unparseable verdicts
    are not errors — they record faithful=None.
    """
    errors = []
    for r in records:
        if r["outcome"] is not Outcome.ANSWERED_CITED:
            continue
        context = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(r.get("contexts", [])))
        prompt = JUDGE_PROMPT.format(question=r["case"].question, context=context,
                                     answer=r.get("answer", ""))
        try:
            reply = llm.invoke(prompt)
        except Exception as e:  # noqa: BLE001 — one bad judge call must not kill the run
            errors.append({"case": r["case"], "error": f"{type(e).__name__}: {e}"})
            continue
        r["faithful"], r["judge_reason"] = parse_verdict(getattr(reply, "content", reply))
    return errors


def faithfulness_summary(records: list[dict]) -> dict:
    judged = [r for r in records
              if r["outcome"] is Outcome.ANSWERED_CITED and "faithful" in r]
    return {"judged": len(judged),
            "faithful": sum(1 for r in judged if r["faithful"] is True),
            "unfaithful": sum(1 for r in judged if r["faithful"] is False),
            "unparsed": sum(1 for r in judged if r["faithful"] is None)}
