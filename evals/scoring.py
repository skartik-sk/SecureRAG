"""Outcome classification and refusal metrics for the eval suite.

Pure logic only — no LLM, no DB — so everything here is unit-testable.
"""

import re
from collections import defaultdict
from enum import Enum


class Outcome(str, Enum):
    REFUSED = "refused"                    # guard or pipeline refusal
    DONT_KNOW = "dont_know"                # generator fell back to "I don't know."
    ANSWERED_CITED = "answered_cited"      # answered with [n] citations + sources
    ANSWERED_UNCITED = "answered_uncited"  # answered but failed the citation sanity check
    EMPTY = "empty"                        # no answer text at all


_CITATION = re.compile(r"\[(\d+)\]")


def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("'", "")
    return re.sub(r"\s+", " ", text.strip().rstrip(".")).strip()


def classify(state: dict) -> Outcome:
    """Map a final graph state (refused / answer / sources) to an Outcome."""
    if state.get("refused"):
        return Outcome.REFUSED
    answer = (state.get("answer") or "").strip()
    if not answer:
        return Outcome.EMPTY
    if _normalize(answer) == "i dont know":
        return Outcome.DONT_KNOW
    if _CITATION.search(answer) and state.get("sources"):
        return Outcome.ANSWERED_CITED
    return Outcome.ANSWERED_UNCITED


_ABSTAIN = frozenset({Outcome.REFUSED, Outcome.DONT_KNOW, Outcome.EMPTY})


def abstained(outcome: Outcome) -> bool:
    """True when the bot declined to answer (any refusal flavor)."""
    return outcome in _ABSTAIN


def refusal_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _slice(records: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for r in records:
        should_refuse = r["case"].expect == "refuse"
        did_refuse = abstained(r["outcome"])
        if should_refuse and did_refuse:
            tp += 1
        elif did_refuse:
            fp += 1
        elif should_refuse:
            fn += 1
        else:
            tn += 1
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(records)}
    out.update(refusal_metrics(tp, fp, fn))
    return out


def score_run(records: list[dict]) -> dict:
    """Aggregate confusion + metrics overall, per workspace and per category,
    plus the three failure buckets worth reading line by line."""
    leaks = [r for r in records if r["case"].expect == "refuse" and not abstained(r["outcome"])]
    over_refusals = [r for r in records
                     if r["case"].expect == "answer" and abstained(r["outcome"])]
    uncited = [r for r in records
               if r["case"].expect == "answer" and r["outcome"] is Outcome.ANSWERED_UNCITED]
    by_ws, by_cat = defaultdict(list), defaultdict(list)
    for r in records:
        by_ws[r["case"].workspace].append(r)
        by_cat[r["case"].category].append(r)
    return {"overall": _slice(records),
            "by_workspace": {k: _slice(v) for k, v in sorted(by_ws.items())},
            "by_category": {k: _slice(v) for k, v in sorted(by_cat.items())},
            "leaks": leaks, "over_refusals": over_refusals, "uncited_answers": uncited}


def sweep_thresholds(distances: list[float], current: float) -> list[float]:
    """Data-calibrated sweep points: deciles of the observed best distances —
    the scale varies by embedding model, so fixed ranges like 0.8–1.0 can be
    entirely off — plus the currently configured threshold. Deduped, sorted."""
    finite = sorted(d for d in distances if d < float("inf"))
    if not finite:
        return [round(current, 2)]
    n = len(finite)
    points = {round(finite[int(round((n - 1) * i / 9))], 2) for i in range(10)}
    points.add(round(current, 2))
    return sorted(points)


def sweep_guard(rows: list[tuple[float, int]], thresholds: list[float]) -> list[dict]:
    """Re-score the guard decision (distance > threshold ⇒ refuse) at each
    threshold, against ground-truth labels. rows = (best_distance, label 0|1)."""
    out = []
    for t in thresholds:
        tp = fp = fn = 0
        for dist, should_refuse in rows:
            predicted = dist > t
            if predicted and should_refuse:
                tp += 1
            elif predicted:
                fp += 1
            elif should_refuse:
                fn += 1
        row = {"threshold": t, "tp": tp, "fp": fp, "fn": fn}
        row.update(refusal_metrics(tp, fp, fn))
        out.append(row)
    return out
