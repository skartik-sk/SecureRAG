"""Live eval runner: python3 -m evals.run

  python3 -m evals.run                                 # fixed set, 3 demo workspaces
  python3 -m evals.run --auto --workspace <slug>       # generate a set for any workspace
  python3 -m evals.run --dataset <file.json>           # re-run a saved (auto) dataset
  python3 -m evals.run --sweep                         # + guard threshold sweep table
  python3 -m evals.run --seed                          # seed demo workspaces first

Runs the real production graph against the configured (local-dev) database,
scores refusal accuracy + answer sanity, prints a report and writes JSON to
evals/reports/. Needs GROQ_API_KEY, EMBEDDINGS_API_KEY and a seeded DB.
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPORTS = Path(__file__).parent / "reports"
_PROBE_QUERIES = ("delivery", "refund", "charges", "policy", "order",
                  "customer", "rules", "fees", "return", "tracking")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="RAG refusal + answer-sanity evals")
    p.add_argument("--auto", action="store_true",
                   help="generate the eval set from a workspace's own chunks")
    p.add_argument("--workspace", help="workspace slug (required with --auto)")
    p.add_argument("--dataset", help="path to a saved dataset JSON instead of the fixed set")
    p.add_argument("--seed", action="store_true", help="seed the demo workspaces first")
    p.add_argument("--sweep", action="store_true",
                   help="also score the guard distance threshold from 0.80 to 1.00")
    return p.parse_args(argv)


def load_cases(args):
    from evals.dataset import Case, fixed_cases

    if args.dataset:
        raw = json.loads(Path(args.dataset).read_text())
        return [Case(**c) for c in raw["cases"]]
    if args.auto:
        if not args.workspace:
            sys.exit("--auto needs --workspace <slug>")
        return None  # generated after preflight
    return fixed_cases()


def preflight(settings, needed_slugs):
    import sqlalchemy as sa

    from app.db import SessionLocal
    from app.models import Document
    from app.services.workspaces import workspace_by_slug

    missing = []
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if not settings.embeddings_api_key:
        missing.append("EMBEDDINGS_API_KEY")
    if missing:
        sys.exit(f"Missing env: {', '.join(missing)} — see .env.example")
    counts = {}
    with SessionLocal(settings)() as session:
        for slug in needed_slugs:
            ws = workspace_by_slug(session, slug)
            if ws is None:
                counts[slug] = None
                continue
            n = session.scalar(sa.select(sa.func.count(Document.id)).where(
                Document.workspace_id == ws.id, Document.status == "ready"))
            counts[slug] = n
    empty = [s for s, n in counts.items() if not n]
    if empty:
        sys.exit(f"Workspace(s) not seeded: {', '.join(empty)} — run: python3 -m evals.run --seed")
    return counts


def sample_chunks(settings, slug):
    from app.rag.vectorstore import get_store

    store = get_store(settings, slug)
    seen, docs = set(), []
    for q in _PROBE_QUERIES:
        for d, _ in store.similarity_search_with_score(q, k=12):
            if d.page_content not in seen:
                seen.add(d.page_content)
                docs.append(d)
    return docs


def guard_distances(settings, records):
    """Best cosine distance per case — the signal the guard threshold acts on."""
    from app.rag.vectorstore import get_store

    stores = {}
    rows = []
    for r in records:
        slug = r["case"].workspace
        if slug not in stores:
            stores[slug] = get_store(settings, slug)
        best = stores[slug].similarity_search_with_score(r["case"].question, k=1)
        rows.append(best[0][1] if best else float("inf"))
    return rows


def run_cases(settings, cases):
    from app.rag.graph import get_graph

    graph = get_graph()
    from evals.scoring import classify

    records, errors = [], []
    for i, case in enumerate(cases, 1):
        t0 = time.time()
        try:
            state = graph.invoke(
                {"question": case.question, "rewritten": case.question, "attempt": 0,
                 "workspace_slug": case.workspace, "history": []},
                config={"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:12]}"}},
            )
            outcome = classify(state)
            records.append({"case": case, "outcome": outcome,
                            "refused_text": state.get("refused"),
                            "answer": state.get("answer", ""),
                            "sources": state.get("sources", []),
                            "latency_s": time.time() - t0})
            print(f"[{i}/{len(cases)}] {case.workspace} {outcome.value:18s} "
                  f"{time.time() - t0:4.1f}s  {case.question[:60]}", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad case must not kill the run
            errors.append({"case": case, "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(cases)}] {case.workspace} ERROR: {type(e).__name__}: {e}",
                  flush=True)
    return records, errors


def print_report(summary, records, errors, sweep=None):
    from evals.scoring import abstained

    def line(label, s):
        refused_ok = s["tp"] + s["fn"]
        answered_ok = s["tn"]
        print(f"  {label:<28} n={s['n']:<3} refused {s['tp']}/{refused_ok}  "
              f"answered {answered_ok}/{s['tn'] + s['fp']}  "
              f"P {s['precision']:.3f}  R {s['recall']:.3f}")

    print("\n=== Results ===")
    for slug, s in summary["by_workspace"].items():
        line(slug, s)
    print("  " + "-" * 74)
    line("OVERALL", summary["overall"])

    print("\nBy category:")
    for cat, s in summary["by_category"].items():
        leaks = s["fp"] + s["fn"] if cat != "answerable" else s["fp"]
        print(f"  {cat:<16} n={s['n']:<3} wrong={leaks}")

    cited = sum(r["outcome"].value == "answered_cited" for r in records)
    uncited = sum(r["outcome"].value == "answered_uncited" for r in records)
    idk = sum(r["outcome"].value == "dont_know" for r in records)
    refused = sum(r["outcome"].value == "refused" for r in records)
    print(f"\nAnswer sanity: {cited} cited answers · {uncited} uncited answers · "
          f"{idk} 'I don't know' · {refused} refusals")

    for title, bucket in (("LEAKED (should have refused, answered anyway)",
                           summary["leaks"]),
                          ("OVER-REFUSED (should have answered, refused)",
                           summary["over_refusals"]),
                          ("UNCITED ANSWERS", summary["uncited_answers"])):
        if bucket:
            print(f"\n✗ {title}:")
            for r in bucket:
                c = r["case"]
                detail = r.get("answer", "") or r.get("refused_text", "")
                print(f"  [{c.workspace}/{c.category}] \"{c.question}\"\n"
                      f"    → {detail[:140]}")
    if errors:
        print(f"\n✗ {len(errors)} case(s) errored (excluded from metrics):")
        for e in errors:
            print(f"  [{e['case'].workspace}] \"{e['case'].question[:60]}\" → {e['error']}")

    if sweep:
        print("\nGuard sweep (guard-only: best distance > t ⇒ refuse):")
        print("  t     TP    FP    FN    P      R")
        for row in sweep:
            print(f"  {row['threshold']:<6}{row['tp']:<5} {row['fp']:<5} {row['fn']:<5} "
                  f"{row['precision']:.3f}  {row['recall']:.3f}")


def _plain(r):
    c = r["case"]
    return {"question": c.question, "workspace": c.workspace, "expect": c.expect,
            "category": c.category, "outcome": r["outcome"].value,
            "refused_text": r.get("refused_text"), "answer": r.get("answer"),
            "sources": r.get("sources"), "latency_s": round(r.get("latency_s", 0), 2)}


def save_report(settings, mode, summary, records, errors, sweep, extra_meta):
    REPORTS.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    report = {
        "meta": {"mode": mode, "generated_at": datetime.now(timezone.utc).isoformat(),
                 "model": settings.groq_model, "embedding_model": settings.embedding_model,
                 "off_topic_distance": settings.off_topic_distance, **extra_meta},
        "summary": {k: v for k, v in summary.items()},
        "records": [_plain(r) for r in records],
        "errors": [{"question": e["case"].question, "workspace": e["case"].workspace,
                    "error": e["error"]} for e in errors],
        "sweep": sweep or [],
    }
    path = REPORTS / f"{ts}-{mode}.json"
    from enum import Enum

    path.write_text(json.dumps(
        report, indent=2,
        default=lambda o: o.value if isinstance(o, Enum) else o.__dict__))
    return path


def main(argv=None):
    args = parse_args(argv)
    from evals.scoring import score_run, sweep_guard

    from app.config import get_settings

    settings = get_settings()
    mode = "fixed"
    if args.auto:
        mode = f"auto-{args.workspace}"
    elif args.dataset:
        mode = f"dataset-{Path(args.dataset).stem}"

    cases = load_cases(args)
    if args.seed:
        from seed import seed_all

        print("Seeding demo workspaces…", flush=True)
        seed_all("my_docs_folder", settings, notify=print)
    if args.auto:
        from evals.auto import generate_cases

        from app.rag.llm import get_llm

        preflight(settings, [args.workspace])
        print(f"Generating eval set for '{args.workspace}' from its chunks…", flush=True)
        cases, provenance = generate_cases(get_llm(settings),
                                           lambda slug: sample_chunks(settings, slug),
                                           args.workspace)
        dataset_path = save_dataset(mode, cases, provenance)
        print(f"Saved generated dataset → {dataset_path}")
    else:
        slugs = sorted({c.workspace for c in cases})
        preflight(settings, slugs)

    print(f"Running {len(cases)} eval cases (mode: {mode})…", flush=True)
    records, errors = run_cases(settings, cases)

    summary = score_run(records)
    sweep = None
    if args.sweep and records:
        from evals.scoring import sweep_thresholds

        distances = guard_distances(settings, records)
        rows = list(zip(distances, [1 if c.expect == "refuse" else 0
                                    for c in (r["case"] for r in records)]))
        sweep = sweep_guard(rows, sweep_thresholds(distances, settings.off_topic_distance))

    print_report(summary, records, errors, sweep)
    path = save_report(settings, mode, summary, records, errors, sweep,
                       {"cases": len(cases), "errors": len(errors)})
    o = summary["overall"]
    print(f"\nRefusal P {o['precision']:.3f} · R {o['recall']:.3f} · F1 {o['f1']:.3f}"
          f"   Report → {path}")


def save_dataset(mode, cases, provenance):
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{mode}-dataset.json"
    path.write_text(json.dumps({
        "cases": [{"question": c.question, "workspace": c.workspace,
                   "expect": c.expect, "category": c.category} for c in cases],
        "provenance": provenance}, indent=2))
    return path


if __name__ == "__main__":
    main()
