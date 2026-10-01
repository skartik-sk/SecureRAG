from langchain_core.documents import Document

from evals.auto import parse_questions, pick_chunks
from evals.dataset import SEED_SLUGS, Case, fixed_cases
from evals.scoring import (Outcome, abstained, classify, refusal_metrics,
                           score_run, sweep_guard, sweep_thresholds)


def state(refused=None, answer="", sources=None):
    return {"refused": refused, "answer": answer, "sources": sources or []}


class TestClassify:
    def test_guard_refusal_wins_over_any_answer(self):
        assert classify(state(refused="I can only answer ...", answer="x")) is Outcome.REFUSED

    def test_pipeline_refusal(self):
        assert classify(state(refused="I couldn't find anything ...")) is Outcome.REFUSED

    def test_exact_i_dont_know_variants(self):
        for a in ["I don't know.", "I don't know", "i don't know", "I DON'T KNOW.",
                  "I don’t know.", " I dont know. "]:
            assert classify(state(answer=a)) is Outcome.DONT_KNOW, a

    def test_idk_with_extra_text_is_an_answer_not_dont_know(self):
        assert classify(state(answer="I don't know the fee, but returns take 7 days.",
                              sources=[{"file": "f.md", "section": "s"}])) \
            is Outcome.ANSWERED_UNCITED
        assert classify(state(answer="I don't know the fee, but returns take 7 days [1].",
                              sources=[{"file": "f.md", "section": "s"}])) \
            is Outcome.ANSWERED_CITED

    def test_cited_answer_with_sources(self):
        assert classify(state(answer="It is $4.99 [1].",
                              sources=[{"file": "f.md", "section": "s"}])) is Outcome.ANSWERED_CITED

    def test_citations_without_sources_is_uncited(self):
        assert classify(state(answer="It is $4.99 [1].")) is Outcome.ANSWERED_UNCITED

    def test_plain_answer_without_citations(self):
        assert classify(state(answer="It is $4.99.", sources=[{"file": "f.md"}])) \
            is Outcome.ANSWERED_UNCITED

    def test_empty_answer(self):
        assert classify(state(answer="   ")) is Outcome.EMPTY


class TestAbstained:
    def test_refused_dont_know_and_empty_abstain(self):
        assert abstained(Outcome.REFUSED)
        assert abstained(Outcome.DONT_KNOW)
        assert abstained(Outcome.EMPTY)

    def test_answers_do_not_abstain(self):
        assert not abstained(Outcome.ANSWERED_CITED)
        assert not abstained(Outcome.ANSWERED_UNCITED)


class TestRefusalMetrics:
    def test_perfect(self):
        m = refusal_metrics(tp=5, fp=0, fn=0)
        assert m == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_mixed(self):
        m = refusal_metrics(tp=18, fp=1, fn=2)
        assert abs(m["precision"] - 18 / 19) < 1e-9
        assert abs(m["recall"] - 18 / 20) < 1e-9
        assert abs(m["f1"] - 2 * 18 / (2 * 18 + 1 + 2)) < 1e-9

    def test_zero_denominators_are_zero_not_crash(self):
        assert refusal_metrics(tp=0, fp=0, fn=0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def case(ws, expect, cat="answerable", q="q"):
    return Case(question=q, workspace=ws, expect=expect, category=cat)


def rec(ws, expect, outcome, cat="answerable", q="q"):
    return {"case": case(ws, expect, cat, q), "outcome": outcome}


class TestScoreRun:
    def test_confusion_counts_and_metrics(self):
        records = [
            rec("ws", "refuse", Outcome.REFUSED),                       # TP
            rec("ws", "refuse", Outcome.ANSWERED_CITED),                # FN (leak)
            rec("ws", "answer", Outcome.REFUSED),                       # FP (over-refusal)
            rec("ws", "answer", Outcome.ANSWERED_CITED),                # TN
        ]
        r = score_run(records)
        assert r["overall"]["tp"] == 1
        assert r["overall"]["fp"] == 1
        assert r["overall"]["fn"] == 1
        assert r["overall"]["tn"] == 1
        assert r["overall"]["precision"] == 0.5
        assert r["overall"]["recall"] == 0.5

    def test_failure_buckets_are_populated(self):
        records = [
            rec("ws", "refuse", Outcome.ANSWERED_UNCITED, cat="off_topic"),
            rec("ws", "answer", Outcome.DONT_KNOW),
            rec("ws", "answer", Outcome.ANSWERED_UNCITED),
        ]
        r = score_run(records)
        assert [x["case"].question for x in r["leaks"]] == ["q"]
        assert [x["case"].question for x in r["over_refusals"]] == ["q"]
        assert [x["case"].question for x in r["uncited_answers"]] == ["q"]

    def test_slices_by_workspace_and_category(self):
        records = [
            rec("a", "refuse", Outcome.REFUSED, cat="off_topic"),
            rec("b", "answer", Outcome.ANSWERED_CITED),
        ]
        r = score_run(records)
        assert set(r["by_workspace"]) == {"a", "b"}
        assert set(r["by_category"]) == {"off_topic", "answerable"}
        assert r["by_workspace"]["a"]["tp"] == 1
        assert r["by_category"]["answerable"]["tn"] == 1

    def test_empty_run_is_zeroed_not_crash(self):
        r = score_run([])
        assert r["overall"]["n"] == 0
        assert r["overall"]["f1"] == 0.0


class TestSweepThresholds:
    def test_deciles_plus_current(self):
        ts = sweep_thresholds([0.10 * i for i in range(1, 11)], 0.95)
        assert ts == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

    def test_dedupes_rounded_values(self):
        ts = sweep_thresholds([0.30, 0.30, 0.30, 0.30], 0.30)
        assert ts == [0.3]

    def test_ignores_infinite_distances(self):
        ts = sweep_thresholds([0.4, float("inf")], 0.95)
        assert ts == [0.4, 0.95]


class TestSweepGuard:
    def test_threshold_separates(self):
        rows = [(0.5, 1), (0.5, 1), (0.3, 0), (0.3, 0)]
        out = sweep_guard(rows, thresholds=[0.4])
        assert out[0]["threshold"] == 0.4
        assert out[0]["tp"] == 2
        assert out[0]["fp"] == 0
        assert out[0]["fn"] == 0

    def test_low_threshold_over_refuses(self):
        rows = [(0.5, 1), (0.3, 0)]
        out = sweep_guard(rows, thresholds=[0.1, 0.9])
        assert out[0]["fp"] == 1          # 0.1 refuses everything
        assert out[1]["fn"] == 1          # 0.9 refuses nothing


class TestParseQuestions:
    def test_numbered_lines(self):
        assert parse_questions("1. What is the fee?\n2. When is pickup?") == \
            ["What is the fee?", "When is pickup?"]

    def test_json_array(self):
        assert parse_questions('["What is X?", "What is Y?"]') == ["What is X?", "What is Y?"]

    def test_bullets(self):
        assert parse_questions("- What is X?\n- What is Y?") == ["What is X?", "What is Y?"]

    def test_garbage_yields_no_questions(self):
        assert parse_questions("no questions here, sorry") == []

    def test_drops_empty_and_short_fragments(self):
        assert parse_questions("1. \n2. ok") == ["ok"]


class TestPickChunks:
    def doc(content, file, section):
        return Document(page_content=content, metadata={"source_file": file, "section": section})

    def test_dedupes_same_file_and_section(self):
        docs = [TestPickChunks.doc("a", "f.md", "S1"), TestPickChunks.doc("b", "f.md", "S1"),
                TestPickChunks.doc("c", "f.md", "S2")]
        picked = pick_chunks(docs, want=3)
        assert len(picked) == 2

    def test_caps_at_want(self):
        docs = [TestPickChunks.doc(str(i), f"f{i}.md", "S") for i in range(10)]
        assert len(pick_chunks(docs, want=4)) == 4

    def test_spreads_across_files_not_first_file_only(self):
        docs = [TestPickChunks.doc(str(i), "a.md", f"S{i}") for i in range(6)]
        docs += [TestPickChunks.doc("z", "b.md", "SB")]
        picked = pick_chunks(docs, want=4)
        assert {d.metadata["source_file"] for d in picked} == {"a.md", "b.md"}


class TestLoadCases:
    def test_reads_saved_dataset_json(self, tmp_path):
        import argparse
        import json

        from evals.run import load_cases

        p = tmp_path / "ds.json"
        p.write_text(json.dumps({"cases": [
            {"question": "q", "workspace": "delivery-policy",
             "expect": "answer", "category": "answerable"}]}))
        args = argparse.Namespace(dataset=str(p), auto=False, workspace=None)
        cases = load_cases(args)
        assert len(cases) == 1
        assert cases[0].question == "q"
        assert cases[0].expect == "answer"


class TestFixedDataset:
    def test_valid_slugs_expects_categories(self):
        for c in fixed_cases():
            assert c.workspace in SEED_SLUGS, c
            assert c.expect in ("answer", "refuse"), c
            assert c.category in ("answerable", "near_miss", "off_topic", "cross_workspace"), c

    def test_expect_matches_category(self):
        for c in fixed_cases():
            if c.category == "answerable":
                assert c.expect == "answer", c
            else:
                assert c.expect == "refuse", c

    def test_unique_question_per_workspace(self):
        seen = set()
        for c in fixed_cases():
            key = (c.workspace, c.question.lower())
            assert key not in seen, c
            seen.add(key)

    def test_minimum_coverage(self):
        cases = fixed_cases()
        assert len(cases) >= 40
        for slug in SEED_SLUGS:
            ws_cases = [c for c in cases if c.workspace == slug]
            assert sum(c.expect == "answer" for c in ws_cases) >= 5, slug
            assert sum(c.expect == "refuse" for c in ws_cases) >= 4, slug

    def test_cross_workspace_cases_exist_for_each_seed(self):
        cross = {c.workspace for c in fixed_cases() if c.category == "cross_workspace"}
        assert cross == set(SEED_SLUGS)
