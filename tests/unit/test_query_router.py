"""Tests for query.router (fakes only: no model, no database)."""

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

import query.retrieval as retrieval_module
from metadata.executor import PlanResult
from metadata.plan import Filter, FilterOp, Operation, QueryPlan
from metadata.planner import PlanningFailed, QueryPlanner
from models import KeyStatus, MetaKey, ValueType
from query.router import (
    COULD_NOT_INTERPRET_MESSAGE,
    QueryRouter,
    ResultPhraser,
    Routing,
    render_result,
)

DT = "court_decision"
KEY = MetaKey(DT, "issuing_body", ValueType.TEXT, "d", status=KeyStatus.APPROVED)
FILTER = (Filter("issuing_body", FilterOp.EQ, "X"),)


class FakePlanner(QueryPlanner):
    def __init__(self, plan=None, fail=False):
        self._plan, self._fail, self.calls = plan, fail, 0

    def plan(self, question, doc_type, keys, known_values=None):
        self.calls += 1
        if self._fail:
            raise PlanningFailed("bad", "reply")
        return self._plan


class FakeExecutor:
    def __init__(self, result):
        self._result, self.calls = result, 0

    def execute(self, plan):
        self.calls += 1
        assert self._result is not None, "this test expected no plan to run"
        return self._result


class FakeKeys:
    def __init__(self, keys):
        self._keys = keys

    def list_keys(self, doc_type, status=None):
        return self._keys

    def distinct_text_values(self, key, limit):
        return None


class EchoPhraser:
    def phrase(self, question, plan, result):
        return f"PHRASED {result.count}"


def _router(plan=None, result: PlanResult | None = None, keys=(KEY,), fail=False):
    planner, executor = FakePlanner(plan, fail), FakeExecutor(result)
    return (
        QueryRouter(planner, executor, FakeKeys(list(keys)), DT, EchoPhraser()),
        planner,
        executor,
    )


def _count_result(**kw):
    kw.setdefault("count", 7)
    return PlanResult(
        Operation.COUNT, "issuing_body = 'X'", unknown=kw.pop("unknown", 0), **kw
    )


def test_a_count_is_answered_exactly_and_never_reaches_the_retriever():
    router, _, _ = _router(QueryPlan(DT, Operation.COUNT, FILTER), _count_result())

    routing = router.route("how many?")

    assert routing == Routing(answer="PHRASED 7")


def test_an_identifier_forces_lookup_without_asking_the_planner():
    router, planner, executor = _router(QueryPlan(DT, Operation.COUNT))

    routing = router.route("What did the court decide in Pfv.20060/2022/9?")

    assert routing == Routing()
    assert planner.calls == 0 and executor.calls == 0


def test_a_lookup_without_filters_is_unrestricted():
    router, _, executor = _router(QueryPlan(DT, Operation.LOOKUP))

    assert router.route("why did the court dismiss the claim?") == Routing()
    assert executor.calls == 0


def test_a_lookup_with_filters_is_restricted_to_the_selected_documents():
    result = PlanResult(
        Operation.LOOKUP, "issuing_body = 'X'", 2, 0,
        documents=(("h1", "a.docx"), ("h2", "b.docx")),
    )  # fmt: skip
    router, _, _ = _router(QueryPlan(DT, Operation.LOOKUP, FILTER), result)

    assert router.route("what did X decide about costs?") == Routing(
        content_hashes=("h1", "h2")
    )


def test_a_restricted_lookup_says_how_many_documents_it_could_not_check():
    result = PlanResult(
        Operation.LOOKUP, "issuing_body = 'X'", 1, 4, documents=(("h1", "a.docx"),)
    )
    router, _, _ = _router(QueryPlan(DT, Operation.LOOKUP, FILTER), result)

    note = router.route("what did X decide?").note

    assert note is not None and "4 document(s) could not be checked" in note


def test_a_filter_that_matches_nothing_says_so_instead_of_reading_everything():
    result = PlanResult(Operation.LOOKUP, "issuing_body = 'X'", 0, 0)
    router, _, _ = _router(QueryPlan(DT, Operation.LOOKUP, FILTER), result)

    routing = router.route("what did X decide?")

    assert routing.content_hashes is None
    assert routing.answer is not None and "No documents match" in routing.answer


def test_too_many_matches_to_restrict_is_reported_not_truncated_silently():
    result = PlanResult(
        Operation.LOOKUP, "f", 9000, 0, documents=(("h", "a"),), truncated=True
    )
    router, _, _ = _router(QueryPlan(DT, Operation.LOOKUP, FILTER), result)

    assert "9000 documents match" in (router.route("q").answer or "")


def test_an_uninterpretable_question_is_said_plainly_not_guessed():
    router, _, executor = _router(fail=True)

    assert router.route("blah").answer == COULD_NOT_INTERPRET_MESSAGE
    assert executor.calls == 0


def test_a_router_switched_on_without_a_catalog_fails_loudly():
    router, _, _ = _router(keys=())

    with pytest.raises(RuntimeError, match="no approved keys"):
        router.route("how many?")


def test_render_result_states_filter_unknowns_and_the_part_not_applied():
    plan = QueryPlan(
        DT, Operation.COUNT, FILTER, residual="where the parties are companies"
    )

    text = render_result(plan, _count_result(unknown=3))

    assert "Executed filter: issuing_body = 'X'" in text
    assert "Matching documents: 7" in text
    assert "3 further document(s) could not be decided" in text
    assert "Not applied" in text and "companies" in text


def test_render_result_for_a_sum_and_a_list():
    total = PlanResult(
        Operation.SUM, "f", None, 0, total=Decimal("125000.50"), sum_documents=4
    )
    listed = PlanResult(
        Operation.LIST, "f", 3, 0, documents=(("h", "a.docx"),), truncated=True
    )

    assert "Total of legal_costs: 125000.50 (over 4 document(s)" in render_result(
        QueryPlan(DT, Operation.SUM, sum_key="legal_costs"), total
    )
    shown = render_result(QueryPlan(DT, Operation.LIST), listed)
    assert "a.docx" in shown and "(showing 1 of 3)" in shown


class Llm:
    def __init__(self, reply):
        self.reply = reply

    def run_tool_calling_turn(self, messages):
        return SimpleNamespace(content=self.reply)


def test_a_phrased_answer_keeps_the_numbers_and_echoes_the_filter():
    out = ResultPhraser(Llm("Összesen 1 234 határozat van.")).phrase(
        "hány?", QueryPlan(DT, Operation.COUNT), _count_result(count=1234)
    )

    assert out.startswith("Összesen 1 234 határozat van.")
    assert out.endswith("[Executed filter: issuing_body = 'X']")


def test_a_phrased_answer_that_changes_a_number_falls_back_to_the_plain_facts():
    out = ResultPhraser(Llm("Összesen 8 határozat van.")).phrase(
        "hány?", QueryPlan(DT, Operation.COUNT), _count_result(count=7)
    )

    assert out.startswith("Executed filter:") and "Matching documents: 7" in out


def test_query_knowledge_base_returns_an_exact_answer_without_retrieval(monkeypatch):
    monkeypatch.setattr(
        retrieval_module,
        "settings",
        replace(retrieval_module.settings, QUERY_ROUTER=True),
    )
    monkeypatch.setattr(
        retrieval_module,
        "get_query_router",
        lambda: SimpleNamespace(route=lambda q: Routing(answer="EXACT")),
    )
    monkeypatch.setattr(
        retrieval_module,
        "_answer_from_documents",
        lambda *a: pytest.fail("retrieval must not run"),
    )

    assert retrieval_module.query_knowledge_base("how many?") == "EXACT"


def test_query_knowledge_base_restricts_retrieval_and_appends_the_note(monkeypatch):
    seen = {}

    class Store:
        def restricted_to(self, hashes):
            seen["hashes"] = tuple(hashes)
            return "scoped"

    monkeypatch.setattr(
        retrieval_module,
        "settings",
        replace(retrieval_module.settings, QUERY_ROUTER=True),
    )
    monkeypatch.setattr(
        retrieval_module,
        "get_query_router",
        lambda: SimpleNamespace(
            route=lambda q: Routing(content_hashes=("h1",), note="NOTE")
        ),
    )
    monkeypatch.setattr(
        retrieval_module,
        "_answer_from_documents",
        lambda q, k, m, s, f, store: f"ANSWER from {store}",
    )

    out = retrieval_module.query_knowledge_base("q", store=Store())  # type: ignore[arg-type]

    assert seen["hashes"] == ("h1",)
    assert out == "ANSWER from scoped\n\nNOTE"
