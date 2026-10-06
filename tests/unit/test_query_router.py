"""Tests for query.router (fakes only: no model, no database)."""

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

import query.retrieval as retrieval_module
from metadata.executor import PlanResult
from metadata.plan import Filter, FilterOp, Operation, QueryPlan
from metadata.planner import PlanningFailed, QueryPlanner
from models import (
    DocumentSelection,
    DocumentType,
    KeyStatus,
    MetaKey,
    TypeStatus,
    ValueType,
)
from query.router import (
    COULD_NOT_INTERPRET_MESSAGE,
    NOT_SUPPORTED_MESSAGE,
    QueryRouter,
    ResultPhraser,
    Routing,
    render_result,
)

DT = "court_decision"
KEY = MetaKey(DT, "issuing_body", ValueType.TEXT, "d", status=KeyStatus.APPROVED)
FILTER = (Filter("issuing_body", FilterOp.EQ, "X"),)
SELECTION = DocumentSelection(
    "SELECT d.id FROM documents d WHERE d.source_file = %s", ("x",)
)


class FakePlanner(QueryPlanner):
    def __init__(self, plan=None, fail=False):
        self._plan, self._fail, self.calls = plan, fail, 0

    def plan(self, question, catalogs, known_values=None):
        self.calls += 1
        self.catalogs = catalogs
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


class FakeCatalog:
    """The document types and keys the router reads (and the stored values)."""

    def __init__(self, keys, types=None):
        self._keys = keys
        self._types = (
            [DocumentType(DT, "Court decision", "A ruling.", TypeStatus.APPROVED)]
            if types is None
            else types
        )

    def list_types(self, status=None):
        return [t for t in self._types if status is None or t.status is status]

    def list_keys(self, doc_type, status=None):
        return [k for k in self._keys if k.doc_type == doc_type]

    def distinct_text_values(self, key, limit, doc_type=None):
        return None


class EchoPhraser:
    def phrase(self, question, plan, result):
        return f"PHRASED {result.count}"


def _router(plan=None, result: PlanResult | None = None, types=None, fail=False):
    planner, executor = FakePlanner(plan, fail), FakeExecutor(result)
    return (
        QueryRouter(planner, executor, FakeCatalog([KEY], types), EchoPhraser()),
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


def test_an_identifier_is_a_parameter_not_an_intent_so_the_planner_still_decides():
    """It used to skip the planner: 'five cases similar to X' then went to a retrieval
    that returned X alone, and the answer was 'I could not find this information'."""
    router, planner, _ = _router(QueryPlan(None, Operation.LOOKUP))

    routing = router.route("What did the court decide in Pfv.20060/2022/9?")

    assert planner.calls == 1
    assert routing == Routing()  # a lookup: the retrieval pins the named document


def test_a_request_for_similar_cases_is_said_plainly_not_answered_by_a_search():
    plan = QueryPlan(
        None,
        Operation.UNSUPPORTED,
        reason="five cases similar to the 27.P.20.339/2021/37 case",
    )
    router, _, executor = _router(plan)

    routing = router.route("Sorolj fel 5 hasonló ügyet, mint a 27.P.20.339/2021/37.")

    assert routing.answer == (
        f"{NOT_SUPPORTED_MESSAGE} (five cases similar to the 27.P.20.339/2021/37 case)"
    )
    assert routing.selection is None
    assert executor.calls == 0  # nothing was run, nothing was searched


def test_a_lookup_that_names_no_type_and_no_filter_is_unrestricted():
    router, _, executor = _router(QueryPlan(None, Operation.LOOKUP))

    assert router.route("why did the court dismiss the claim?") == Routing()
    assert executor.calls == 0


def test_a_lookup_that_names_only_a_type_is_restricted_to_that_types_documents():
    result = PlanResult(
        Operation.LOOKUP, "court_decision documents", 2235, 0, selection=SELECTION
    )
    router, _, executor = _router(QueryPlan(DT, Operation.LOOKUP), result)

    routing = router.route("what did the court decide about costs?")

    assert routing == Routing(selection=SELECTION) and executor.calls == 1


def test_the_planner_is_offered_the_approved_types_with_their_keys():
    router, planner, _ = _router(QueryPlan(None, Operation.LOOKUP))

    router.route("a content question")

    [catalog] = planner.catalogs
    assert catalog.doc_type.type == DT and [k.key for k in catalog.keys] == [
        "issuing_body"
    ]


def test_a_lookup_with_filters_is_restricted_to_the_selected_documents():
    result = PlanResult(
        Operation.LOOKUP, "issuing_body = 'X'", 2, 0, selection=SELECTION
    )
    router, _, _ = _router(QueryPlan(DT, Operation.LOOKUP, FILTER), result)

    assert router.route("what did X decide about costs?") == Routing(
        selection=SELECTION
    )


def test_a_restricted_lookup_says_how_many_documents_it_could_not_check():
    result = PlanResult(
        Operation.LOOKUP, "issuing_body = 'X'", 1, 4, selection=SELECTION
    )
    router, _, _ = _router(QueryPlan(DT, Operation.LOOKUP, FILTER), result)

    note = router.route("what did X decide?").note

    assert note is not None and "4 document(s) could not be checked" in note


def test_a_filter_that_matches_nothing_says_so_instead_of_reading_everything():
    result = PlanResult(Operation.LOOKUP, "issuing_body = 'X'", 0, 0)
    router, _, _ = _router(QueryPlan(DT, Operation.LOOKUP, FILTER), result)

    routing = router.route("what did X decide?")

    assert routing.selection is None
    assert routing.answer is not None and "No documents match" in routing.answer


def test_a_huge_match_is_restricted_to_not_refused_because_no_list_travels():
    """The restriction is a sub-select the database evaluates: there is no cap."""
    result = PlanResult(Operation.LOOKUP, "f", 800_000, 0, selection=SELECTION)
    router, _, _ = _router(QueryPlan(DT, Operation.LOOKUP, FILTER), result)

    assert router.route("q") == Routing(selection=SELECTION)


def test_a_count_with_a_residual_is_not_answered_exactly_it_is_read_like_a_lookup():
    """A count over the keys alone would answer an easier question than the one asked."""
    plan = QueryPlan(
        DT,
        Operation.COUNT,
        FILTER,
        residual="where the claim was dismissed on limitation",
    )
    lookup = PlanResult(
        Operation.LOOKUP, "issuing_body = 'X'", 1, 0, selection=SELECTION
    )
    router, _, executor = _router(plan, lookup)

    routing = router.route("how many X decisions dismissed the claim on limitation?")

    assert routing == Routing(selection=SELECTION)
    assert routing.answer is None and executor.calls == 1


def test_a_listing_with_a_residual_and_no_filters_reads_its_types_documents_not_a_list():
    plan = QueryPlan(DT, Operation.LIST, residual="that discuss limitation")
    result = PlanResult(
        Operation.LOOKUP, "court_decision documents", 2235, 0, selection=SELECTION
    )
    router, _, executor = _router(plan, result)

    routing = router.route("which decisions discuss limitation?")

    assert routing == Routing(selection=SELECTION)  # read, restricted to the type
    assert routing.answer is None and executor.calls == 1


def test_an_uninterpretable_question_is_said_plainly_not_guessed():
    router, _, executor = _router(fail=True)

    assert router.route("blah").answer == COULD_NOT_INTERPRET_MESSAGE
    assert executor.calls == 0


def test_a_router_switched_on_without_an_approved_document_type_fails_loudly():
    router, _, _ = _router(types=[])

    with pytest.raises(RuntimeError, match="no approved document types"):
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
        def restricted_to(self, selection):
            seen["selection"] = selection
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
            route=lambda q: Routing(selection=SELECTION, note="NOTE")
        ),
    )
    monkeypatch.setattr(
        retrieval_module,
        "_answer_from_documents",
        lambda q, k, m, s, f, store: f"ANSWER from {store}",
    )

    out = retrieval_module.query_knowledge_base("q", store=Store())  # type: ignore[arg-type]

    assert seen["selection"] == SELECTION
    assert out == "ANSWER from scoped\n\nNOTE"


def test_as_routed_turns_an_exact_plan_with_a_residual_into_a_lookup_and_leaves_the_rest():
    from query.router import as_routed

    exact_with_residual = QueryPlan(
        DT, Operation.COUNT, FILTER, group_by="document_kind", residual="limitation"
    )
    exact = QueryPlan(DT, Operation.COUNT, FILTER)
    lookup = QueryPlan(DT, Operation.LOOKUP, FILTER, residual="limitation")
    unsupported = QueryPlan(None, Operation.UNSUPPORTED, reason="similar", residual="x")

    routed = as_routed(exact_with_residual)

    assert routed.operation is Operation.LOOKUP and routed.group_by is None
    assert routed.filters == FILTER and routed.doc_type == DT  # the narrowing stays
    assert as_routed(exact) is exact
    assert as_routed(lookup) is lookup
    assert as_routed(unsupported) is unsupported  # never turned into a search


def test_a_lookup_that_names_an_identifier_is_not_restricted_by_the_plans_filters():
    """Identifiers are stored in written variants and an anonymised document may lack a
    court name: a metadata restriction could only exclude the document that was named.
    (The single-document golden questions fell to 50-75% when it did.)"""
    result = PlanResult(
        Operation.LOOKUP, "court_decision documents", 0, 0
    )  # would refuse
    plan = QueryPlan(
        DT,
        Operation.LOOKUP,
        (Filter("issuing_body", FilterOp.EQ, "X"),),
    )
    router, _, executor = _router(plan, result)

    routing = router.route(
        "Ki képviseli az alperest a 104.K.700.027/2023/3. számú ügyben?"
    )

    assert routing == Routing()  # unrestricted: the retrieval pins the named case
    assert executor.calls == 0
