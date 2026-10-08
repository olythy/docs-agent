"""The decision: which way a question goes, with which profile, over which documents.

Only the **decision** is checked here: which branch, which profile, which scope. Running the
SQL of an exact answer and wording an answer or a refusal are other classes with other
tests, so nothing below executes a count or compares a sentence.
"""

import pytest
from test_scope import (  # type: ignore[import-not-found]
    DOCS,
    FakePlans,
    FakeSource,
    lookup,
)

from metadata.identifier_resolver import IdentifierResolver
from metadata.plan import Filter, FilterOp, Operation, QueryPlan
from metadata.planner import PlanningFailed, QueryPlanner
from models import DocumentType, KeyStatus, MetaKey, TypeStatus, ValueType
from query.decision import (
    AnswerExactly,
    PlanningDecider,
    ProfileSelector,
    ReadDocuments,
    Refuse,
    Scope,
    ScopeResolver,
    UnplannedDecider,
)
from query.facts import QueryFacts
from query.outcome import DeclineReason

DT = "court_decision"
KEY = MetaKey(DT, "issuing_body", ValueType.TEXT, "d", status=KeyStatus.APPROVED)
FILTER = (Filter("issuing_body", FilterOp.EQ, "Debrecen"),)


class FakePlanner(QueryPlanner):
    def __init__(self, plan=None, fail=None):
        self._plan, self._fail = plan, fail
        self.questions: list[str] = []

    def plan(self, question, catalogs, known_values=None):
        self.questions.append(question)
        if self._fail:
            raise PlanningFailed(self._fail, "the model's reply")
        return self._plan


class FakeCatalog:
    def __init__(self, approved=True):
        self._types = (
            [DocumentType(DT, "Court decision", "A ruling.", TypeStatus.APPROVED)]
            if approved
            else []
        )

    def list_types(self, status=None):
        return list(self._types)

    def list_keys(self, doc_type, status=None):
        return [KEY]

    def distinct_text_values(self, key, limit, doc_type=None):
        return None


def facts(question="a question", identifiers=()):
    return QueryFacts(question, identifiers=tuple(identifiers))


def decider(plan=None, fail=None, plans=None, profile="hybrid", approved=True):
    planner = FakePlanner(plan, fail)
    plans = plans or FakePlans()
    scopes = ScopeResolver(plans, IdentifierResolver(FakeSource(DOCS)))
    made = PlanningDecider(
        planner, FakeCatalog(approved), scopes, ProfileSelector(profile)
    )
    return made, planner, plans


class TestExactAnswers:
    @pytest.mark.parametrize(
        "operation",
        [Operation.COUNT, Operation.LIST, Operation.OVERVIEW, Operation.SUM],
    )
    def test_a_question_the_plan_covers_is_answered_exactly(self, operation):
        plan = QueryPlan(DT, operation, FILTER)
        made, _, plans = decider(plan)

        decision = made.decide(facts())

        assert isinstance(decision, AnswerExactly)
        assert decision.plan == plan
        assert plans.executed == []  # deciding is not executing

    def test_a_plan_with_a_residual_is_read_not_counted(self):
        """Part of the question is covered by no key, so a count would answer an easier one."""
        plan = QueryPlan(
            DT, Operation.COUNT, FILTER, residual="and the claim was rejected"
        )
        made, _, _ = decider(plan)

        decision = made.decide(facts())

        assert isinstance(decision, ReadDocuments)
        assert decision.plan is not None
        assert decision.plan.operation is Operation.LOOKUP  # the plan as it is treated
        assert decision.plan.filters == FILTER  # the filters still narrow


class TestRefusals:
    def test_a_request_the_system_cannot_do_yet_is_refused_with_the_planners_reason(
        self,
    ):
        plan = QueryPlan(
            None, Operation.UNSUPPORTED, reason="five similar cases were asked"
        )
        made, _, plans = decider(plan)

        decision = made.decide(facts())

        assert isinstance(decision, Refuse)
        assert decision.declined.reason is DeclineReason.NOT_SUPPORTED
        assert decision.declined.stage == "planning"
        assert decision.declined.detail == "five similar cases were asked"
        assert plans.executed == []

    def test_a_question_the_planner_cannot_interpret_is_refused_with_why(self):
        made, _, _ = decider(fail="unknown key 'colour'")

        decision = made.decide(facts())

        assert isinstance(decision, Refuse)
        assert decision.declined.reason is DeclineReason.COULD_NOT_INTERPRET
        assert decision.declined.stage == "planning"
        assert decision.declined.detail == "unknown key 'colour'"

    def test_filters_that_select_no_document_are_a_refusal_at_the_scope(self):
        made, _, _ = decider(lookup(), plans=FakePlans(count=0))

        decision = made.decide(facts())

        assert isinstance(decision, Refuse)
        assert decision.declined.reason is DeclineReason.NO_MATCHING_DOCUMENTS
        assert decision.declined.stage == "scope"

    def test_no_approved_type_is_an_error_not_a_refusal(self):
        """The router was switched on before the catalog was loaded: a setup mistake."""
        made, _, _ = decider(lookup(), approved=False)

        with pytest.raises(RuntimeError, match="no approved document types"):
            made.decide(facts())


class TestReading:
    def test_a_content_question_names_no_type_and_reads_everything(self):
        made, _, plans = decider(lookup(restricted=False))

        decision = made.decide(facts())

        assert isinstance(decision, ReadDocuments)
        assert decision.scope == Scope()
        assert plans.executed == []

    def test_filters_narrow_the_scope(self):
        made, _, _ = decider(lookup(), plans=FakePlans(count=5))

        decision = made.decide(facts())

        assert isinstance(decision, ReadDocuments)
        assert decision.scope.selection is not None
        assert decision.scope.selection.params == ("court",)

    def test_a_named_identifier_makes_that_document_the_scope(self):
        made, _, plans = decider(lookup(restricted=False))

        decision = made.decide(facts("case?", identifiers=["4.P.20.409/2023/4"]))

        assert isinstance(decision, ReadDocuments)
        assert decision.scope.selection is not None
        assert decision.scope.selection.params == ([1],)
        assert plans.executed == []  # no filter to run over the corpus

    def test_an_identifier_wins_over_a_filter_that_disagrees_and_the_note_says_so(self):
        made, _, _ = decider(lookup(), plans=FakePlans(matching=set()))

        decision = made.decide(facts("case?", identifiers=["4.P.20.409/2023/4"]))

        assert isinstance(decision, ReadDocuments)
        assert decision.scope.selection is not None
        assert decision.scope.selection.params == ([1],)
        assert decision.scope.note and "was not applied" in decision.scope.note

    def test_an_unknown_identifier_neither_narrows_nor_refuses(self):
        made, _, _ = decider(lookup(restricted=False))

        decision = made.decide(facts("case?", identifiers=["99.P.99.999/2099/1"]))

        assert isinstance(decision, ReadDocuments)
        assert decision.scope.selection is None
        assert decision.scope.identifiers.unresolved == ("99.P.99.999/2099/1",)

    def test_an_unknown_identifier_is_read_unrestricted_even_with_filters_that_would_find_nothing(
        self,
    ):
        """The step-1 regression: a filter must not turn a readable question into a refusal."""
        made, _, plans = decider(lookup(), plans=FakePlans(count=0))

        decision = made.decide(facts("case?", identifiers=["99.P.99.999/2099/1"]))

        assert isinstance(decision, ReadDocuments)
        assert decision.scope.selection is None
        assert plans.executed == []


class TestProfile:
    @pytest.mark.parametrize("name", ["hybrid", "vector"])
    def test_the_profile_is_the_one_the_selector_chooses(self, name):
        made, _, _ = decider(lookup(restricted=False), profile=name)

        decision = made.decide(facts())

        assert isinstance(decision, ReadDocuments) and decision.profile == name

    def test_the_selector_returns_its_default_whatever_the_question(self):
        selector = ProfileSelector("hybrid")

        assert selector.select(facts("anything"), None) == "hybrid"
        assert selector.select(facts("x", ["A/1"]), lookup()) == "hybrid"


def test_every_decision_carries_the_facts_it_was_made_from():
    f = facts("q", identifiers=["4.P.20.409/2023/4"])
    for plan in (
        QueryPlan(DT, Operation.COUNT, FILTER),
        QueryPlan(None, Operation.UNSUPPORTED, reason="r"),
        lookup(restricted=False),
    ):
        made, _, _ = decider(plan)
        assert made.decide(f).facts is f


def test_the_planner_is_asked_the_question_once():
    made, planner, _ = decider(lookup(restricted=False))

    made.decide(facts("What did the court decide?"))

    assert planner.questions == ["What did the court decide?"]


class TestUnplannedDecider:
    def test_it_reads_everything_unrestricted_without_asking_a_planner(self):
        decision = UnplannedDecider(ProfileSelector("vector")).decide(facts("anything"))

        assert isinstance(decision, ReadDocuments)
        assert decision.plan is None
        assert decision.scope == Scope()
        assert decision.profile == "vector"
