"""The new decision against the original router, on the same cases.

The decider is the router's successor, so on everything except the one thing that was
changed on purpose it must decide as the router does. The cases are the router's: an exact
count, a plan with a residual, a request that is not supported yet, a plan that cannot be
made, content questions with and without a filter, a filter that selects nothing. Each is
run through both and the kind of outcome compared (exact answer / refusal / read, and over
which documents).

The intended differences are listed as such, each in its own test: a lookup that names an
identifier which resolves. The router leaves it unrestricted and ignores the filters (the
retrieval pins the document by text); the decider restricts it to the documents the identifier
resolves to, and says so when the filters disagree. When the named identifier resolves to
nothing the decider does what the router does (unrestricted, filters not applied), and adds a note.
"""

import pytest
from test_decider import (  # type: ignore[import-not-found]
    FILTER,
    FakeCatalog,
    FakePlanner,
)
from test_scope import (  # type: ignore[import-not-found]
    DOCS,
    SELECTED,
    FakePlans,
    FakeSource,
    lookup,
)

from metadata.identifier_resolver import IdentifierResolver
from metadata.plan import Operation, QueryPlan
from query.decision import (
    AnswerExactly,
    PlanningDecider,
    ProfileSelector,
    ReadDocuments,
    Refuse,
    ScopeResolver,
)
from query.facts import QueryFactsReader
from query.outcome import (
    COULD_NOT_INTERPRET_MESSAGE,
    NOT_SUPPORTED_MESSAGE,
    RefusalRenderer,
)
from query.router import QueryRouter

DT = "court_decision"
CASE = "4.P.20.409/2023/4"


class EchoPhraser:
    def phrase(self, question, plan, result):
        return "PHRASED"


def run_both(question, plan=None, fail=None, **plans_kwargs):
    """The router's routing and the decider's decision for one question."""
    router = QueryRouter(
        FakePlanner(plan, fail), FakePlans(**plans_kwargs), FakeCatalog(), EchoPhraser()
    )
    decider = PlanningDecider(
        FakePlanner(plan, fail),
        FakeCatalog(),
        ScopeResolver(FakePlans(**plans_kwargs), IdentifierResolver(FakeSource(DOCS))),
        ProfileSelector("hybrid"),
    )
    return router.route(question), decider.decide(QueryFactsReader().read(question))


def kind_of_routing(routing):
    """What the router did, in the decision's terms."""
    if routing.answer is None:
        return ("read", routing.selection)
    refusal = (
        routing.answer == COULD_NOT_INTERPRET_MESSAGE
        or routing.answer.startswith(NOT_SUPPORTED_MESSAGE)
        or routing.answer.startswith("No documents match")
    )
    return ("refuse", None) if refusal else ("exact", None)


def kind_of_decision(decision):
    if isinstance(decision, AnswerExactly):
        return ("exact", None)
    if isinstance(decision, Refuse):
        return ("refuse", None)
    assert isinstance(decision, ReadDocuments)
    return ("read", decision.scope.selection)


QUESTION = "What did the court decide?"

SAME = [
    (
        "an exact count",
        {"plan": QueryPlan(DT, Operation.COUNT, FILTER)},
        ("exact", None),
    ),
    (
        "a count with a residual is read, narrowed by its filters",
        {
            "plan": QueryPlan(
                DT, Operation.COUNT, FILTER, residual="and it was rejected"
            )
        },
        ("read", SELECTED),
    ),
    (
        "not supported yet",
        {"plan": QueryPlan(None, Operation.UNSUPPORTED, reason="similar cases")},
        ("refuse", None),
    ),
    ("a plan that cannot be made", {"fail": "bad key"}, ("refuse", None)),
    (
        "a content question with no filter",
        {"plan": lookup(restricted=False)},
        ("read", None),
    ),
    (
        "a content question with a filter",
        {"plan": lookup(), "count": 5},
        ("read", SELECTED),
    ),
    ("a filter that selects nothing", {"plan": lookup(), "count": 0}, ("refuse", None)),
]


@pytest.mark.parametrize(("name", "case", "expected"), SAME, ids=[c[0] for c in SAME])
def test_the_decider_decides_as_the_router_does(name, case, expected):
    routing, decision = run_both(QUESTION, **case)

    assert kind_of_routing(routing) == expected
    assert kind_of_decision(decision) == expected


class TestWhatWasChangedOnPurpose:
    def test_a_lookup_that_names_an_identifier_is_restricted_to_its_document(self):
        question = f"What did the court decide in case {CASE}?"

        routing, decision = run_both(question, plan=lookup(restricted=False))

        assert kind_of_routing(routing) == (
            "read",
            None,
        )  # unrestricted, the pin finds it
        assert isinstance(decision, ReadDocuments)
        assert decision.scope.selection is not None
        assert decision.scope.selection.params == ([1],)  # that document only

    def test_the_filters_no_longer_replace_the_named_document_nor_are_silently_dropped(
        self,
    ):
        question = f"The Debrecen court's decision in case {CASE}?"

        routing, decision = run_both(question, plan=lookup(), count=5, matching=set())

        assert kind_of_routing(routing) == (
            "read",
            None,
        )  # the router drops the filters, unsaid
        assert isinstance(decision, ReadDocuments)
        assert decision.scope.selection is not None
        assert decision.scope.selection.params == ([1],)
        assert (
            decision.scope.note and "was not applied" in decision.scope.note
        )  # now said

    def test_an_identifier_no_document_carries_reads_unrestricted_like_the_router(self):
        question = "What did the court decide in case 99.P.99.999/2099/1?"

        routing, decision = run_both(question, plan=lookup(restricted=False))

        assert kind_of_routing(routing) == ("read", None)
        assert kind_of_decision(decision) == ("read", None)

    def test_and_so_does_a_filter_that_would_select_nothing(self):
        """The 2026-10-06 regression: the router ignores the filters when an identifier is
        named; the decider must too, or 'no documents match' comes back for a case that exists."""
        question = "What did the Debrecen court decide in case 99.P.99.999/2099/1?"

        routing, decision = run_both(question, plan=lookup(), count=0)

        assert kind_of_routing(routing) == ("read", None)
        assert kind_of_decision(decision) == ("read", None)
        assert isinstance(decision, ReadDocuments)
        assert (
            decision.scope.note and "was not applied" in decision.scope.note
        )  # now said


class TestTheWordsOfARefusalAreTheRouters:
    """The decider only decides; worded by the renderer it says what the router said."""

    @pytest.mark.parametrize(
        "case",
        [
            {"fail": "bad key"},
            {
                "plan": QueryPlan(
                    None, Operation.UNSUPPORTED, reason="five similar cases"
                )
            },
            {"plan": lookup(), "count": 0, "unknown": 2},
        ],
        ids=["could not interpret", "not supported", "no matching documents"],
    )
    def test_the_same_sentence(self, case):
        routing, decision = run_both(QUESTION, **case)

        assert isinstance(decision, Refuse)
        assert RefusalRenderer().render(decision.declined) == routing.answer
