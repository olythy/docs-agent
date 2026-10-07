"""Which documents the retrieval may look at: the rules for combining an identifier and the filters."""

from collections.abc import Iterable, Sequence

import pytest

from metadata.executor import Membership, PlanResult
from metadata.identifier_resolver import IdentifierResolver
from metadata.identifiers import identifier_matches
from metadata.plan import Filter, FilterOp, Operation, QueryPlan
from models import DocumentSelection
from query.decision import Scope, ScopeResolver
from query.outcome import Declined, DeclineReason

SELECTED = DocumentSelection("SELECT d.id FROM documents d WHERE x", ("court",))

DOCS = {
    1: ["4.P.20.409/2023/4"],
    2: ["27.P.20.339/2021/37", "27.P.20.339/2021/37-ítélet"],
    3: ["8.P.21.329/2024/12-III", "8.P.XI.21.329/2024/12."],
    4: ["HU001"],
    5: ["HU001-A"],  # a second document with a number that continues it
}


class FakeSource:
    """The identifier lookup, applying the real rule in Python to a small table."""

    def __init__(self, documents):
        self.documents = documents

    def documents_with_identifiers(self, wanted: Sequence[str]):
        return [
            (position, doc_id)
            for position, w in enumerate(wanted, start=1)
            for doc_id, values in sorted(self.documents.items())
            if any(identifier_matches(v, w) for v in values)
        ]


class FakePlans:
    """A planner-filter runner: a fixed count and a fixed set of documents that match."""

    def __init__(self, count=5, unknown=0, matching: Iterable[int] = ()):
        self.count, self.unknown, self.matching = count, unknown, frozenset(matching)
        self.executed: list[QueryPlan] = []
        self.asked: list[tuple[int, ...]] = []

    def execute(self, plan):
        self.executed.append(plan)
        return PlanResult(
            Operation.LOOKUP,
            "court_decision documents: issuing_body is 'Debrecen'",
            count=self.count,
            unknown=self.unknown,
            selection=SELECTED,
        )

    def members(self, plan, document_ids):
        self.asked.append(tuple(document_ids))
        return Membership(
            "court_decision documents: issuing_body is 'Debrecen'",
            self.matching & frozenset(document_ids),
        )


def lookup(*, restricted=True):
    filters = (Filter("issuing_body", FilterOp.EQ, "Debrecen"),) if restricted else ()
    return QueryPlan(
        doc_type="court_decision" if restricted else None,
        operation=Operation.LOOKUP,
        filters=filters,
    )


def resolver(plans):
    return ScopeResolver(plans, IdentifierResolver(FakeSource(DOCS)))


class TestFiltersAlone:
    def test_a_plan_with_no_type_and_no_filter_restricts_nothing(self):
        plans = FakePlans()

        scope = resolver(plans).resolve(lookup(restricted=False))

        assert scope == Scope()
        assert plans.executed == []  # nothing to run

    def test_the_filters_select_the_documents(self):
        plans = FakePlans(count=5)

        scope = resolver(plans).resolve(lookup())

        assert isinstance(scope, Scope)
        assert scope.selection == SELECTED and scope.note is None

    def test_documents_that_could_not_be_checked_are_said_so(self):
        scope = resolver(FakePlans(count=5, unknown=3)).resolve(lookup())

        assert isinstance(scope, Scope) and scope.note is not None
        assert "3 document(s) could not be checked" in scope.note
        assert "issuing_body is 'Debrecen'" in scope.note

    def test_filters_that_select_no_document_are_an_explicit_refusal(self):
        refusal = resolver(FakePlans(count=0, unknown=2)).resolve(lookup())

        assert isinstance(refusal, Declined)
        assert refusal.reason is DeclineReason.NO_MATCHING_DOCUMENTS
        assert refusal.stage == "scope"
        assert refusal.detail == "court_decision documents: issuing_body is 'Debrecen'"
        assert refusal.note and "2 document(s)" in refusal.note


class TestAnIdentifierNamesTheDocument:
    def test_the_scope_is_the_named_document(self):
        scope = resolver(FakePlans()).resolve(
            lookup(restricted=False), ["4.P.20.409/2023/4"]
        )

        assert isinstance(scope, Scope)
        assert scope.selection == DocumentSelection(
            "SELECT id FROM documents WHERE id = ANY(%s)", ([1],)
        )
        assert scope.note is None

    def test_with_no_filter_the_planner_is_not_asked_at_all(self):
        plans = FakePlans()

        resolver(plans).resolve(lookup(restricted=False), ["4.P.20.409/2023/4"])

        assert plans.executed == [] and plans.asked == []

    def test_a_filter_that_agrees_adds_no_note(self):
        plans = FakePlans(matching={1})

        scope = resolver(plans).resolve(lookup(), ["4.P.20.409/2023/4"])

        assert isinstance(scope, Scope) and scope.note is None
        assert plans.asked == [(1,)]  # asked only about the named document

    def test_a_filter_that_disagrees_does_not_win_and_the_note_says_so(self):
        """The slip in the question (wrong court) must not exclude the named case."""
        plans = FakePlans(matching=set())  # the named document is not a Debrecen one

        scope = resolver(plans).resolve(lookup(), ["4.P.20.409/2023/4"])

        assert isinstance(scope, Scope)
        assert scope.selection == DocumentSelection(
            "SELECT id FROM documents WHERE id = ANY(%s)", ([1],)
        )
        assert scope.note is not None
        assert "was not applied" in scope.note
        assert "1 of the 1 document(s) named by identifier" in scope.note
        assert "issuing_body is 'Debrecen'" in scope.note
        assert plans.executed == []  # the filters were not run over the corpus

    def test_two_identifiers_give_the_union_of_their_documents(self):
        scope = resolver(FakePlans()).resolve(
            lookup(restricted=False), ["4.P.20.409/2023/4", "27.P.20.339/2021/37"]
        )

        assert isinstance(scope, Scope)
        assert scope.selection and scope.selection.params == ([1, 2],)

    def test_a_note_counts_only_the_documents_the_filter_leaves_out(self):
        plans = FakePlans(matching={1})  # 1 agrees, 2 does not

        scope = resolver(plans).resolve(
            lookup(), ["4.P.20.409/2023/4", "27.P.20.339/2021/37"]
        )

        assert isinstance(scope, Scope) and scope.note is not None
        assert "1 of the 2 document(s)" in scope.note


class TestAnIdentifierThatResolvesToNothing:
    def test_it_neither_narrows_nor_refuses(self):
        scope = resolver(FakePlans()).resolve(
            lookup(restricted=False), ["99.P.99.999/2099/1"]
        )

        assert isinstance(scope, Scope)
        assert scope.selection is None  # the retrieval runs as it would without it
        assert scope.identifiers.unresolved == ("99.P.99.999/2099/1",)

    def test_the_filters_still_apply(self):
        plans = FakePlans(count=5)

        scope = resolver(plans).resolve(lookup(), ["99.P.99.999/2099/1"])

        assert isinstance(scope, Scope)
        assert scope.selection == SELECTED
        assert scope.identifiers.unresolved == ("99.P.99.999/2099/1",)

    def test_a_resolved_one_beside_an_unresolved_one_names_its_document(self):
        scope = resolver(FakePlans()).resolve(
            lookup(restricted=False), ["P.20.457/2018/11", "4.P.20.409/2023/4"]
        )

        assert isinstance(scope, Scope)
        assert scope.selection and scope.selection.params == ([1],)
        assert scope.identifiers.unresolved == ("P.20.457/2018/11",)


def test_the_resolution_is_kept_for_the_explain_record():
    scope = resolver(FakePlans()).resolve(lookup(restricted=False), ["HU001"])

    assert isinstance(scope, Scope)
    assert scope.identifiers.ambiguous == ("HU001",)  # two documents share it
    assert scope.selection and scope.selection.params == ([4, 5],)


@pytest.mark.parametrize("junk", ["", " ./ "])
def test_an_identifier_that_is_only_punctuation_changes_nothing(junk):
    plans = FakePlans()

    scope = resolver(plans).resolve(lookup(restricted=False), [junk])

    assert isinstance(scope, Scope)
    assert scope == Scope(identifiers=scope.identifiers)  # an unrestricted scope
    assert plans.executed == []
