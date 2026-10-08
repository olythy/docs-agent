"""The end-to-end order: decide, then refuse, answer exactly, or retrieve and write."""

from typing import Any

import pytest

from metadata.plan import Filter, FilterOp, Operation, QueryPlan
from models import ChunkMetadata, DocumentSelection, RetrievedChunk
from query.answering import GroundedAnswer
from query.decision import AnswerExactly, ReadDocuments, Refuse, Scope
from query.facts import QueryFacts, QueryFactsReader
from query.outcome import (
    NO_RESULTS_MESSAGE,
    Answerable,
    Declined,
    DeclineReason,
    RefusalRenderer,
)
from query.query_service import QueryService
from query.service import RetrievalResult

PLAN = QueryPlan("court_decision", Operation.COUNT, (Filter("k", FilterOp.EQ, "v"),))
CHUNK = RetrievedChunk(
    id=1,
    content="c",
    metadata=ChunkMetadata(source_file="a.pdf", page_number=1, chunk_index=0),
    score=0.5,
)
STORE: Any = object()


class FixedDecider:
    def __init__(self, make):
        self.make, self.seen = make, []

    def decide(self, facts):
        self.seen.append(facts)
        return self.make(facts)


class FakeRetrieval:
    def __init__(self, outcome, scope=None):
        self.outcome, self.scope, self.calls = outcome, scope, []

    def retrieve(self, request, store):
        self.calls.append((request, store))
        return RetrievalResult(self.outcome, (), request.scope or Scope())


class FakeExact:
    def __init__(self):
        self.calls = []

    def answer(self, question, plan):
        self.calls.append((question, plan))
        return "EXACT"


class FakeGrounded:
    def __init__(self, text="GROUNDED", refused=False):
        self.reply, self.calls = GroundedAnswer(text, refused), []

    def answer(self, question, chunks):
        self.calls.append((question, chunks))
        return self.reply


def service(make, outcome=None, grounded=None):
    decider = FixedDecider(make)
    retrieval = FakeRetrieval(outcome or Answerable((CHUNK,)))
    fake_exact = FakeExact()
    fake_grounded = grounded or FakeGrounded()
    qs = QueryService(
        QueryFactsReader(),
        decider,
        retrieval,
        fake_exact,
        fake_grounded,
        RefusalRenderer(),
        default_profile="best_chunks",
    )
    return qs, decider, retrieval, fake_exact, fake_grounded


def read(profile="best_chunks", scope=None):
    return lambda facts: ReadDocuments(facts, None, profile, scope or Scope())


class TestRefusal:
    def test_a_refusal_is_worded_and_nothing_else_runs(self):
        declined = Declined(DeclineReason.COULD_NOT_INTERPRET, "planning")
        qs, _, retrieval, exact, grounded = service(lambda f: Refuse(f, declined))

        answer = qs.answer("q", store=STORE)

        assert answer.text == RefusalRenderer().render(declined)
        assert answer.explain.declined is declined
        assert answer.explain.retrieval is None
        assert not (retrieval.calls or exact.calls or grounded.calls)


class TestExact:
    def test_an_exact_decision_is_carried_out_and_nothing_is_read(self):
        qs, _, retrieval, exact, grounded = service(lambda f: AnswerExactly(f, PLAN))

        answer = qs.answer("How many?", store=STORE)

        assert answer.text == "EXACT"
        assert exact.calls == [("How many?", PLAN)]
        assert not (retrieval.calls or grounded.calls)
        assert isinstance(answer.explain.decision, AnswerExactly)


class TestRead:
    def test_the_decisions_profile_and_scope_reach_the_retrieval(self):
        scope = Scope(selection=DocumentSelection("SELECT 1", ()), note="n")
        qs, _, retrieval, *_ = service(read("another_profile", scope))

        qs.answer("q", top_k=3, min_score=0.2, store=STORE)

        request, store = retrieval.calls[0]
        assert (request.question, request.profile, request.scope) == (
            "q",
            "another_profile",
            scope,
        )
        assert (request.top_k, request.min_score) == (3, 0.2)
        assert store is STORE

    def test_the_answer_is_written_from_the_chunks_that_were_read(self):
        qs, _, _, _, grounded = service(read())

        answer = qs.answer("q", store=STORE)

        assert answer.text == "GROUNDED"
        assert grounded.calls == [("q", (CHUNK,))]
        assert answer.explain.retrieval is not None
        assert answer.explain.declined is None and not answer.explain.model_refused

    def test_the_scope_note_is_appended_to_the_answer(self):
        qs, *_ = service(read(scope=Scope(note="2 documents were left out")))

        assert (
            qs.answer("q", store=STORE).text == "GROUNDED\n\n2 documents were left out"
        )

    def test_a_retrieval_refusal_is_worded_keeps_the_note_and_asks_no_model(self):
        declined = Declined(DeclineReason.NOT_RELEVANT, "relevance_gate")
        qs, _, _, _, grounded = service(
            read(scope=Scope(note="NOTE")), outcome=declined
        )

        answer = qs.answer("q", store=STORE)

        assert answer.text == f"{NO_RESULTS_MESSAGE}\n\nNOTE"
        assert answer.explain.declined is declined
        assert grounded.calls == []

    def test_the_models_own_refusal_is_told_as_it_is_and_marked(self):
        qs, *_ = service(read(), grounded=FakeGrounded("I could not find ...", True))

        answer = qs.answer("q", store=STORE)

        assert answer.text == "I could not find ..."
        assert answer.explain.model_refused


def test_a_given_profile_replaces_the_decided_one_and_nothing_else():
    scope = Scope(note="n")
    qs, _, retrieval, *_ = service(read("best_chunks", scope))

    answer = qs.answer("q", profile="another_profile", store=STORE)

    assert retrieval.calls[0][0].profile == "another_profile"
    assert retrieval.calls[0][0].scope is scope
    assert answer.explain.decision.profile == "another_profile"  # type: ignore[union-attr]


def test_the_decider_gets_the_facts_of_the_question():
    qs, decider, *_ = service(read())

    qs.answer("Mi volt a 4.P.20.409/2023/4 ügyben?", store=STORE)

    (facts,) = decider.seen
    assert isinstance(facts, QueryFacts) and facts.identifiers == ("4.P.20.409/2023/4",)


class TestACallerFixedScope:
    def test_no_decision_is_taken_and_the_scope_is_read_with_the_default_profile(self):
        scope = Scope(note="one file")
        qs, decider, retrieval, exact, _ = service(read())

        answer = qs.answer("q", scope=scope, store=STORE)

        assert decider.seen == []  # nobody was asked
        request = retrieval.calls[0][0]
        assert (request.scope, request.profile) == (scope, "best_chunks")
        assert answer.text == "GROUNDED\n\none file"
        assert isinstance(answer.explain.decision, ReadDocuments)
        assert answer.explain.decision.plan is None
        assert not exact.calls

    def test_a_given_profile_still_wins(self):
        qs, _, retrieval, *_ = service(read())

        qs.answer("q", scope=Scope(), profile="another_profile", store=STORE)

        assert retrieval.calls[0][0].profile == "another_profile"


class TestExplainChunks:
    def test_the_chunks_are_those_the_answer_was_written_from(self):
        qs, *_ = service(read())

        assert qs.answer("q", store=STORE).explain.chunks == (CHUNK,)

    @pytest.mark.parametrize(
        "make, outcome",
        [
            (lambda f: AnswerExactly(f, PLAN), None),
            (
                lambda f: Refuse(f, Declined(DeclineReason.NOT_SUPPORTED, "planning")),
                None,
            ),
            (read(), Declined(DeclineReason.NOT_RELEVANT, "relevance_gate")),
        ],
        ids=["exact", "refused by the decision", "refused by the retrieval"],
    )
    def test_nothing_read_or_kept_means_no_chunks(self, make, outcome):
        qs, *_ = service(make, outcome=outcome)

        assert qs.answer("q", store=STORE).explain.chunks == ()
