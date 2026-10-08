"""Tests for the pure parts of corpus.commands.routing_eval."""

import json

import pytest

from corpus.commands.routing_eval import (
    CASES,
    QUESTIONS,
    flow_of_decision,
    load_cases,
)
from metadata.plan import Operation, QueryPlan
from query.decision import AnswerExactly, ReadDocuments, Refuse, Scope
from query.facts import QueryFacts
from query.outcome import Declined, DeclineReason


def test_the_shipped_cases_load_resolve_golden_ids_and_cover_every_flow():
    cases = load_cases()

    assert {c["expected"] for c in cases} == {"lookup", "exact", "unsupported"}
    assert all(c["question"].strip() for c in cases)
    assert any(
        "27.P.20.339/2021/37" in c["question"] for c in cases
    )  # a resolved golden id
    assert CASES.exists() and QUESTIONS.exists()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ({"expected": "lookup"}, "exactly one of"),
        (
            {"question": "q", "golden_id": "q0001", "expected": "lookup"},
            "exactly one of",
        ),
        ({"question": "q", "expected": "guess"}, "unknown expected flow"),
        ({"golden_id": "q9999", "expected": "lookup"}, "no golden question"),
    ],
)
def test_a_malformed_case_is_rejected(tmp_path, case, message):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [case]}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_cases(path)


FACTS = QueryFacts("q")
PLAN = QueryPlan("court_decision", Operation.COUNT)


class TestFlowOfDecision:
    """Every decision lands on one flow, and a refusal says which kind it is."""

    def test_a_question_to_read_is_a_lookup(self):
        decision = ReadDocuments(FACTS, None, "best_chunks", Scope())

        assert flow_of_decision(decision)[0] == "lookup"

    def test_an_exact_answer_is_exact(self):
        assert flow_of_decision(AnswerExactly(FACTS, PLAN))[0] == "exact"

    def test_not_supported_is_unsupported_and_keeps_the_planners_reason(self):
        declined = Declined(DeclineReason.NOT_SUPPORTED, "planning", detail="similar")

        assert flow_of_decision(Refuse(FACTS, declined)) == ("unsupported", "similar")

    def test_filters_that_select_no_document_are_a_routing_failure_not_a_lookup(self):
        declined = Declined(
            DeclineReason.NO_MATCHING_DOCUMENTS, "scope", detail="the filter"
        )

        flow, detail = flow_of_decision(Refuse(FACTS, declined))

        assert flow == "lookup with an empty restriction" and detail == "the filter"

    def test_a_question_that_could_not_be_planned_failed(self):
        declined = Declined(
            DeclineReason.COULD_NOT_INTERPRET, "planning", detail="no json"
        )

        flow, detail = flow_of_decision(Refuse(FACTS, declined))

        assert flow == "failed" and "no json" in detail
