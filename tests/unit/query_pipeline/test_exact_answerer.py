"""Carrying out an exact plan and wording the result (the wording itself has its own tests,
in test_query_router.py, where the phraser was first written)."""

import pytest

from metadata.executor import PlanResult
from metadata.plan import Filter, FilterOp, Operation, PlanError, QueryPlan
from query.answering import ExactAnswerer

PLAN = QueryPlan(
    "court_decision",
    Operation.COUNT,
    (Filter("issuing_body", FilterOp.EQ, "Debrecen"),),
)
RESULT = PlanResult(Operation.COUNT, "issuing_body = 'Debrecen'", count=7, unknown=0)


class RecordingExecutor:
    def __init__(self, result=RESULT, error=None):
        self.result, self.error, self.executed = result, error, []

    def execute(self, plan):
        self.executed.append(plan)
        if self.error:
            raise self.error
        return self.result


class RecordingPhraser:
    def __init__(self):
        self.calls = []

    def phrase(self, question, plan, result):
        self.calls.append((question, plan, result))
        return "PHRASED"


def make(**kwargs):
    executor = RecordingExecutor(**kwargs)
    phraser = RecordingPhraser()
    return ExactAnswerer(executor, phraser), executor, phraser


class TestExactAnswerer:
    def test_it_executes_the_plan_and_words_what_it_found(self):
        answerer, executor, phraser = make()

        text = answerer.answer("Hány ítélet?", PLAN)

        assert text == "PHRASED"
        assert executor.executed == [PLAN]  # once, and the plan as given
        assert phraser.calls == [
            ("Hány ítélet?", PLAN, RESULT)
        ]  # the result of that run

    def test_it_does_not_word_anything_before_the_plan_has_run(self):
        answerer, _, phraser = make(error=PlanError("bad plan"))

        with pytest.raises(PlanError):
            answerer.answer("q", PLAN)

        assert phraser.calls == []

    def test_a_plan_that_no_longer_fits_the_catalog_is_not_hidden(self):
        answerer, _, _ = make(error=PlanError("unknown key 'x'"))

        with pytest.raises(PlanError, match="unknown key"):
            answerer.answer("q", PLAN)
