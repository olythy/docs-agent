"""Tests for the pure parts of corpus.commands.routing_eval."""

import json

import pytest

from corpus.commands.routing_eval import CASES, QUESTIONS, flow_of, load_cases


def test_every_operation_lands_on_one_of_the_three_flows():
    assert [flow_of(op) for op in ("lookup", "unsupported")] == [
        "lookup",
        "unsupported",
    ]
    assert {flow_of(op) for op in ("count", "list", "sum", "overview")} == {"exact"}


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


class _Executor:
    def __init__(self, count):
        self.count = count
        self.plans = []

    def execute(self, plan):
        from metadata.executor import PlanResult
        from metadata.plan import Operation

        self.plans.append(plan)
        return PlanResult(
            Operation.LOOKUP, "the executed filter", count=self.count, unknown=0
        )


def _lookup(document_type=None, filters=()):
    from metadata.plan import Filter, FilterOp, Operation, QueryPlan

    return QueryPlan(
        doc_type=document_type,
        operation=Operation.LOOKUP,
        filters=tuple(Filter(k, FilterOp(op), v) for k, op, v in filters),
    )


class TestRestrictionProblem:
    """A lookup restriction that selects no document must be reported, not hidden."""

    def test_a_restriction_that_selects_nothing_is_a_problem(self):
        from corpus.commands.routing_eval import restriction_problem

        plan = _lookup("court_decision", [("decision_date", "between", {"kind": "x"})])

        problem = restriction_problem(plan, _Executor(count=0))

        assert problem and "selects no document" in problem
        assert "the executed filter" in problem  # says which filter

    def test_a_restriction_that_selects_documents_is_fine(self):
        from corpus.commands.routing_eval import restriction_problem

        assert restriction_problem(_lookup("court_decision"), _Executor(3)) is None

    def test_no_restriction_means_nothing_is_executed(self):
        from corpus.commands.routing_eval import restriction_problem

        executor = _Executor(count=0)

        assert restriction_problem(_lookup(), executor) is None
        assert executor.plans == []
