"""Wording an exact result: the plain facts, and the model's phrasing checked against them."""

from decimal import Decimal
from types import SimpleNamespace

from metadata.executor import PlanResult
from metadata.plan import Filter, FilterOp, Operation, QueryPlan
from query.answering import ResultPhraser, render_result

DT = "court_decision"
FILTER = (Filter("issuing_body", FilterOp.EQ, "X"),)


def _count_result(**kw):
    kw.setdefault("count", 7)
    return PlanResult(
        Operation.COUNT, "issuing_body = 'X'", unknown=kw.pop("unknown", 0), **kw
    )


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
