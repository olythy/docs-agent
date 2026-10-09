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


def _grouped_result(**kw):
    base = {
        "operation": Operation.COUNT,
        "explanation": "court_decision documents",
        "count": 10,
        "unknown": 0,
        "groups": (("judgment", 5), ("order", 3)),
        "grouped_documents": 8,
    }
    return PlanResult(**{**base, **kw})


def test_a_grouped_count_says_who_is_in_no_group_and_why():
    result = _grouped_result(
        ungrouped=(("confirmed_absent", 1), ("unverified", 1)),
    )

    text = render_result(
        QueryPlan(DT, Operation.COUNT, group_by="document_kind"), result
    )

    assert "judgment: 5" in text
    assert "(in no group, the document does not state it): 1" in text
    assert "(in no group, its value could not be verified): 1" in text


def test_a_grouped_count_with_everything_labelled_adds_no_gap_line():
    result = _grouped_result(count=8)

    text = render_result(
        QueryPlan(DT, Operation.COUNT, group_by="document_kind"), result
    )

    assert "in no group" not in text
    assert "smaller groups" not in text


def test_a_grouped_count_says_when_the_limit_hid_small_groups():
    result = _grouped_result(count=8, groups=(("judgment", 5),), grouped_documents=8)

    text = render_result(
        QueryPlan(DT, Operation.COUNT, group_by="document_kind"), result
    )

    assert "3 more document(s) are in smaller groups that are not listed" in text


def test_overlapping_groups_are_flagged_instead_of_pretending_to_add_up():
    result = _grouped_result(
        groups_overlap=True, groups=(("a", 7), ("b", 6)), grouped_documents=10
    )

    text = render_result(QueryPlan(DT, Operation.COUNT, group_by="topic"), result)

    assert "can be in more than one group" in text
    assert (
        "smaller groups" not in text
    )  # the shortfall check is meaningless when they overlap


def test_a_phrasing_that_drops_the_documents_in_no_group_falls_back_to_the_facts():
    """The 1 and 2 in no group are part of the answer; leaving them out would hide the gap."""
    result = _grouped_result(
        count=11,
        groups=(("judgment", 5), ("order", 3)),
        grouped_documents=8,
        ungrouped=(("not_attempted", 2), ("unverified", 1)),
    )
    plan = QueryPlan(DT, Operation.COUNT, group_by="document_kind")

    kept = ResultPhraser(
        Llm("11 határozat: 5 ítélet, 3 végzés, 2 nincs kinyerve, 1 nem ellenőrzött.")
    ).phrase("hány?", plan, result)
    dropped = ResultPhraser(Llm("11 határozat: 5 ítélet és 3 végzés.")).phrase(
        "hány?", plan, result
    )

    assert kept.startswith("11 határozat: 5 ítélet, 3 végzés")
    assert dropped.startswith("Executed filter:")
    assert "in no group" in dropped


def test_a_caveat_of_the_plan_is_in_the_plain_facts_and_after_the_phrased_answer():
    plan = QueryPlan(DT, Operation.COUNT, caveat="counted by a model")
    result = _count_result(count=7)

    assert "Read with care: counted by a model" in render_result(plan, result)
    phrased = ResultPhraser(Llm("Összesen 7 határozat van.")).phrase(
        "hány?", plan, result
    )
    assert phrased.endswith(
        "[Read with care: counted by a model]"
    )  # not left to the model


def test_a_plan_without_a_caveat_adds_no_such_line():
    plan = QueryPlan(DT, Operation.COUNT)
    result = _count_result(count=7)

    assert "Read with care" not in render_result(plan, result)
    assert "Read with care" not in ResultPhraser(Llm("7 határozat.")).phrase(
        "hány?", plan, result
    )


def test_documents_that_state_no_filtered_value_are_told_and_must_be_mentioned():
    result = _count_result(count=10, unstated=3)
    plan = QueryPlan(DT, Operation.COUNT)

    assert (
        "3 further document(s) do not state a value a filter asked for"
        in render_result(plan, result)
    )
    kept = ResultPhraser(
        Llm("10 határozat, további 3 nem tünteti fel a dátumot.")
    ).phrase("hány?", plan, result)
    dropped = ResultPhraser(Llm("10 határozat van.")).phrase("hány?", plan, result)
    assert kept.startswith("10 határozat, további 3")
    assert dropped.startswith(
        "Executed filter:"
    )  # a dropped figure falls back to the facts


def test_without_unstated_documents_no_such_line_is_written():
    assert "do not state a value" not in render_result(
        QueryPlan(DT, Operation.COUNT), _count_result(count=10)
    )


def _yearly_result():
    return _count_result(
        count=5,
        yearly=((2024, 3), (2025, 1)),
        by_period=((2024, "judgment", 2), (2024, "order", 1), (2025, "judgment", 1)),
        groups=(("judgment", 3), ("order", 1)),
        grouped_documents=4,
        ungrouped=(("confirmed_absent", 1),),
    )


def test_a_count_by_year_is_listed_year_by_year_with_the_groups_inside_each_year():
    plan = QueryPlan(
        DT, Operation.COUNT, group_by="document_kind", by_year="decision_date"
    )

    text = render_result(plan, _yearly_result())

    assert "By year (decision_date):" in text
    assert "  2024: 3 (judgment 2, order 1)" in text
    assert "  2025: 1 (judgment 1)" in text


def test_the_year_table_is_added_by_the_code_after_the_phrased_answer():
    plan = QueryPlan(
        DT, Operation.COUNT, group_by="document_kind", by_year="decision_date"
    )

    out = ResultPhraser(
        Llm("5 határozat: 3 ítélet, 1 végzés, 1 csoport nélküli.")
    ).phrase("hány?", plan, _yearly_result())

    assert "  2024: 3 (judgment 2, order 1)" in out
    assert (
        out.index("5 határozat") < out.index("By year") < out.index("[Executed filter")
    )


def test_year_figures_are_not_required_of_the_model_but_the_totals_are():
    plan = QueryPlan(
        DT, Operation.COUNT, group_by="document_kind", by_year="decision_date"
    )
    result = _yearly_result()

    kept = ResultPhraser(
        Llm("5 határozat: 3 ítélet, 1 végzés, 1 csoport nélküli.")
    ).phrase("hány?", plan, result)
    dropped = ResultPhraser(Llm("5 határozat van.")).phrase("hány?", plan, result)

    assert kept.startswith("5 határozat: 3 ítélet")
    assert dropped.startswith(
        "Executed filter:"
    )  # a dropped total falls back to the facts


def test_with_no_dated_document_the_year_line_says_so():
    plan = QueryPlan(DT, Operation.COUNT, by_year="decision_date")

    assert "no matching document has a value for it" in render_result(
        plan, _count_result(count=2)
    )


def test_a_plan_without_by_year_prints_no_year_table():
    assert "By year" not in render_result(
        QueryPlan(DT, Operation.COUNT), _count_result(count=2)
    )


def test_the_model_is_told_not_to_repeat_a_year_table_that_the_code_adds():
    plan = QueryPlan(
        DT, Operation.COUNT, group_by="document_kind", by_year="decision_date"
    )
    llm = Llm("5 határozat: 3 ítélet, 1 végzés, 1 csoport nélküli.")
    asked = []
    llm.run_tool_calling_turn = lambda messages: (
        asked.append(messages[0]["content"]),
        SimpleNamespace(content=llm.reply),
    )[1]

    ResultPhraser(llm).phrase("hány?", plan, _yearly_result())
    ResultPhraser(llm).phrase(
        "hány?", QueryPlan(DT, Operation.COUNT), _count_result(count=5)
    )

    assert "do not repeat it" in asked[0]
    assert "do not repeat it" not in asked[1]  # no table, nothing to warn about
