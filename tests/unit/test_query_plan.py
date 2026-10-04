"""Tests for metadata.plan.parse_plan: the shape of an LLM-produced plan."""

import pytest

from metadata.plan import (
    DEFAULT_LIMIT,
    Filter,
    FilterOp,
    Operation,
    PlanError,
    parse_plan,
)

GOOD = {
    "operation": "count",
    "filters": [
        {"key": "issuing_body", "op": "eq", "value": "Debreceni Ítélőtábla"},
        {
            "key": "decision_date",
            "op": "between",
            "value": {"kind": "calendar", "year_offset": -1, "month": 10},
        },
    ],
    "group_by": None,
    "limit": 20,
    "residual": None,
}


def test_a_good_plan_parses_into_typed_parts():
    plan = parse_plan(GOOD, "court_decision")

    assert plan.doc_type == "court_decision"
    assert plan.operation is Operation.COUNT
    assert plan.limit == 20
    assert plan.filters[0] == Filter(
        "issuing_body", FilterOp.EQ, "Debreceni Ítélőtábla"
    )
    assert plan.filters[1].op is FilterOp.BETWEEN
    assert plan.filters[1].value["month"] == 10


def test_defaults_a_minimal_plan_needs_only_an_operation():
    plan = parse_plan({"operation": "list"}, "invoice")

    assert (plan.filters, plan.group_by, plan.sum_key, plan.limit, plan.residual) == (
        (),
        None,
        None,
        DEFAULT_LIMIT,
        None,
    )


def test_the_doc_type_comes_from_the_caller_never_from_the_model():
    with pytest.raises(PlanError, match="unknown plan field"):
        parse_plan({"operation": "list", "doc_type": "secrets"}, "court_decision")


def test_group_by_and_sum_are_accepted_where_they_belong():
    assert (
        parse_plan({"operation": "count", "group_by": "document_kind"}, "t").group_by
        == "document_kind"
    )
    assert (
        parse_plan({"operation": "sum", "sum_key": "legal_costs_awarded"}, "t").sum_key
        == "legal_costs_awarded"
    )


def test_a_residual_is_kept_as_text():
    plan = parse_plan(
        {"operation": "overview", "residual": "cases about expropriation"}, "t"
    )

    assert plan.residual == "cases about expropriation"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("count them", "must be an object"),
        ([], "must be an object"),
        ({}, "needs an 'operation'"),
        ({"operation": "delete"}, "unknown operation 'delete'"),
        ({"operation": "count", "sql": "DROP TABLE documents"}, "unknown plan field"),
        ({"operation": "count", "filters": "all"}, "'filters' must be a list"),
        (
            {"operation": "count", "filters": [{"key": "k", "op": "eq"}]},
            "needs a 'value'",
        ),
        (
            {"operation": "count", "filters": [{"key": "k", "op": "like", "value": 1}]},
            "unknown filters\\[0\\] operator",
        ),
        (
            {"operation": "count", "filters": [{"key": "", "op": "eq", "value": 1}]},
            "non-empty string",
        ),
        (
            {
                "operation": "count",
                "filters": [{"key": "k", "op": "eq", "value": 1, "extra": 1}],
            },
            "unknown field",
        ),
        ({"operation": "list", "group_by": "kind"}, "only goes with the 'count'"),
        ({"operation": "count", "sum_key": "amount"}, "only goes with the 'sum'"),
        ({"operation": "sum"}, "needs a 'sum_key'"),
        ({"operation": "count", "group_by": ""}, "non-empty string or null"),
        ({"operation": "count", "limit": 0}, "'limit' must be"),
        ({"operation": "count", "limit": 5000}, "'limit' must be"),
        ({"operation": "count", "limit": "ten"}, "'limit' must be"),
        ({"operation": "count", "limit": True}, "'limit' must be"),
    ],
)
def test_a_malformed_or_inconsistent_plan_is_rejected_with_a_message(raw, message):
    with pytest.raises((PlanError, TypeError), match=message):
        parse_plan(raw, "t")
