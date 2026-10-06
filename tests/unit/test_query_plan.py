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

TYPES = ["court_decision", "invoice"]


def _parse(raw, types=TYPES):
    """Parse a plan; a plan that names no document type gets the first one (most tests
    are about something else than the type, which has its own tests below)."""
    if isinstance(raw, dict) and "document_type" not in raw:
        raw = {"document_type": types[0], **raw}
    return parse_plan(raw, types)


GOOD = {
    "document_type": "court_decision",
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
    plan = parse_plan(GOOD, TYPES)

    assert plan.doc_type == "court_decision"
    assert plan.operation is Operation.COUNT
    assert plan.limit == 20
    assert plan.filters[0] == Filter(
        "issuing_body", FilterOp.EQ, "Debreceni Ítélőtábla"
    )
    assert plan.filters[1].op is FilterOp.BETWEEN
    assert plan.filters[1].value["month"] == 10


def test_defaults_a_minimal_plan_needs_only_an_operation_and_a_document_type():
    plan = parse_plan({"document_type": "invoice", "operation": "list"}, TYPES)

    assert (plan.filters, plan.group_by, plan.sum_key, plan.limit, plan.residual) == (
        (),
        None,
        None,
        DEFAULT_LIMIT,
        None,
    )


def test_the_type_must_be_one_the_caller_offered_the_model_cannot_invent_one():
    with pytest.raises(
        PlanError,
        match=r"unknown document_type 'secrets'; use one of court_decision, invoice",
    ):
        _parse({"document_type": "secrets", "operation": "list"})


def test_a_plan_must_name_its_document_type_even_when_it_is_null():
    with pytest.raises(
        PlanError,
        match=r"needs a 'document_type': one of court_decision, invoice, or null",
    ):
        parse_plan({"operation": "list"}, TYPES)


def test_a_lookup_may_name_no_type_and_then_has_no_filters():
    plan = parse_plan({"document_type": None, "operation": "lookup"}, TYPES)

    assert plan.doc_type is None and plan.filters == ()
    with pytest.raises(PlanError, match="filters are keys of a document type"):
        parse_plan(
            {
                "document_type": None,
                "operation": "lookup",
                "filters": [{"key": "k", "op": "eq", "value": 1}],
            },
            TYPES,
        )


@pytest.mark.parametrize("operation", ["count", "list", "overview", "sum"])
def test_an_exact_answer_needs_a_document_type(operation):
    raw = {
        "document_type": None,
        "operation": operation,
        "sum_key": "x" if operation == "sum" else None,
    }

    with pytest.raises(
        PlanError, match=f"the '{operation}' operation needs a document_type"
    ):
        parse_plan(raw, TYPES)


def test_a_type_list_that_is_empty_says_so_in_the_message():
    with pytest.raises(PlanError, match=r"none is known|\(none known\)"):
        parse_plan({"document_type": None, "operation": "count"}, [])


def test_group_by_and_sum_are_accepted_where_they_belong():
    assert (
        _parse({"operation": "count", "group_by": "document_kind"}).group_by
        == "document_kind"
    )
    assert (
        _parse({"operation": "sum", "sum_key": "legal_costs_awarded"}).sum_key
        == "legal_costs_awarded"
    )


def test_a_residual_is_kept_as_text():
    plan = _parse({"operation": "overview", "residual": "cases about expropriation"})

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
        _parse(raw)


def test_an_unsupported_plan_carries_the_reason_and_needs_no_type():
    plan = parse_plan(
        {
            "document_type": None,
            "operation": "unsupported",
            "reason": "five cases similar to a named one",
        },
        TYPES,
    )

    assert plan.operation is Operation.UNSUPPORTED
    assert plan.reason == "five cases similar to a named one" and plan.doc_type is None


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({}, "needs a 'reason'"),
        ({"reason": "  "}, "needs a 'reason'"),
        (
            {"reason": "x", "filters": [{"key": "k", "op": "eq", "value": 1}]},
            "has no filters",
        ),
    ],
)
def test_a_malformed_unsupported_plan_is_rejected(extra, message):
    raw = {"document_type": "court_decision", "operation": "unsupported", **extra}

    with pytest.raises(PlanError, match=message):
        parse_plan(raw, TYPES)


def test_a_reason_belongs_to_the_unsupported_operation_only():
    with pytest.raises(PlanError, match="only goes with the 'unsupported'"):
        _parse({"operation": "count", "reason": "why"})
