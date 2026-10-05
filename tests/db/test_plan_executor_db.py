"""DB-gated tests for the plan compiler + executor against a real Postgres.

The point of these is the thing the pure compiler tests cannot show: that the
generated SQL returns the right documents, counts the right unknowns, and is safe.
Requires AGENT_ENV=test -- see tests/db/conftest.py.
"""

from datetime import date
from decimal import Decimal

import psycopg2
import pytest

from config import settings
from document_store import DocumentStore
from metadata.clock import FixedClock
from metadata.compiler import PlanCompiler
from metadata.date_ranges import DateRangeResolver
from metadata.executor import PlanExecutor
from metadata.plan import parse_plan
from models import (
    Document,
    KeyStatus,
    MetaKey,
    MetaSource,
    MetaState,
    MetaStatus,
    MetaValue,
    ValueType,
)

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(settings.AGENT_ENV != "test", reason="AGENT_ENV is not 'test'"),
]

DT = "court_decision"
TODAY = date(2026, 10, 4)


def _h(n: int) -> str:
    return f"{n:064x}"


def _put(
    store,
    n,
    court,
    when,
    kind,
    costs=None,
    statuses=("issuing_body", "decision_date", "document_kind"),
):
    """One document with its values; every listed key gets a PRESENT status."""
    store.upsert_document(
        Document(_h(n), f"{court.split()[0]}__doc_{n}.docx", summary=f"summary {n}")
    )
    rows = [
        ("issuing_body", {"value_text": court}),
        ("decision_date", {"value_date": when}),
        ("document_kind", {"value_text": kind}),
    ]
    if costs is not None:
        rows.append(("legal_costs", {"value_number": Decimal(costs)}))
    for key, value in rows:
        if value and any(v is not None for v in value.values()):
            store.add_value(MetaValue(_h(n), key, 1, MetaSource.LLM, **value))
    for key in statuses:
        store.set_status(MetaStatus(_h(n), key, MetaState.PRESENT, key_version=1))


@pytest.fixture
def world(db_conn):
    store = DocumentStore()
    store.ensure_type(DT)
    for name, vtype, allowed in (
        ("issuing_body", ValueType.TEXT, None),
        ("decision_date", ValueType.DATE, None),
        ("document_kind", ValueType.TEXT, ("judgment", "order", "other")),
        ("legal_costs", ValueType.NUMBER, None),
    ):
        store.upsert_key(
            MetaKey(
                DT, name, vtype, "d", allowed_values=allowed, status=KeyStatus.APPROVED
            )
        )
    # Debrecen: two decisions last October, one in September; Eger: one last October
    _put(store, 1, "Debreceni Ítélőtábla", date(2025, 10, 3), "judgment", costs=1000)
    _put(store, 2, "Debreceni Ítélőtábla", date(2025, 10, 28), "order", costs=2500)
    _put(store, 3, "Debreceni Ítélőtábla", date(2025, 9, 30), "judgment", costs=500)
    _put(store, 4, "Egri Törvényszék", date(2025, 10, 15), "judgment", costs=7000)
    return store


def _run(store, **plan):
    resolver = DateRangeResolver(FixedClock(TODAY))
    return PlanExecutor(store, PlanCompiler(resolver)).execute(parse_plan(plan, DT))


LAST_OCTOBER = {"kind": "calendar", "year_offset": -1, "month": 10}


def test_how_many_decisions_did_a_court_issue_last_october(world):
    """The question the whole design started from."""
    result = _run(
        world,
        operation="count",
        filters=[
            {"key": "issuing_body", "op": "eq", "value": "Debreceni Ítélőtábla"},
            {"key": "decision_date", "op": "between", "value": LAST_OCTOBER},
        ],
    )

    assert (result.count, result.unknown) == (2, 0)
    assert result.explanation == (
        "issuing_body = 'Debreceni Ítélőtábla' AND decision_date between 2025-10-01 and 2025-10-31"
    )


def test_the_range_includes_both_end_days_and_excludes_the_day_outside(world):
    # doc 1 is 2025-10-03, doc 2 is 2025-10-28, doc 3 is 2025-09-30 (the day before October)
    result = _run(
        world,
        operation="list",
        filters=[{"key": "decision_date", "op": "between", "value": LAST_OCTOBER}],
    )

    assert sorted(row[1] for row in result.documents) == [
        "Debreceni__doc_1.docx",
        "Debreceni__doc_2.docx",
        "Egri__doc_4.docx",
    ]


def test_a_list_reports_the_exact_total_beside_a_truncated_list(world):
    result = _run(
        world,
        operation="list",
        limit=1,
        filters=[{"key": "issuing_body", "op": "eq", "value": "Debreceni Ítélőtábla"}],
    )

    assert (result.count, len(result.documents), result.truncated) == (3, 1, True)


def test_overview_carries_the_summaries(world):
    result = _run(
        world,
        operation="overview",
        filters=[{"key": "issuing_body", "op": "eq", "value": "Egri Törvényszék"}],
    )

    assert result.documents == ((_h(4), "Egri__doc_4.docx", "summary 4"),)


def test_lookup_returns_the_document_set_to_restrict_retrieval_to(world):
    result = _run(
        world,
        operation="lookup",
        filters=[{"key": "document_kind", "op": "eq", "value": "order"}],
    )

    assert result.documents == ((_h(2),),)


def test_a_sum_over_the_matching_documents(world):
    result = _run(
        world,
        operation="sum",
        sum_key="legal_costs",
        filters=[{"key": "issuing_body", "op": "eq", "value": "Debreceni Ítélőtábla"}],
    )

    assert (result.total, result.sum_documents) == (Decimal(4000), 3)


def test_a_grouped_count_and_its_total(world):
    result = _run(world, operation="count", group_by="document_kind")

    assert dict(result.groups) == {"judgment": 3, "order": 1}
    assert result.count == 4


def test_number_and_text_filters(world):
    costly = _run(
        world,
        operation="count",
        filters=[{"key": "legal_costs", "op": "gt", "value": 2000}],
    )
    contains = _run(
        world,
        operation="count",
        filters=[{"key": "issuing_body", "op": "contains", "value": "ítélő"}],
    )
    either = _run(
        world,
        operation="count",
        filters=[
            {
                "key": "issuing_body",
                "op": "in",
                "value": ["Egri Törvényszék", "Nowhere"],
            }
        ],
    )

    assert (costly.count, contains.count, either.count) == (2, 3, 1)


def test_documents_that_cannot_be_decided_are_counted_as_unknown_not_dropped(world):
    # doc 5: the court is known, but its date was never attempted
    world.upsert_document(Document(_h(5), "Debreceni__doc_5.docx"))
    world.add_value(
        MetaValue(
            _h(5), "issuing_body", 1, MetaSource.LLM, value_text="Debreceni Ítélőtábla"
        )
    )
    world.set_status(
        MetaStatus(_h(5), "issuing_body", MetaState.PRESENT, key_version=1)
    )
    # doc 6: the date was looked for and the document does not state it -> cannot match
    world.upsert_document(Document(_h(6), "Debreceni__doc_6.docx"))
    world.add_value(
        MetaValue(
            _h(6), "issuing_body", 1, MetaSource.LLM, value_text="Debreceni Ítélőtábla"
        )
    )
    world.set_status(
        MetaStatus(_h(6), "issuing_body", MetaState.PRESENT, key_version=1)
    )
    world.set_status(
        MetaStatus(_h(6), "decision_date", MetaState.CONFIRMED_ABSENT, key_version=1)
    )
    # doc 7: the date is unverified, the court is a different one (known) -> cannot match
    world.upsert_document(Document(_h(7), "Egri__doc_7.docx"))
    world.add_value(
        MetaValue(
            _h(7), "issuing_body", 1, MetaSource.LLM, value_text="Egri Törvényszék"
        )
    )
    world.set_status(
        MetaStatus(_h(7), "issuing_body", MetaState.PRESENT, key_version=1)
    )
    world.set_status(
        MetaStatus(_h(7), "decision_date", MetaState.UNVERIFIED, key_version=1)
    )

    result = _run(
        world,
        operation="count",
        filters=[
            {"key": "issuing_body", "op": "eq", "value": "Debreceni Ítélőtábla"},
            {"key": "decision_date", "op": "between", "value": LAST_OCTOBER},
        ],
    )

    # exact: docs 1 and 2; unknown: only doc 5 (court matches, date undecided)
    assert (result.count, result.unknown) == (2, 1)


def test_a_key_whose_definition_changed_makes_old_extractions_unknown(world):
    world.upsert_key(
        MetaKey(
            DT,
            "issuing_body",
            ValueType.TEXT,
            "sharper",
            status=KeyStatus.APPROVED,
            version=2,
        )
    )

    result = _run(
        world,
        operation="count",
        filters=[{"key": "issuing_body", "op": "eq", "value": "Egri Törvényszék"}],
    )

    # the values are still there and still match, but every status is from version 1
    assert result.count == 1
    assert (
        result.unknown == 3
    )  # the three Debrecen documents are now "not known at version 2"


def test_a_malicious_value_is_data_not_sql(world):
    result = _run(
        world,
        operation="count",
        filters=[
            {"key": "issuing_body", "op": "eq", "value": "x'; DROP TABLE documents; --"}
        ],
    )

    assert result.count == 0
    assert DocumentStore().count_documents() == 4  # the table is still there, untouched


def test_the_executing_connection_is_read_only(world):
    with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
        world.execute_query("DELETE FROM documents", ())

    assert world.count_documents() == 4


def test_a_runaway_query_is_cancelled_by_the_time_limit(world):
    with pytest.raises(psycopg2.errors.QueryCanceled):
        world.execute_query("SELECT pg_sleep(2)", (), timeout_ms=50)
