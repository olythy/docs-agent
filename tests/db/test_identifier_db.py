"""DB-gated tests of the identifier type: the SQL must be the Python rule, and the migration safe.

``metadata/identifiers.py`` defines how identifiers are compared; the compiled SQL repeats
the rule on the stored value. They are written twice, so a test has to keep them equal:
every pair of a stored value and a wished one below is decided by both, and they must
agree. Requires AGENT_ENV=test -- see tests/db/conftest.py.
"""

from datetime import date

import psycopg2
import pytest

from config import settings
from document_store import DocumentStore
from metadata.clock import FixedClock
from metadata.compiler import PlanCompiler
from metadata.date_ranges import DateRangeResolver
from metadata.identifiers import identifier_matches
from metadata.plan import parse_plan
from migrations import base  # noqa: F401  (the package must be importable)
from models import (
    Document,
    KeyStatus,
    MetaKey,
    MetaSource,
    MetaValue,
    ValueType,
)

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(settings.AGENT_ENV != "test", reason="AGENT_ENV is not 'test'"),
]

DT = "court_decision"

#: Stored values, as written, with the awkward cases: spaces, case, full stops, suffixes,
#: a longer number, characters that mean something to LIKE, full-width and no-break forms.
STORED = [
    "4.P.20.409/2023/4",
    "4.P.20.409/2023/4-ítélet",
    "4.P.20.409/2023/40",
    "4.P.20.409/2023/4.",
    "  4.p.20.409/2023/4 ",
    "10. P. 20.277/2019/77.",
    "10.P.20.277/2019/77/ítélet",
    "A_B%1",
    "AXB%1",
    "ＡＢ１２３",
    "x y/1",
    "8.P.21.329/2024/12-III",
    "8.P.XI.21.329/2024/12.",
    "104.K.702.368/2022",
    "104.K.702.368/2022/22",
    "11.P.20.377/2023/11JAV",
]

WISHED = [
    "4.P.20.409/2023/4",
    "4.P.20.409/2023/40",
    "4.p.20.409/2023/4.",
    "10.P.20.277/2019/77",
    "10. P. 20.277/2019/77.",
    "A_B%1",
    "a_b",
    "ab123",
    "xy/1",
    "8.P.21.329/2024/12",
    "8.P.XI.21.329/2024/12",
    "104.K.702.368/2022",
    "104.K.702.368/2022/22",
    "11.P.20.377/2023/11",
    "nothing like these",
]


def _h(n: int) -> str:
    return f"{n:064x}"


@pytest.fixture
def world(db_conn):
    store = DocumentStore()
    store.ensure_type(DT)
    store.upsert_key(
        MetaKey(
            DT,
            "document_identifier",
            ValueType.IDENTIFIER,
            "d",
            status=KeyStatus.APPROVED,
        )
    )
    for n, value in enumerate(STORED):
        store.upsert_document(Document(_h(n), f"doc_{n}.docx", summary="s"))
        store.set_document_type(_h(n), DT)
        store.add_value(
            MetaValue(_h(n), "document_identifier", 1, MetaSource.LLM, value_text=value)
        )
    return store


def _documents_found(store, op, value):
    compiler = PlanCompiler(DateRangeResolver(FixedClock(date(2026, 10, 7))))
    plan = parse_plan(
        {
            "document_type": DT,
            "operation": "list",
            "filters": [{"key": "document_identifier", "op": op, "value": value}],
            "limit": 100,
        },
        [DT],
    )
    keys = store.list_keys(DT, KeyStatus.APPROVED)
    query = compiler.compile(plan, keys)
    rows = store.execute_query(query.sql, query.params)
    return {row[1] for row in rows}  # source_file


@pytest.mark.parametrize("wished", WISHED)
def test_the_sql_decides_exactly_as_the_python_rule_does(world, wished):
    found = _documents_found(world, "eq", wished)

    expected = {
        f"doc_{n}.docx" for n, v in enumerate(STORED) if identifier_matches(v, wished)
    }
    assert found == expected


def test_the_awkward_cases_are_really_in_play(world):
    """Guards the test above against passing because nothing ever matches."""
    assert _documents_found(world, "eq", "4.P.20.409/2023/4") == {
        "doc_0.docx",  # the same
        "doc_1.docx",  # a suffix
        "doc_3.docx",  # a full stop
        "doc_4.docx",  # case and spaces
    }  # and not doc_2 (".../40")
    assert _documents_found(world, "eq", "ab123") == {"doc_9.docx"}  # full-width
    assert _documents_found(world, "eq", "a_b%1") == {
        "doc_7.docx"
    }  # not AXB%1: _ is literal


def test_in_is_any_of_them(world):
    found = _documents_found(
        world, "in", ["10.P.20.277/2019/77", "104.K.702.368/2022/22"]
    )

    assert found == {"doc_5.docx", "doc_6.docx", "doc_14.docx"}


def test_contains_looks_in_the_normalised_form(world):
    found = _documents_found(world, "contains", "p.20.277")

    assert found == {"doc_5.docx", "doc_6.docx"}


def test_the_migration_widens_the_rule_and_its_down_is_safe_with_data(db_conn):
    """Down turns identifier keys back into text (the old rule would refuse them), up widens again."""
    from importlib import import_module

    migration = import_module("migrations.0007_allow_identifier_value_type")
    step = migration.AllowIdentifierValueType()
    DocumentStore().ensure_type("t")  # a key belongs to a document type
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO meta_keys (doc_type, key, value_type, description) "
            "VALUES ('t', 'k', 'identifier', 'd');"
        )
    try:
        step.down(db_conn)
        with db_conn.cursor() as cur:
            cur.execute("SELECT value_type FROM meta_keys WHERE key = 'k';")
            row = cur.fetchone()
            assert row is not None and row[0] == "text"
            with pytest.raises(psycopg2.errors.CheckViolation):
                cur.execute(
                    "INSERT INTO meta_keys (doc_type, key, value_type, description) "
                    "VALUES ('t', 'j', 'identifier', 'd');"
                )
        db_conn.rollback()  # leave the schema as the migrations made it
    finally:
        db_conn.rollback()


def test_after_the_migration_an_identifier_key_can_be_stored(db_conn):
    DocumentStore().ensure_type("t")
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO meta_keys (doc_type, key, value_type, description) "
            "VALUES ('t', 'k', 'identifier', 'd') RETURNING value_type;"
        )
        row = cur.fetchone()

    assert row is not None and row[0] == "identifier"
