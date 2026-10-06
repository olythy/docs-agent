"""DB-gated tests of the schema the migrations end in.

The old ``content_hash`` links were replaced by the numeric document id in two
migrations (0005 adds, 0006 removes). These tests pin the *end state*, so a later
migration that brought a dropped column back, or loosened the rule that a chunk, a
value and a status always belong to a document, would fail here. A new database
runs every migration in order and must end up the same.
"""

import pytest

from config import settings

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(
        settings.AGENT_ENV != "test",
        reason="AGENT_ENV is not 'test' -- see tests/db/conftest.py.",
    ),
]

_REFERRERS = ("document_chunks", "document_meta", "document_meta_status")


def _columns(db_conn, table: str) -> dict[str, str]:
    """Column name -> 'YES'/'NO' (nullable) for one table."""
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s;",
            (table,),
        )
        return dict(cur.fetchall())


@pytest.mark.parametrize("table", _REFERRERS)
def test_the_old_hash_link_is_gone_from_every_table_that_refers_to_a_document(
    db_conn, table
):
    assert "content_hash" not in _columns(db_conn, table)


@pytest.mark.parametrize("table", _REFERRERS)
def test_a_chunk_a_value_and_a_status_cannot_exist_without_their_document(
    db_conn, table
):
    assert _columns(db_conn, table)["document_id"] == "NO"  # NOT NULL


@pytest.mark.parametrize("table", _REFERRERS)
def test_deleting_a_document_cascades_to_what_refers_to_it(db_conn, table):
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT confrelid::regclass::text, confdeltype
            FROM pg_constraint
            WHERE contype = 'f' AND conrelid = %s::regclass
              AND EXISTS (
                  SELECT 1 FROM unnest(conkey) k
                  JOIN pg_attribute a ON a.attrelid = conrelid AND a.attnum = k
                  WHERE a.attname = 'document_id')
            """,
            (table,),
        )
        rows = cur.fetchall()

    assert rows == [("documents", "c")]  # 'c' = ON DELETE CASCADE


def test_documents_are_identified_by_id_and_the_hash_is_a_unique_natural_key(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT contype, array_agg(a.attname ORDER BY a.attname)
            FROM pg_constraint c
            JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
            WHERE c.conrelid = 'documents'::regclass AND c.contype IN ('p', 'u')
            GROUP BY c.oid, contype
            """
        )
        constraints = sorted(cur.fetchall())

    assert constraints == [("p", ["id"]), ("u", ["content_hash"])]


def test_values_and_statuses_are_unique_per_document_not_per_hash(db_conn):
    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT conrelid::regclass::text, array_agg(a.attname ORDER BY a.attname)
            FROM pg_constraint c
            JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
            WHERE c.conrelid IN ('document_meta'::regclass, 'document_meta_status'::regclass)
              AND c.contype IN ('p', 'u')
            GROUP BY c.oid, conrelid
            """
        )
        constraints = dict(cur.fetchall())

    assert constraints == {
        "document_meta": ["document_id", "key", "ordinal"],
        "document_meta_status": ["document_id", "key"],
    }
