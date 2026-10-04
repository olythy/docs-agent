"""DB-gated tests for DocumentStore and the structured-metadata schema (migration 0004).

Requires AGENT_ENV=test (and a .env.test) -- see tests/db/conftest.py and
config.py. Chunks are written with the real VectorStore.save() using the
embedding dimension the schema declares, so the generated ``content_hash``
column is exercised exactly as ingestion would populate it.
"""

from datetime import date
from decimal import Decimal

import psycopg2
import pytest

from config import settings
from document_store import DocumentStore
from models import (
    Chunk,
    ChunkMetadata,
    Document,
    KeyStatus,
    MetaKey,
    MetaSource,
    MetaState,
    MetaStatus,
    MetaValue,
    ValueType,
)
from store import VectorStore

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(
        settings.AGENT_ENV != "test",
        reason=(
            "AGENT_ENV is not 'test' -- no test database is configured. "
            "Run with AGENT_ENV=test (e.g. `make test`)."
        ),
    ),
]

HASH_A = "a" * 64
HASH_B = "b" * 64


def _save_chunks(content_hash: str, source_file: str, summary: str | None, n: int = 2):
    chunks = [
        Chunk(
            content=f"{source_file} chunk {i}",
            metadata=ChunkMetadata(
                source_file=source_file,
                page_number=1,
                chunk_index=i,
                content_hash=content_hash,
                document_summary=summary,
            ),
        )
        for i in range(n)
    ]
    embeddings = [[0.1] * settings.EMBEDDING_DIMENSION for _ in chunks]
    VectorStore().save(chunks, embeddings)


def test_generated_content_hash_column_is_filled_from_the_metadata(db_conn):
    _save_chunks(HASH_A, "a.docx", "summary a")

    with db_conn.cursor() as cur:
        cur.execute("SELECT DISTINCT content_hash FROM document_chunks;")
        assert cur.fetchall() == [(HASH_A,)]


def test_sync_registers_one_document_per_hash_with_its_summary(db_conn):
    _save_chunks(HASH_A, "a.docx", "summary a", n=3)
    _save_chunks(HASH_B, "b.docx", None)

    result = DocumentStore().sync_from_chunks()

    assert result.upserted == 2
    assert result.removed == 0
    store = DocumentStore()
    assert store.count_documents() == 2
    doc = store.get_document(HASH_A)
    assert doc is not None
    assert (doc.source_file, doc.summary) == ("a.docx", "summary a")
    assert doc.ingested_at is not None
    doc_b = store.get_document(HASH_B)
    assert doc_b is not None and doc_b.summary is None


def test_sync_is_idempotent_and_refreshes_a_changed_summary(db_conn):
    _save_chunks(HASH_A, "a.docx", "old summary")
    DocumentStore().sync_from_chunks()
    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE document_chunks SET metadata = jsonb_set("
            "metadata, '{document_summary}', '\"new summary\"');"
        )
    db_conn.commit()

    result = DocumentStore().sync_from_chunks()

    assert result.removed == 0
    assert DocumentStore().count_documents() == 1
    doc = DocumentStore().get_document(HASH_A)
    assert doc is not None and doc.summary == "new summary"


def test_sync_removes_a_document_whose_chunks_are_gone_and_cascades(db_conn):
    """What a re-ingest that replaced a document leaves behind: the old hash's chunks
    are deleted, so its document row, values and status must go too."""
    _save_chunks(HASH_A, "a.docx", "summary")
    store = DocumentStore()
    store.sync_from_chunks()
    store.add_value(
        MetaValue(
            content_hash=HASH_A,
            key="court",
            key_version=1,
            source=MetaSource.LLM,
            value_text="Egri Törvényszék",
            evidence="Egri Törvényszék",
        )
    )
    store.set_status(MetaStatus(HASH_A, "court", MetaState.PRESENT, key_version=1))
    VectorStore().delete_chunks_by_hash(HASH_A)

    result = store.sync_from_chunks()

    assert result.removed == 1
    assert store.get_document(HASH_A) is None
    assert store.get_values(HASH_A, "court") == []
    assert store.get_status(HASH_A, "court").state is MetaState.NOT_ATTEMPTED


def test_catalog_round_trips_and_filters_by_status(db_conn):
    store = DocumentStore()
    store.upsert_key(
        MetaKey(
            doc_type="court_decision",
            key="document_kind",
            value_type=ValueType.TEXT,
            description="Kind of court document.",
            allowed_values=("judgment", "order", "other"),
            status=KeyStatus.APPROVED,
        )
    )
    store.upsert_key(
        MetaKey(
            doc_type="court_decision",
            key="mystery",
            value_type=ValueType.NUMBER,
            description="A proposed key.",
        )
    )

    approved = store.list_keys("court_decision", KeyStatus.APPROVED)
    everything = store.list_keys("court_decision")

    assert [k.key for k in approved] == ["document_kind"]
    assert approved[0].allowed_values == ("judgment", "order", "other")
    assert [k.key for k in everything] == ["document_kind", "mystery"]
    assert everything[1].status is KeyStatus.PROPOSED
    assert store.list_keys("invoice") == []


def test_upserting_a_key_replaces_its_definition(db_conn):
    store = DocumentStore()
    key = MetaKey("invoice", "total", ValueType.NUMBER, "Total amount.", version=1)
    store.upsert_key(key)
    store.upsert_key(
        MetaKey("invoice", "total", ValueType.NUMBER, "Gross total.", version=2)
    )

    [stored] = store.list_keys("invoice")

    assert (stored.description, stored.version) == ("Gross total.", 2)


def test_typed_values_round_trip(db_conn):
    store = DocumentStore()
    store.upsert_document(Document(HASH_A, "a.docx"))
    store.add_value(
        MetaValue(
            content_hash=HASH_A,
            key="decision_date",
            key_version=1,
            source=MetaSource.DETERMINISTIC,
            value_date=date(2024, 12, 16),
            evidence="Budapest, 2024. december 16.",
            evidence_chunk_index=7,
            page=4,
        )
    )
    store.add_value(
        MetaValue(
            content_hash=HASH_A,
            key="legal_costs_awarded",
            key_version=1,
            source=MetaSource.LLM,
            value_number=Decimal(629920),
            unit="HUF",
            ordinal=1,
            qualifiers={"payer": "plaintiff", "instance": "first"},
        )
    )

    [d] = store.get_values(HASH_A, "decision_date")
    [m] = store.get_values(HASH_A, "legal_costs_awarded")

    assert d.value_date == date(2024, 12, 16)
    assert (d.evidence_chunk_index, d.page, d.source) == (
        7,
        4,
        MetaSource.DETERMINISTIC,
    )
    assert m.value_number == Decimal(629920)
    assert (m.unit, m.ordinal) == ("HUF", 1)
    assert m.qualifiers == {"payer": "plaintiff", "instance": "first"}


def test_a_value_row_must_carry_exactly_one_value(db_conn):
    store = DocumentStore()
    store.upsert_document(Document(HASH_A, "a.docx"))

    with pytest.raises(psycopg2.errors.CheckViolation):
        store.add_value(MetaValue(HASH_A, "court", 1, MetaSource.LLM))  # none set
    with pytest.raises(psycopg2.errors.CheckViolation):
        store.add_value(
            MetaValue(
                HASH_A, "court", 1, MetaSource.LLM, value_text="x", value_bool=True
            )
        )


def test_a_value_needs_a_registered_document_and_a_unique_ordinal(db_conn):
    store = DocumentStore()
    with pytest.raises(psycopg2.errors.ForeignKeyViolation):
        store.add_value(MetaValue("c" * 64, "court", 1, MetaSource.LLM, value_text="x"))

    store.upsert_document(Document(HASH_A, "a.docx"))
    value = MetaValue(HASH_A, "court", 1, MetaSource.LLM, value_text="x")
    store.add_value(value)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        store.add_value(value)


def test_status_defaults_to_not_attempted_and_can_be_replaced(db_conn):
    store = DocumentStore()
    store.upsert_document(Document(HASH_A, "a.docx"))

    assert store.get_status(HASH_A, "court").state is MetaState.NOT_ATTEMPTED

    store.set_status(MetaStatus(HASH_A, "court", MetaState.UNVERIFIED, key_version=1))
    store.set_status(MetaStatus(HASH_A, "court", MetaState.PRESENT, key_version=2))

    status = store.get_status(HASH_A, "court")
    assert (status.state, status.key_version) == (MetaState.PRESENT, 2)


def test_deleting_a_document_cascades_to_values_and_status(db_conn):
    store = DocumentStore()
    store.upsert_document(Document(HASH_A, "a.docx"))
    store.add_value(MetaValue(HASH_A, "court", 1, MetaSource.LLM, value_text="x"))
    store.set_status(MetaStatus(HASH_A, "court", MetaState.PRESENT, key_version=1))

    assert store.delete_document(HASH_A) == 1

    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM document_meta;")
        assert cur.fetchone() == (0,)
        cur.execute("SELECT count(*) FROM document_meta_status;")
        assert cur.fetchone() == (0,)


def test_the_store_can_share_one_connection_across_calls(db_conn):
    with DocumentStore() as store:
        store.upsert_document(Document(HASH_A, "a.docx"))
        store.upsert_document(Document(HASH_B, "b.docx"))
        assert store.count_documents() == 2
