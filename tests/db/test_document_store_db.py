"""DB-gated tests for DocumentStore and the structured-metadata schema (migration 0004).

Requires AGENT_ENV=test (and a .env.test) -- see tests/db/conftest.py and
config.py. Chunks are written with the real VectorStore.save() using the
embedding dimension the schema declares, so the generated ``content_hash``
column is exercised exactly as ingestion would populate it.
"""

from datetime import date
from decimal import Decimal

import psycopg2
import psycopg2.errors
import pytest

from config import settings
from document_store import DocumentStore
from models import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentType,
    KeyStatus,
    MetaKey,
    MetaSource,
    MetaState,
    MetaStatus,
    MetaValue,
    TypeStatus,
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
    store.ensure_type("court_decision")
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
    store.ensure_type("invoice")
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


def test_importing_a_catalog_adds_then_leaves_an_unchanged_one_alone(db_conn):
    from metadata.catalog import KeyCatalog

    keys = [
        MetaKey(
            "invoice",
            "total",
            ValueType.NUMBER,
            "Gross total.",
            status=KeyStatus.APPROVED,
        ),
        MetaKey(
            "invoice",
            "currency",
            ValueType.TEXT,
            "Currency code.",
            status=KeyStatus.APPROVED,
        ),
    ]
    catalog = KeyCatalog(DocumentStore())

    first = catalog.import_seed(keys)
    second = catalog.import_seed(keys)

    assert (first.added, first.revised, first.unchanged) == (2, 0, 0)
    assert (second.added, second.revised, second.unchanged) == (0, 0, 2)
    assert {k.version for k in DocumentStore().list_keys("invoice")} == {1}


def test_changing_a_keys_description_bumps_its_version(db_conn):
    """So values extracted under the old definition can be recognised as stale."""
    from metadata.catalog import KeyCatalog

    catalog = KeyCatalog(DocumentStore())
    catalog.import_seed(
        [
            MetaKey(
                "invoice",
                "total",
                ValueType.NUMBER,
                "Gross total.",
                status=KeyStatus.APPROVED,
            )
        ]
    )

    result = catalog.import_seed(
        [
            MetaKey(
                "invoice",
                "total",
                ValueType.NUMBER,
                "Net total.",
                status=KeyStatus.APPROVED,
            )
        ]
    )

    [stored] = DocumentStore().list_keys("invoice")
    assert (result.revised, stored.description, stored.version) == (1, "Net total.", 2)


def _key(name, version=1):
    return MetaKey(
        "court_decision",
        name,
        ValueType.TEXT,
        "d",
        status=KeyStatus.APPROVED,
        version=version,
    )


def test_replace_values_swaps_the_rows_of_one_key_only(db_conn):
    store = DocumentStore()
    store.upsert_document(Document(HASH_A, "a.docx"))
    store.add_value(MetaValue(HASH_A, "court", 1, MetaSource.LLM, value_text="old"))
    store.add_value(MetaValue(HASH_A, "kind", 1, MetaSource.LLM, value_text="keep"))

    store.replace_values(
        HASH_A,
        "court",
        [
            MetaValue(
                HASH_A, "court", 2, MetaSource.LLM, value_text="new 0", ordinal=0
            ),
            MetaValue(
                HASH_A, "court", 2, MetaSource.LLM, value_text="new 1", ordinal=1
            ),
        ],
    )

    assert [v.value_text for v in store.get_values(HASH_A, "court")] == [
        "new 0",
        "new 1",
    ]
    assert [v.value_text for v in store.get_values(HASH_A, "kind")] == ["keep"]
    store.replace_values(HASH_A, "court", [])  # clearing is allowed
    assert store.get_values(HASH_A, "court") == []


def test_documents_needing_extraction_follow_the_status_and_the_key_version(db_conn):
    store = DocumentStore()
    store.upsert_document(Document(HASH_A, "a.docx"))
    store.upsert_document(Document(HASH_B, "b.docx"))
    keys = [_key("court"), _key("kind")]

    # nothing attempted: both documents need work
    assert [d.source_file for d in store.documents_needing(keys)] == [
        "a.docx",
        "b.docx",
    ]

    # A is done for both keys, B only for one
    for key in ("court", "kind"):
        store.set_status(MetaStatus(HASH_A, key, MetaState.PRESENT, key_version=1))
    store.set_status(
        MetaStatus(HASH_B, "court", MetaState.CONFIRMED_ABSENT, key_version=1)
    )
    assert [d.source_file for d in store.documents_needing(keys)] == ["b.docx"]

    # a key whose definition changed (version 2) makes every document pending again
    bumped = [_key("court", version=2), _key("kind")]
    assert [d.source_file for d in store.documents_needing(bumped)] == [
        "a.docx",
        "b.docx",
    ]

    assert [d.source_file for d in store.documents_needing(keys, limit=1)] == ["b.docx"]
    assert store.documents_needing([]) == []


def test_statuses_of_a_document_are_returned_by_key(db_conn):
    store = DocumentStore()
    store.upsert_document(Document(HASH_A, "a.docx"))
    store.set_status(MetaStatus(HASH_A, "court", MetaState.PRESENT, key_version=3))

    statuses = store.get_statuses(HASH_A)

    assert set(statuses) == {"court"}
    assert (statuses["court"].state, statuses["court"].key_version) == (
        MetaState.PRESENT,
        3,
    )


def test_coverage_counts_every_document_in_exactly_one_state(db_conn):
    store = DocumentStore()
    for h, f in (
        (HASH_A, "a.docx"),
        (HASH_B, "b.docx"),
        ("c" * 64, "c.docx"),
        ("d" * 64, "d.docx"),
    ):
        store.upsert_document(Document(h, f))
    store.set_status(MetaStatus(HASH_A, "court", MetaState.PRESENT, key_version=1))
    store.set_status(MetaStatus(HASH_B, "court", MetaState.UNVERIFIED, key_version=1))
    store.set_status(
        MetaStatus("c" * 64, "court", MetaState.CONFIRMED_ABSENT, key_version=1)
    )
    # d: never attempted

    [cov] = store.coverage([_key("court")])

    assert (cov.present, cov.unverified, cov.confirmed_absent, cov.not_attempted) == (
        1,
        1,
        1,
        1,
    )
    assert cov.total == 4 and cov.unknown == 2


def test_a_status_from_an_older_definition_counts_as_not_attempted(db_conn):
    store = DocumentStore()
    store.upsert_document(Document(HASH_A, "a.docx"))
    store.set_status(MetaStatus(HASH_A, "court", MetaState.PRESENT, key_version=1))

    [cov] = store.coverage([_key("court", version=2)])

    assert (cov.present, cov.not_attempted) == (0, 1)


def test_a_documents_chunks_come_back_in_document_order(db_conn):
    _save_chunks(HASH_A, "a.docx", "summary", n=3)
    _save_chunks(HASH_B, "b.docx", "summary", n=1)

    chunks = VectorStore().get_document_chunks(HASH_A)

    assert [c.metadata.chunk_index for c in chunks] == [0, 1, 2]
    assert {c.metadata.source_file for c in chunks} == {"a.docx"}
    assert VectorStore().get_document_chunks("f" * 64) == []


def test_the_runner_extracts_end_to_end_with_the_real_stores_and_a_scripted_source(
    db_conn,
):
    """The whole path on a real database: documents, chunks, status, values, resumability."""
    from unittest.mock import MagicMock

    from metadata.catalog import KeyCatalog
    from metadata.evidence import EvidenceSelector
    from metadata.runner import MetaExtractionRunner
    from metadata.sources import Candidate, ChunkMetadataSource, SourceResult
    from metadata.sources import MetaSource as SourceBase

    class ScriptedCourtSource(SourceBase):
        kind = MetaSource.LLM
        needs_verification = True

        def supports(self, key):
            return key.key == "issuing_body"

        def extract(self, chunks, keys):
            quote = "Egri Törvényszék"
            return SourceResult(
                candidates=[
                    Candidate(
                        "issuing_body", quote, evidence=quote, evidence_chunk_index=0
                    )
                ]
            )

    _save_chunks(HASH_A, "a.docx", "summary", n=2)
    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE document_chunks SET content = 'Az Egri Törvényszék ítélete. Eger, 2023. május 4.',"
            " metadata = jsonb_set(metadata, '{document_date}', '\"2023-05-04\"');"
        )
    db_conn.commit()
    DocumentStore().sync_from_chunks()
    KeyCatalog(DocumentStore()).import_seed(
        [
            MetaKey(
                "court_decision",
                "issuing_body",
                ValueType.TEXT,
                "The court.",
                status=KeyStatus.APPROVED,
            ),
            MetaKey(
                "court_decision",
                "decision_date",
                ValueType.DATE,
                "Date.",
                status=KeyStatus.APPROVED,
            ),
        ]
    )
    runner = MetaExtractionRunner(
        DocumentStore(),
        VectorStore(),
        EvidenceSelector(MagicMock()),
        [
            ChunkMetadataSource({"decision_date": "document_date"}),
            ScriptedCourtSource(),
        ],
    )

    first = runner.run("court_decision")
    second = runner.run("court_decision")

    assert (first.documents, first.present) == (1, 2)
    assert second.documents == 0  # resumable: nothing left to do
    store = DocumentStore()
    [court] = store.get_values(HASH_A, "issuing_body")
    [when] = store.get_values(HASH_A, "decision_date")
    assert (court.value_text, court.source) == ("Egri Törvényszék", MetaSource.LLM)
    assert court.evidence_chunk_index == 0
    assert (when.value_date, when.source) == (
        date(2023, 5, 4),
        MetaSource.DETERMINISTIC,
    )
    by_key = {c.key: c for c in store.coverage(store.list_keys("court_decision"))}
    assert (by_key["issuing_body"].present, by_key["decision_date"].present) == (1, 1)


def test_list_values_returns_every_value_of_a_key_with_its_file_name(db_conn):
    store = DocumentStore()
    store.upsert_document(Document(HASH_A, "a.docx"))
    store.upsert_document(Document(HASH_B, "b.docx"))
    store.add_value(MetaValue(HASH_B, "court", 1, MetaSource.LLM, value_text="B court"))
    store.add_value(MetaValue(HASH_A, "court", 1, MetaSource.LLM, value_text="A court"))
    store.add_value(
        MetaValue(HASH_A, "kind", 1, MetaSource.LLM, value_text="other key")
    )

    rows = store.list_values("court")

    assert [(f, v.value_text) for f, v in rows] == [
        ("a.docx", "A court"),
        ("b.docx", "B court"),
    ]


def test_a_seeded_sample_is_random_across_files_yet_reproducible(db_conn):
    store = DocumentStore()
    for i in range(12):
        store.upsert_document(
            Document(f"{i:064x}", f"court_{i // 4}__doc_{i:02d}.docx")
        )
    keys = [_key("court")]

    by_name = [d.source_file for d in store.documents_needing(keys, limit=4)]
    seeded = [d.source_file for d in store.documents_needing(keys, limit=4, seed=7)]
    again = [d.source_file for d in store.documents_needing(keys, limit=4, seed=7)]
    other_seed = [d.source_file for d in store.documents_needing(keys, limit=4, seed=8)]

    assert by_name == sorted(by_name) and len({f.split("__")[0] for f in by_name}) == 1
    assert seeded == again  # reproducible
    assert seeded != other_seed  # a different seed, a different sample
    assert len({f.split("__")[0] for f in seeded}) > 1  # spread over the courts


def test_a_proposed_key_can_be_approved_or_retired(db_conn):
    store = DocumentStore()
    store.ensure_type("court_decision")
    store.upsert_key(
        MetaKey("court_decision", "date_of_issue", ValueType.DATE, "A duplicate.")
    )

    assert store.set_key_status("court_decision", "date_of_issue", KeyStatus.APPROVED)
    assert [k.key for k in store.list_keys("court_decision", KeyStatus.APPROVED)] == [
        "date_of_issue"
    ]
    assert store.set_key_status("court_decision", "date_of_issue", KeyStatus.RETIRED)
    assert store.list_keys("court_decision", KeyStatus.APPROVED) == []
    assert not store.set_key_status("court_decision", "no_such_key", KeyStatus.APPROVED)


def test_a_key_needs_a_registered_document_type(db_conn):
    store = DocumentStore()

    with pytest.raises(ValueError, match="unknown document type 'invoice'"):
        store.upsert_key(MetaKey("invoice", "total", ValueType.NUMBER, "Total."))

    store.upsert_type(
        DocumentType("invoice", "Invoice", "A bill for goods.", TypeStatus.APPROVED)
    )
    store.upsert_key(
        MetaKey("invoice", "total", ValueType.NUMBER, "Total.")
    )  # now fine


def test_types_round_trip_and_their_status_can_change(db_conn):
    store = DocumentStore()
    store.upsert_type(
        DocumentType("invoice", "Invoice", "A bill.", TypeStatus.PROPOSED)
    )
    store.ensure_type("court_decision")

    assert store.get_type("invoice") == DocumentType(
        "invoice", "Invoice", "A bill.", TypeStatus.PROPOSED
    )
    assert [t.type for t in store.list_types()] == ["court_decision", "invoice"]
    assert [t.type for t in store.list_types(TypeStatus.APPROVED)] == ["court_decision"]
    assert store.set_type_status("invoice", TypeStatus.APPROVED) is True
    assert store.set_type_status("missing", TypeStatus.APPROVED) is False
    assert store.get_type("invoice").status is TypeStatus.APPROVED  # type: ignore[union-attr]
    assert store.get_type("nothing") is None


def test_ensure_type_leaves_an_existing_type_alone(db_conn):
    store = DocumentStore()
    store.upsert_type(
        DocumentType("invoice", "Invoice", "A bill.", TypeStatus.PROPOSED)
    )

    store.ensure_type("invoice")

    assert store.get_type("invoice") == DocumentType(
        "invoice", "Invoice", "A bill.", TypeStatus.PROPOSED
    )


def test_deleting_a_document_removes_its_chunks_values_and_status_by_cascade(db_conn):
    """The numeric links carry ON DELETE CASCADE: no orphans after a delete."""
    with db_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (content_hash, source_file) VALUES ('h-cascade', 'c.docx') RETURNING id;"
        )
        doc_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO document_chunks (content, metadata, embedding, document_id) "
            "VALUES ('x', '{}'::jsonb, %s, %s);",
            ("[" + ",".join(["0"] * settings.EMBEDDING_DIMENSION) + "]", doc_id),
        )
    db_conn.commit()

    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE id = %s;", (doc_id,))
        cur.execute(
            "SELECT count(*) FROM document_chunks WHERE document_id = %s;", (doc_id,)
        )
        assert cur.fetchone()[0] == 0
    db_conn.commit()


def test_upsert_document_returns_its_numeric_id_and_keeps_it_on_a_refresh(db_conn):
    store = DocumentStore()

    first = store.upsert_document(Document(HASH_A, "a.docx", "summary"))
    again = store.upsert_document(Document(HASH_A, "a.docx", "new summary"))
    other = store.upsert_document(Document(HASH_B, "b.docx", None))

    assert isinstance(first, int)
    assert again == first and other != first


def test_values_and_statuses_are_stored_with_the_numeric_document_id(db_conn):
    store = DocumentStore()
    doc_id = store.upsert_document(Document(HASH_A, "a.docx", None))
    assert isinstance(doc_id, int)  # else a NULL column would "match" None
    store.ensure_type("court_decision")
    store.upsert_key(MetaKey("court_decision", "court", ValueType.TEXT, "The court."))

    store.replace_values(
        HASH_A,
        "court",
        [
            MetaValue(
                HASH_A,
                "court",
                1,
                MetaSource.LLM,
                value_text="Kúria",
                evidence="Kúria",
            )
        ],
    )
    store.set_status(MetaStatus(HASH_A, "court", MetaState.PRESENT, key_version=1))

    with db_conn.cursor() as cur:
        cur.execute("SELECT document_id FROM document_meta WHERE key = 'court';")
        assert cur.fetchall() == [(doc_id,)]
        cur.execute("SELECT document_id FROM document_meta_status WHERE key = 'court';")
        assert cur.fetchall() == [(doc_id,)]


def test_a_value_for_an_unregistered_document_still_fails_loudly(db_conn):
    store = DocumentStore()
    store.ensure_type("court_decision")
    store.upsert_key(MetaKey("court_decision", "court", ValueType.TEXT, "The court."))

    with pytest.raises(psycopg2.errors.ForeignKeyViolation):
        store.add_value(
            MetaValue(
                "f" * 64, "court", 1, MetaSource.LLM, value_text="x", evidence="x"
            )
        )


def test_sync_links_chunks_that_were_saved_without_a_document_id(db_conn):
    """What the ingest leaves behind until it sets the id itself."""
    _save_chunks(HASH_A, "a.docx", None, n=3)

    result = DocumentStore().sync_from_chunks()

    assert result.linked == 3
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM document_chunks c JOIN documents d "
            "ON d.id = c.document_id AND d.content_hash = c.content_hash;"
        )
        assert cur.fetchone() == (3,)
    assert DocumentStore().sync_from_chunks().linked == 0  # nothing left to link


def test_importing_a_catalog_writes_its_types_then_its_keys_and_is_idempotent(db_conn):
    from metadata.catalog import Catalog, KeyCatalog

    catalog = Catalog(
        types=[
            DocumentType("invoice", "Invoice", "A bill.", TypeStatus.APPROVED),
            DocumentType("contract", "Contract", "An agreement.", TypeStatus.APPROVED),
        ],
        keys=[
            MetaKey(
                "invoice",
                "total",
                ValueType.NUMBER,
                "Total.",
                status=KeyStatus.APPROVED,
            ),
            MetaKey(
                "contract",
                "total",
                ValueType.NUMBER,
                "Value.",
                status=KeyStatus.APPROVED,
            ),
        ],
    )
    importer = KeyCatalog(DocumentStore())

    first = importer.import_catalog(catalog)
    second = importer.import_catalog(catalog)

    assert (first.types_added, first.types_updated, first.added) == (2, 0, 2)
    assert (second.types_added, second.types_updated, second.unchanged) == (0, 0, 2)
    store = DocumentStore()
    assert store.get_type("contract") == catalog.types[1]
    assert (
        len(store.list_keys("invoice")) == 1 and len(store.list_keys("contract")) == 1
    )


def test_a_changed_type_description_is_updated_by_the_next_import(db_conn):
    from metadata.catalog import Catalog, KeyCatalog

    importer = KeyCatalog(DocumentStore())
    old = DocumentType("invoice", "Invoice", "A bill.", TypeStatus.APPROVED)
    importer.import_catalog(Catalog(types=[old], keys=[]))

    result = importer.import_catalog(
        Catalog(
            types=[
                DocumentType(
                    "invoice", "Invoice", "A bill for goods.", TypeStatus.APPROVED
                )
            ],
            keys=[],
        )
    )

    assert (result.types_added, result.types_updated) == (0, 1)
    assert DocumentStore().get_type("invoice").description == "A bill for goods."  # type: ignore[union-attr]
