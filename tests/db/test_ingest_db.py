"""DB-gated tests for VectorStore.save() and add_document() end-to-end.

Requires AGENT_ENV=test (and a .env.test) — see tests/db/conftest.py and
config.py. Uses the real local embedding driver (already cached locally,
no network call) against a real Postgres+pgvector instance.
"""

from pathlib import Path

import pytest

from config import settings
from document_store import DocumentStore
from drivers.embedding import get_embedding_driver
from ingestion.ingest import add_directory, add_document
from models import Chunk, ChunkMetadata, Document
from store import VectorStore

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(
        settings.AGENT_ENV != "test",
        reason=(
            "AGENT_ENV is not 'test' — no test database is configured. "
            "These tests are written and ready; run with AGENT_ENV=test "
            "(e.g. `make test`, which sets this automatically) and a "
            ".env.test file (see .env.test.example) to run them."
        ),
    ),
]


def test_save_inserts_rows_that_are_readable_back(db_conn):
    chunks = [
        Chunk(
            content="hello world",
            metadata=ChunkMetadata(source_file="t.pdf", page_number=1, chunk_index=0),
        ),
        Chunk(
            content="second chunk",
            metadata=ChunkMetadata(source_file="t.pdf", page_number=2, chunk_index=1),
        ),
    ]
    embeddings = [[0.1] * 384, [0.2] * 384]

    document_id = DocumentStore().upsert_document(Document("h-save", "t.pdf"))

    inserted = VectorStore().save(chunks, embeddings, document_id=document_id)
    assert inserted == 2

    with db_conn.cursor() as cur:
        cur.execute("SELECT content, metadata FROM document_chunks ORDER BY id;")
        rows = cur.fetchall()

    assert [r[0] for r in rows] == ["hello world", "second chunk"]
    assert rows[0][1] == {
        "source_file": "t.pdf",
        "page_number": 1,
        "chunk_index": 0,
    }


def test_add_document_end_to_end_with_real_document(db_conn):
    """Runs the full extract -> chunk -> embed -> store pipeline for real.

    Uses settings.TEST_DOC_PATH, whatever format it points at (PDF or
    Markdown — add_document() dispatches on the extension).
    """
    if not settings.TEST_DOC_PATH or not Path(settings.TEST_DOC_PATH).exists():
        pytest.skip("TEST_DOC_PATH is not set or the file doesn't exist")

    add_document(settings.TEST_DOC_PATH)

    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        count = cur.fetchone()[0]

    assert count > 0


def test_add_document_end_to_end_with_real_markdown(db_conn):
    """Runs the full pipeline for a Markdown file — the MarkdownExtractor path.

    Uses a small, hand-written, deliberately committed Markdown fixture —
    not settings.TEST_DOC_PATH, which is never committed (PDF or Markdown,
    it could be a real personal/client document) — so this proves the
    MarkdownExtractor branch specifically, regardless of what the user's
    own TEST_DOC_PATH happens to point at.
    """
    add_document("tests/data/sample.md")

    with db_conn.cursor() as cur:
        cur.execute("SELECT metadata FROM document_chunks ORDER BY id;")
        rows = cur.fetchall()

    assert len(rows) > 0
    assert all(r[0]["source_file"] == "sample.md" for r in rows)
    # sample.md has multiple headers, so its header-based section index is
    # meaningfully > 1 somewhere (proves the MarkdownExtractor's section
    # counting ran, not just a hardcoded "1" — exact chunk count/boundaries
    # depend on CHUNK_SIZE, which _markdown_section_map's own unit tests
    # already cover in detail, so this only checks the pipeline wiring).
    assert max(r[0]["page_number"] for r in rows) > 1


def test_delete_chunks_from_source_removes_only_target_file(db_conn):
    """Proves VectorStore.delete_chunks_from_source deletes rows for that file only."""
    driver = get_embedding_driver()
    chunks = [
        Chunk(
            content="c1",
            metadata=ChunkMetadata(
                source_file="file_a.pdf", page_number=None, chunk_index=0
            ),
        ),
        Chunk(
            content="c2",
            metadata=ChunkMetadata(
                source_file="file_b.pdf", page_number=None, chunk_index=0
            ),
        ),
    ]
    embeddings = [driver.embed_text("c1"), driver.embed_text("c2")]
    documents = DocumentStore()
    store = VectorStore()
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        document_id = documents.upsert_document(
            Document(f"h-{chunk.metadata.source_file}", chunk.metadata.source_file)
        )
        store.save([chunk], [embedding], document_id=document_id)

    deleted = VectorStore().delete_chunks_from_source("file_a.pdf")
    assert deleted == 1

    with db_conn.cursor() as cur:
        cur.execute("SELECT metadata->>'source_file' FROM document_chunks;")
        remaining = [r[0] for r in cur.fetchall()]

    assert remaining == ["file_b.pdf"]


def test_add_document_force_replaces_existing_chunks_without_duplicates(db_conn):
    """Proves add_document(force=True) replaces existing chunks instead of duplicating."""
    add_document("tests/data/sample.md")

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE metadata->>'source_file' = 'sample.md';"
        )
        initial_count = cur.fetchone()[0]

    assert initial_count > 0

    # Re-ingest with force=True — must cleanly replace, keeping identical count
    add_document("tests/data/sample.md", force=True)

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM document_chunks WHERE metadata->>'source_file' = 'sample.md';"
        )
        reingested_count = cur.fetchone()[0]

    assert reingested_count == initial_count


def test_add_document_end_to_end_with_hungarian_markdown(db_conn):
    """Proves ingestion of the new Hungarian IT policy fixture with header enrichment."""
    add_document("tests/data/sample_hu.md")

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT content, metadata FROM document_chunks WHERE metadata->>'source_file' = 'sample_hu.md';"
        )
        rows = cur.fetchall()

    assert len(rows) > 0
    # Verify header enrichment was stored in metadata and content
    assert any("header_path" in r[1] for r in rows)
    assert any(r[1].get("header_path") is not None for r in rows)


def test_add_directory_end_to_end_real_db(tmp_path, db_conn):
    """Proves add_directory batch-ingests multiple files into real Postgres."""
    folder = tmp_path / "batch_docs"
    folder.mkdir()
    (folder / "file1.md").write_text("# File 1\nSome initial content.")
    (folder / "file2.md").write_text("# File 2\nSome second content.")
    (folder / "ignored.txt").write_text("Should be ignored.")

    summary = add_directory(folder)

    assert summary["total_found"] == 2
    assert len(summary["ingested"]) == 2
    assert len(summary["skipped"]) == 0
    assert len(summary["failed"]) == 0

    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT metadata->>'source_file' FROM document_chunks ORDER BY 1;"
        )
        sources = [r[0] for r in cur.fetchall()]

    assert sources == ["file1.md", "file2.md"]


def test_add_directory_aliases_identical_files_in_real_db(tmp_path, db_conn):
    """Proves that identical files under different paths are aliased without duplicate chunks."""
    folder = tmp_path / "alias_docs"
    folder.mkdir()
    (folder / "original.md").write_text("# Shared Knowledge\nIdentical body text.")
    sub = folder / "archive"
    sub.mkdir()
    (sub / "copy.md").write_text("# Shared Knowledge\nIdentical body text.")

    summary = add_directory(folder)

    assert summary["total_found"] == 2
    assert len(summary["ingested"]) == 1
    assert len(summary["aliased"]) == 1

    with db_conn.cursor() as cur:
        cur.execute("SELECT metadata FROM document_chunks;")
        rows = cur.fetchall()

    # Chunks are stored only once
    assert len(rows) > 0
    metadata = rows[0][0]
    # sources array includes both file paths
    assert "original.md" in metadata["sources"]
    assert "archive/copy.md" in metadata["sources"]


def test_delete_chunks_by_hash_in_real_db(db_conn):
    """Proves VectorStore.delete_chunks_by_hash atomically deletes only target chunks."""
    from ingestion.hash import compute_file_hash

    add_document("tests/data/sample.md")
    content_hash = compute_file_hash("tests/data/sample.md")

    store = VectorStore()
    assert store.has_content_hash(content_hash) is True

    deleted = store.delete_chunks_by_hash(content_hash)
    assert deleted > 0
    assert store.has_content_hash(content_hash) is False


def _extracted_value_count(db_conn) -> int:
    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM document_meta;")
        return cur.fetchone()[0]


def _give_the_document_an_extracted_value(db_conn, source_file: str) -> None:
    """Store one extracted value for a just-ingested document, as extract-meta would."""
    from document_store import DocumentStore
    from models import MetaKey, MetaSource, MetaValue, ValueType

    store = DocumentStore()
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT content_hash FROM documents WHERE source_file = %s;", (source_file,)
        )
        content_hash = cur.fetchone()[0]
    store.ensure_type("note")
    store.upsert_key(MetaKey("note", "topic", ValueType.TEXT, "The topic."))
    store.replace_values(
        content_hash,
        "topic",
        [
            MetaValue(
                content_hash, "topic", 1, MetaSource.LLM, value_text="x", evidence="x"
            )
        ],
    )


def test_ingest_registers_the_document_and_links_every_chunk_to_its_id(db_conn):
    add_document("tests/data/sample.md")

    with db_conn.cursor() as cur:
        cur.execute("SELECT id, content_hash, source_file FROM documents;")
        (doc_id, content_hash, source_file) = cur.fetchone()
        cur.execute(
            "SELECT count(*), count(document_id), count(*) FILTER (WHERE document_id = %s) "
            "FROM document_chunks;",
            (doc_id,),
        )
        total, linked, to_this_document = cur.fetchone()

    assert source_file == "sample.md" and len(content_hash) == 64
    assert total > 0 and total == linked == to_this_document


def test_forcing_a_reindex_keeps_the_document_id_and_its_extracted_values(db_conn):
    add_document("tests/data/sample.md")
    _give_the_document_an_extracted_value(db_conn, "sample.md")
    with db_conn.cursor() as cur:
        cur.execute("SELECT id FROM documents;")
        first_id = cur.fetchone()[0]

    add_document("tests/data/sample.md", force=True)

    with db_conn.cursor() as cur:
        cur.execute("SELECT id FROM documents;")
        assert cur.fetchall() == [(first_id,)]
        cur.execute("SELECT count(*) FROM document_chunks WHERE document_id IS NULL;")
        assert cur.fetchone()[0] == 0
    assert _extracted_value_count(db_conn) == 1  # the paid-for value survived


def test_a_new_version_of_a_document_drops_the_old_versions_extracted_values(
    tmp_path, db_conn
):
    doc = tmp_path / "evolving.md"
    doc.write_text("# Notes\nThe first version of this note.")
    add_document(doc)
    _give_the_document_an_extracted_value(db_conn, "evolving.md")
    assert _extracted_value_count(db_conn) == 1

    doc.write_text("# Notes\nA completely different second version.")
    add_document(doc)

    with db_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents;")
        assert cur.fetchone()[0] == 1  # the old version's row is gone
    assert _extracted_value_count(db_conn) == 0


def test_deleting_by_hash_cli_path_removes_the_metadata_too(db_conn):
    from document_store import DocumentStore

    add_document("tests/data/sample.md")
    _give_the_document_an_extracted_value(db_conn, "sample.md")
    with db_conn.cursor() as cur:
        cur.execute("SELECT content_hash FROM documents;")
        content_hash = cur.fetchone()[0]

    VectorStore().delete_chunks_by_hash(content_hash)
    assert DocumentStore().delete_document(content_hash) == 1

    assert _extracted_value_count(db_conn) == 0
