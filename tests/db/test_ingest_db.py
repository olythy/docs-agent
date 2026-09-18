"""DB-gated tests for VectorStore.save() and add_document() end-to-end.

Requires AGENT_ENV=test (and a .env.test) — see tests/db/conftest.py and
config.py. Uses the real local embedding driver (already cached locally,
no network call) against a real Postgres+pgvector instance.
"""

from pathlib import Path

import pytest

from config import settings
from ingestion.ingest import add_document
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
        {
            "content": "hello world",
            "metadata": {"source_file": "t.pdf", "page_number": 1},
        },
        {
            "content": "second chunk",
            "metadata": {"source_file": "t.pdf", "page_number": 2},
        },
    ]
    embeddings = [[0.1] * 384, [0.2] * 384]

    inserted = VectorStore().save(chunks, embeddings)
    assert inserted == 2

    with db_conn.cursor() as cur:
        cur.execute("SELECT content, metadata FROM document_chunks ORDER BY id;")
        rows = cur.fetchall()

    assert [r[0] for r in rows] == ["hello world", "second chunk"]
    assert rows[0][1] == {"source_file": "t.pdf", "page_number": 1}


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
