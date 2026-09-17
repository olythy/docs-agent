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


def test_add_document_end_to_end_with_real_pdf(db_conn):
    """Runs the full extract -> chunk -> embed -> store pipeline for real."""
    if not settings.TEST_PDF_PATH or not Path(settings.TEST_PDF_PATH).exists():
        pytest.skip("TEST_PDF_PATH is not set or the file doesn't exist")

    add_document(settings.TEST_PDF_PATH)

    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM document_chunks;")
        count = cur.fetchone()[0]

    assert count > 0
