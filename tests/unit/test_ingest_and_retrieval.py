"""Tests for ingestion.ingest orchestration.

Only the early-exit validation paths are covered here for now — both raise
before any database connection is attempted, so no mocking of psycopg2 is
needed. Full add_document/query_knowledge_base coverage (embedding + DB
interaction mocked) is added in later steps of the fix plan.
"""

from unittest.mock import patch

import pytest

from ingestion.ingest import add_document


def test_add_document_raises_on_empty_pages(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"")

    with (
        patch("ingestion.ingest.extract_pages", return_value=[]),
        pytest.raises(ValueError, match="has no pages"),
    ):
        add_document(pdf_path)


def test_add_document_raises_on_scanned_pdf(tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"")

    scanned_pages = [{"page_number": 1, "text": "", "char_count": 0}]
    with (
        patch("ingestion.ingest.extract_pages", return_value=scanned_pages),
        pytest.raises(ValueError, match="scanned PDF"),
    ):
        add_document(pdf_path)
