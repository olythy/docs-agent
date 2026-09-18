"""Tests for ingestion.ingest orchestration.

Only the early-exit validation path is covered here — it raises before any
embedding driver or database work is attempted. The specific validation
cases (empty PDF, scanned PDF) are covered per-extractor in
tests/unit/test_extractors.py; this only confirms add_document() actually
wires extractor.validate() into the pipeline. Full add_document/
query_knowledge_base coverage (embedding + DB interaction mocked) is added
in later steps of the fix plan.
"""

from unittest.mock import MagicMock, patch

import pytest

from ingestion.ingest import add_document


def test_add_document_propagates_extractor_validation_error(tmp_path):
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"")

    fake_extractor = MagicMock()
    fake_extractor.validate.side_effect = ValueError("no usable content")

    with (
        patch("ingestion.ingest.get_extractor", return_value=fake_extractor),
        pytest.raises(ValueError, match="no usable content"),
    ):
        add_document(pdf_path)

    fake_extractor.validate.assert_called_once_with(pdf_path)


def _mock_ingest_pipeline(monkeypatch, *, already_present: bool):
    """Patch every ingestion.ingest dependency for a full mocked run.

    Returns the fake VectorStore so callers can assert on it.
    """
    fake_extractor = MagicMock()
    fake_extractor.extract.return_value = ("full text here", [1, 1, 1])
    monkeypatch.setattr("ingestion.ingest.get_extractor", lambda p: fake_extractor)
    monkeypatch.setattr("ingestion.ingest.get_embedding_driver", lambda: MagicMock())
    monkeypatch.setattr(
        "ingestion.ingest.chunk_document",
        lambda *a, **kw: [{"content": "x", "metadata": {}}],
    )
    fake_overflow_strategy = MagicMock()
    fake_overflow_strategy.apply.side_effect = lambda chunks, driver: chunks
    monkeypatch.setattr(
        "ingestion.ingest.get_chunk_overflow_strategy", lambda: fake_overflow_strategy
    )

    fake_store = MagicMock()
    fake_store.has_chunks_from_source.return_value = already_present
    fake_store.save.return_value = 1
    # MagicMock treats "assert_*" names as typo-guards by default (raises
    # AttributeError), not real attributes — assign explicitly since
    # VectorStore genuinely has a method with this name.
    fake_store.assert_dimension_matches = MagicMock()
    monkeypatch.setattr("ingestion.ingest.VectorStore", lambda: fake_store)

    return fake_store


def test_add_document_raises_when_already_present(tmp_path, monkeypatch):
    """Found via a real bug: re-ingesting the same file silently duplicated
    its chunks, which then crowded a real query's top-k results, hiding
    other genuinely relevant chunks. This is the guard against that.
    """
    doc_path = tmp_path / "notes.md"
    doc_path.write_text("hello")
    fake_store = _mock_ingest_pipeline(monkeypatch, already_present=True)

    with pytest.raises(ValueError, match="already in the knowledge base"):
        add_document(doc_path)

    fake_store.has_chunks_from_source.assert_called_once_with("notes.md")
    fake_store.save.assert_not_called()


def test_add_document_force_skips_the_already_present_check(tmp_path, monkeypatch):
    doc_path = tmp_path / "notes.md"
    doc_path.write_text("hello")
    fake_store = _mock_ingest_pipeline(monkeypatch, already_present=True)

    add_document(doc_path, force=True)

    fake_store.has_chunks_from_source.assert_not_called()
    fake_store.save.assert_called_once()


def test_add_document_proceeds_when_not_already_present(tmp_path, monkeypatch):
    doc_path = tmp_path / "notes.md"
    doc_path.write_text("hello")
    fake_store = _mock_ingest_pipeline(monkeypatch, already_present=False)

    add_document(doc_path)

    fake_store.save.assert_called_once()
