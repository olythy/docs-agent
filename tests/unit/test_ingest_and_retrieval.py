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

from ingestion.ingest import (
    IngestAction,
    _resolve_ingest_action,
    add_directory,
    add_document,
)
from models import Chunk, ChunkMetadata


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
        lambda *a, **kw: [
            Chunk(
                content="x",
                metadata=ChunkMetadata(
                    source_file="x", page_number=None, chunk_index=0
                ),
            )
        ],
    )
    fake_overflow_strategy = MagicMock()
    fake_overflow_strategy.apply.side_effect = lambda chunks, driver: chunks
    monkeypatch.setattr(
        "ingestion.ingest.get_chunk_overflow_strategy", lambda: fake_overflow_strategy
    )

    fake_store = MagicMock()
    fake_store.has_chunks_from_source.return_value = already_present
    fake_store.has_content_hash.return_value = already_present
    fake_store.get_hash_by_source.return_value = (
        "fakehash123" if already_present else None
    )
    fake_store.delete_chunks_by_hash.return_value = 1 if already_present else 0
    fake_store.delete_chunks_from_source.return_value = 0
    fake_store.save.return_value = 1
    fake_store.assert_dimension_matches = MagicMock()
    fake_store.add_source_alias.return_value = 1
    monkeypatch.setattr("ingestion.ingest.VectorStore", lambda: fake_store)

    return fake_store


def test_add_document_raises_when_already_present(tmp_path, monkeypatch):
    """Found via a real bug: re-ingesting the same file silently duplicated
    its chunks, which then crowded a real query's top-k results, hiding
    other genuinely relevant chunks. This is the guard against that.
    """
    from ingestion.hash import compute_file_hash

    doc_path = tmp_path / "notes.md"
    doc_path.write_text("hello")
    expected_hash = compute_file_hash(doc_path)

    fake_store = _mock_ingest_pipeline(monkeypatch, already_present=True)
    fake_store.get_hash_by_source.return_value = expected_hash

    with pytest.raises(ValueError, match="already in the knowledge base"):
        add_document(doc_path)

    fake_store.get_hash_by_source.assert_called_once_with(str(doc_path))
    fake_store.save.assert_not_called()


def test_add_document_force_skips_the_already_present_check(tmp_path, monkeypatch):
    from ingestion.hash import compute_file_hash

    doc_path = tmp_path / "notes.md"
    doc_path.write_text("hello")
    expected_hash = compute_file_hash(doc_path)

    fake_store = _mock_ingest_pipeline(monkeypatch, already_present=True)
    fake_store.get_hash_by_source.return_value = expected_hash

    add_document(doc_path, force=True)

    fake_store.delete_chunks_by_hash.assert_called_once_with(expected_hash)
    fake_store.save.assert_called_once()


def test_add_document_proceeds_when_not_already_present(tmp_path, monkeypatch):
    doc_path = tmp_path / "notes.md"
    doc_path.write_text("hello")
    fake_store = _mock_ingest_pipeline(monkeypatch, already_present=False)

    add_document(doc_path)

    fake_store.save.assert_called_once()


def test_add_directory_raises_on_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        add_directory(tmp_path / "nonexistent")


def test_add_directory_raises_on_file_path(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("not a dir")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        add_directory(f)


def test_add_directory_empty_dir_returns_zero_summary(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    summary = add_directory(empty)
    assert summary == {
        "ingested": [],
        "updated": [],
        "aliased": [],
        "skipped": [],
        "failed": [],
        "total_found": 0,
    }


def test_add_directory_ingests_supported_and_skips_hidden(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()

    # Supported files
    (folder / "doc1.md").write_text("# Doc 1")
    (folder / "doc2.pdf").write_bytes(b"%PDF fake")
    sub = folder / "sub"
    sub.mkdir()
    (sub / "doc3.markdown").write_text("# Doc 3")

    # Unsupported / hidden files to ignore
    (folder / "notes.txt").write_text("ignore me")
    (folder / ".hidden.md").write_text("hidden")
    hidden_dir = folder / ".git"
    hidden_dir.mkdir()
    (hidden_dir / "git_doc.md").write_text("inside hidden dir")

    fake_store = _mock_ingest_pipeline(monkeypatch, already_present=False)

    summary = add_directory(folder, recursive=True)

    assert summary["total_found"] == 3
    assert len(summary["ingested"]) == 3
    assert len(summary["skipped"]) == 0
    assert len(summary["failed"]) == 0
    assert fake_store.save.call_count == 3


def test_add_directory_skips_already_ingested_files(tmp_path, monkeypatch):
    from ingestion.hash import compute_file_hash

    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "new.md").write_text("new content")
    existing_file = folder / "existing.md"
    existing_file.write_text("existing content")
    existing_hash = compute_file_hash(existing_file)

    fake_store = _mock_ingest_pipeline(monkeypatch, already_present=False)
    # Make existing.md return its exact hash so it's recognized as unchanged
    fake_store.get_hash_by_source.side_effect = lambda s: (
        existing_hash if "existing.md" in s else None
    )

    summary = add_directory(folder, force=False)

    assert summary["total_found"] == 2
    assert len(summary["ingested"]) == 1
    assert "new.md" in summary["ingested"][0]
    assert len(summary["skipped"]) == 1
    assert "existing.md" in summary["skipped"][0]


def test_add_directory_aliases_duplicate_content(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "original.md").write_text("identical content")
    (folder / "copy.md").write_text("identical content")

    fake_store = _mock_ingest_pipeline(monkeypatch, already_present=False)
    # Simulate copy.md finding that its content_hash is already present
    fake_store.has_content_hash.side_effect = lambda h: True
    fake_store.get_hash_by_source.return_value = None

    summary = add_directory(folder, force=False)

    assert summary["total_found"] == 2
    assert len(summary["aliased"]) == 2
    assert fake_store.add_source_alias.call_count == 2


def test_add_directory_records_failures_without_aborting(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "good.md").write_text("good")
    (folder / "bad.md").write_text("bad")

    _mock_ingest_pipeline(monkeypatch, already_present=False)

    def fake_extract_with_headers(path, **kw):
        if "bad.md" in str(path):
            raise ValueError("Corrupt file content")
        return "content", [1], None

    fake_extractor = MagicMock()
    fake_extractor.extract_with_headers.side_effect = fake_extract_with_headers
    fake_extractor.extract.side_effect = fake_extract_with_headers
    monkeypatch.setattr("ingestion.ingest.get_extractor", lambda p: fake_extractor)

    summary = add_directory(folder)

    assert summary["total_found"] == 2
    assert len(summary["ingested"]) == 1
    assert len(summary["failed"]) == 1
    assert "bad.md" in summary["failed"][0]["file"]
    assert "Corrupt file content" in summary["failed"][0]["error"]


def test_add_directory_filters_by_runtime_allowed_extensions(tmp_path, monkeypatch):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "file.md").write_text("# Markdown")
    (folder / "file.pdf").write_bytes(b"%PDF dummy")

    fake_store = _mock_ingest_pipeline(monkeypatch, already_present=False)

    summary = add_directory(folder, allowed_extensions=[".md"])

    assert summary["total_found"] == 1
    assert len(summary["ingested"]) == 1
    assert "file.md" in summary["ingested"][0]
    assert fake_store.save.call_count == 1


def test_add_directory_logs_warning_for_unregistered_extensions(
    tmp_path, monkeypatch, caplog
):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "file.doc").write_text("legacy doc dummy")

    import logging

    with caplog.at_level(logging.WARNING):
        summary = add_directory(folder, allowed_extensions=[".doc"])

    assert summary["total_found"] == 0
    assert any("have no registered extractor" in r.message for r in caplog.records)


class TestResolveIngestAction:
    """Isolated coverage for the pure dedup/versioning decision extracted out
    of add_document/add_directory — see ingestion.ingest.IngestAction's
    docstring for what each case means. No I/O, no mocking needed: every
    case is just the four (existing_hash, hash_already_stored, force)
    inputs mapped to the expected decision.
    """

    CONTENT_HASH = "abc123"

    def test_skip_unchanged_when_same_hash_and_not_forced(self):
        decision = _resolve_ingest_action(
            existing_hash=self.CONTENT_HASH,
            hash_already_stored=True,
            content_hash=self.CONTENT_HASH,
            force=False,
        )
        assert decision.action is IngestAction.SKIP_UNCHANGED
        assert decision.hash_to_delete is None

    def test_alias_when_content_exists_elsewhere_and_not_forced(self):
        decision = _resolve_ingest_action(
            existing_hash=None,
            hash_already_stored=True,
            content_hash=self.CONTENT_HASH,
            force=False,
        )
        assert decision.action is IngestAction.ALIAS_EXISTING_CONTENT
        assert decision.hash_to_delete is None

    def test_replace_previous_version_when_source_has_a_different_hash(self):
        decision = _resolve_ingest_action(
            existing_hash="old_hash",
            hash_already_stored=False,
            content_hash=self.CONTENT_HASH,
            force=False,
        )
        assert decision.action is IngestAction.REPLACE_PREVIOUS_VERSION
        assert decision.hash_to_delete == "old_hash"

    def test_replace_previous_version_also_applies_when_forced(self):
        """force=True re-processes a changed source unconditionally — same
        REPLACE outcome as the unforced case, just reached without the
        skip/alias checks running first."""
        decision = _resolve_ingest_action(
            existing_hash="old_hash",
            hash_already_stored=False,
            content_hash=self.CONTENT_HASH,
            force=True,
        )
        assert decision.action is IngestAction.REPLACE_PREVIOUS_VERSION
        assert decision.hash_to_delete == "old_hash"

    def test_force_replace_duplicate_when_forced_past_an_alias_case(self):
        """The force=True counterpart to the alias case: this source path is
        new (no existing_hash), but the content duplicates chunks already
        stored under another path — force reingests it anyway, deleting the
        other path's chunks by content_hash first."""
        decision = _resolve_ingest_action(
            existing_hash=None,
            hash_already_stored=True,
            content_hash=self.CONTENT_HASH,
            force=True,
        )
        assert decision.action is IngestAction.FORCE_REPLACE_DUPLICATE
        assert decision.hash_to_delete == self.CONTENT_HASH

    def test_insert_new_when_nothing_matches(self):
        decision = _resolve_ingest_action(
            existing_hash=None,
            hash_already_stored=False,
            content_hash=self.CONTENT_HASH,
            force=False,
        )
        assert decision.action is IngestAction.INSERT_NEW
        assert decision.hash_to_delete is None

    def test_insert_new_also_applies_when_forced_with_nothing_to_replace(self):
        decision = _resolve_ingest_action(
            existing_hash=None,
            hash_already_stored=False,
            content_hash=self.CONTENT_HASH,
            force=True,
        )
        assert decision.action is IngestAction.INSERT_NEW
        assert decision.hash_to_delete is None
