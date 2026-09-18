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
