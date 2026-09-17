"""Tests for ingestion.pdf_loader: extract_pages and is_scanned_pdf."""

from pathlib import Path

import pytest

from ingestion.pdf_loader import extract_pages, is_scanned_pdf


def test_is_scanned_pdf_empty_pages_list_is_not_scanned():
    """An empty pages list means no pages (empty/corrupt file), not scanning.

    `all()` on an empty iterable is vacuously True, which would otherwise
    misclassify a 0-page/corrupt PDF as "scanned".
    """
    assert is_scanned_pdf([]) is False


def test_is_scanned_pdf_all_pages_empty_is_scanned():
    pages = [
        {"page_number": 1, "text": "", "char_count": 0},
        {"page_number": 2, "text": "", "char_count": 0},
    ]
    assert is_scanned_pdf(pages) is True


def test_is_scanned_pdf_some_text_is_not_scanned():
    pages = [
        {"page_number": 1, "text": "", "char_count": 0},
        {"page_number": 2, "text": "hello", "char_count": 5},
    ]
    assert is_scanned_pdf(pages) is False


def test_extract_pages_raises_file_not_found(tmp_path: Path):
    missing = tmp_path / "does-not-exist.pdf"
    with pytest.raises(FileNotFoundError):
        extract_pages(missing)
