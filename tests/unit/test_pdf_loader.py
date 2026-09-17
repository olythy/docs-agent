"""Tests for ingestion.pdf_loader: extract_pages and is_scanned_pdf."""

from pathlib import Path
from unittest.mock import MagicMock

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


def test_extract_pages_strips_nul_characters(monkeypatch, tmp_path: Path):
    """Regression test for a real bug found via a real PDF.

    Some PDFs have malformed font/encoding data that makes pdfplumber emit
    literal NUL characters for un-decodable glyphs. Postgres TEXT columns
    reject NUL outright, so add_document() failed with
    "string literal cannot contain NUL (0x00) characters" on that document.
    """
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")  # content unused, pdfplumber.open is mocked

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "hello\x00world"

    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf

    monkeypatch.setattr("pdfplumber.open", lambda path: fake_pdf)

    pages = extract_pages(pdf_path)

    assert pages[0]["text"] == "helloworld"
    assert pages[0]["char_count"] == len("helloworld")
