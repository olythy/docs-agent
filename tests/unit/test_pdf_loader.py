"""Tests for ingestion.pdf_loader: extract_pages and is_scanned_pdf."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestion.pdf_loader import extract_document_text, extract_pages, is_scanned_pdf


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


def test_extract_pages_strips_unmapped_glyph_markers(monkeypatch, tmp_path: Path):
    """Regression test for a real bug found via a real PDF.

    Some PDFs have a font with a broken/missing ToUnicode mapping for
    certain glyphs (here: bullet points in a list) — pdfplumber falls back
    to rendering them as literal "(cid:N)" text instead of the actual
    character. Left in, this becomes garbage stored (and embedded)
    verbatim, e.g. "(cid:127) 1-2 courts" instead of "1-2 courts".
    """
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    fake_page = MagicMock()
    fake_page.extract_text.return_value = "Pricing:\n(cid:127) Starter\n(cid:127) Pro"

    fake_pdf = MagicMock()
    fake_pdf.pages = [fake_page]
    fake_pdf.__enter__.return_value = fake_pdf

    monkeypatch.setattr("pdfplumber.open", lambda path: fake_pdf)

    pages = extract_pages(pdf_path)

    assert "(cid:" not in pages[0]["text"]
    assert pages[0]["text"] == "Pricing:\n Starter\n Pro"


def _fake_page_with_words(words: list[dict]):
    """Build a MagicMock page whose extract_words() returns the given word dicts."""
    page = MagicMock()
    page.extract_words.return_value = words
    return page


def _open_fake_pdf(monkeypatch, pages: list):
    fake_pdf = MagicMock()
    fake_pdf.pages = pages
    fake_pdf.__enter__.return_value = fake_pdf
    monkeypatch.setattr("pdfplumber.open", lambda path: fake_pdf)


# --- extract_document_text(mode="flat") ---


def test_extract_document_text_flat_joins_pages_with_page_map(monkeypatch, tmp_path):
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    page1 = MagicMock()
    page1.extract_text.return_value = "one two"
    page2 = MagicMock()
    page2.extract_text.return_value = "three"
    _open_fake_pdf(monkeypatch, [page1, page2])

    full_text, word_page_map = extract_document_text(pdf_path, mode="flat")

    assert full_text == "one two three"
    assert word_page_map == [1, 1, 2]


def test_extract_document_text_flat_raises_file_not_found(tmp_path):
    missing = tmp_path / "does-not-exist.pdf"
    with pytest.raises(FileNotFoundError):
        extract_document_text(missing, mode="flat")


def test_extract_document_text_raises_on_unknown_mode(tmp_path):
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(ValueError, match="Unknown PDF_EXTRACTION_MODE"):
        extract_document_text(pdf_path, mode="bogus")


# --- extract_document_text(mode="blocks") ---


def test_extract_document_text_blocks_marks_paragraph_break_on_large_gap(
    monkeypatch, tmp_path
):
    """A y-gap much larger than the page's median line spacing becomes '\\n\\n'."""
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    # Line gaps: 10, 10, 60, 10 -> median=10, threshold=18 -> only the 60-gap
    # (between "friend" and "New") counts as a paragraph break.
    words = [
        {"text": "Hello", "top": 10},
        {"text": "there", "top": 20},
        {"text": "friend", "top": 30},
        {"text": "New", "top": 90},
        {"text": "paragraph", "top": 100},
    ]
    _open_fake_pdf(monkeypatch, [_fake_page_with_words(words)])

    full_text, word_page_map = extract_document_text(pdf_path, mode="blocks")

    assert full_text == "Hello there friend\n\n New paragraph"
    assert word_page_map == [1, 1, 1, 1, 1]
    # The paragraph marker is whitespace-only, so it must not appear as its
    # own token — the invariant this whole design leans on.
    assert len(full_text.split()) == len(word_page_map)


def test_extract_document_text_blocks_no_break_across_pages(monkeypatch, tmp_path):
    """No paragraph break is inserted at a page boundary (coordinates aren't comparable)."""
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    page1_words = [{"text": "end", "top": 10}]
    page2_words = [{"text": "start", "top": 700}]  # huge gap vs. page 1, but different page
    _open_fake_pdf(
        monkeypatch,
        [_fake_page_with_words(page1_words), _fake_page_with_words(page2_words)],
    )

    full_text, word_page_map = extract_document_text(pdf_path, mode="blocks")

    assert "\n\n" not in full_text
    assert full_text == "end start"
    assert word_page_map == [1, 2]


def test_extract_document_text_blocks_strips_nul_characters(monkeypatch, tmp_path):
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    words = [{"text": "hello\x00world", "top": 10}]
    _open_fake_pdf(monkeypatch, [_fake_page_with_words(words)])

    full_text, _ = extract_document_text(pdf_path, mode="blocks")

    assert full_text == "helloworld"


def test_extract_document_text_blocks_skips_pages_with_no_words(monkeypatch, tmp_path):
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    empty_page = _fake_page_with_words([])
    words_page = _fake_page_with_words([{"text": "hi", "top": 10}])
    _open_fake_pdf(monkeypatch, [empty_page, words_page])

    full_text, word_page_map = extract_document_text(pdf_path, mode="blocks")

    assert full_text == "hi"
    assert word_page_map == [2]


def test_extract_document_text_blocks_raises_file_not_found(tmp_path):
    missing = tmp_path / "does-not-exist.pdf"
    with pytest.raises(FileNotFoundError):
        extract_document_text(missing, mode="blocks")
