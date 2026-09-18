"""Tests for ingestion.extractors: Extractor ABC, PDFExtractor, MarkdownExtractor, get_extractor."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestion.extractors import (
    MarkdownExtractor,
    PDFExtractor,
    _markdown_section_map,
    get_extractor,
)


def _open_fake_pdf(monkeypatch, pages: list):
    fake_pdf = MagicMock()
    fake_pdf.pages = pages
    fake_pdf.__enter__.return_value = fake_pdf
    monkeypatch.setattr("pdfplumber.open", lambda path: fake_pdf)


def _fake_page(text: str):
    page = MagicMock()
    page.extract_text.return_value = text
    return page


# --- get_extractor ---


def test_get_extractor_returns_pdf_extractor_for_pdf():
    assert isinstance(get_extractor(Path("doc.pdf")), PDFExtractor)


def test_get_extractor_is_case_insensitive():
    assert isinstance(get_extractor(Path("doc.PDF")), PDFExtractor)


def test_get_extractor_raises_on_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported document type"):
        get_extractor(Path("doc.txt"))


def test_get_extractor_returns_markdown_extractor_for_md():
    assert isinstance(get_extractor(Path("notes.md")), MarkdownExtractor)


def test_get_extractor_returns_markdown_extractor_for_markdown_extension():
    assert isinstance(get_extractor(Path("notes.markdown")), MarkdownExtractor)


# --- PDFExtractor.validate ---


def test_pdf_extractor_validate_raises_on_empty_pages(monkeypatch, tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    _open_fake_pdf(monkeypatch, [])

    with pytest.raises(ValueError, match="has no pages"):
        PDFExtractor().validate(pdf_path)


def test_pdf_extractor_validate_raises_on_scanned_pdf(monkeypatch, tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    _open_fake_pdf(monkeypatch, [_fake_page("")])

    with pytest.raises(ValueError, match="scanned PDF"):
        PDFExtractor().validate(pdf_path)


def test_pdf_extractor_validate_passes_for_real_content(monkeypatch, tmp_path):
    pdf_path = tmp_path / "real.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    _open_fake_pdf(monkeypatch, [_fake_page("hello world")])

    PDFExtractor().validate(pdf_path)  # must not raise


# --- PDFExtractor.extract ---


def test_pdf_extractor_extract_delegates_to_extract_document_text(monkeypatch, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    page1 = _fake_page("one two")
    page2 = _fake_page("three")
    _open_fake_pdf(monkeypatch, [page1, page2])

    full_text, word_page_map = PDFExtractor().extract(pdf_path, mode="flat")

    assert full_text == "one two three"
    assert word_page_map == [1, 1, 2]


# --- MarkdownExtractor.validate ---


def test_markdown_extractor_validate_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.md"
    with pytest.raises(FileNotFoundError):
        MarkdownExtractor().validate(missing)


def test_markdown_extractor_validate_raises_on_empty_file(tmp_path):
    md_path = tmp_path / "empty.md"
    md_path.write_text("   \n\n  ")
    with pytest.raises(ValueError, match="is empty"):
        MarkdownExtractor().validate(md_path)


def test_markdown_extractor_validate_passes_for_real_content(tmp_path):
    md_path = tmp_path / "real.md"
    md_path.write_text("# Title\n\nSome content.")
    MarkdownExtractor().validate(md_path)  # must not raise


# --- MarkdownExtractor.extract ---


def test_markdown_extractor_extract_reads_file_as_is(tmp_path):
    md_path = tmp_path / "notes.md"
    md_path.write_text("# Title\n\nHello world.")

    full_text, word_page_map = MarkdownExtractor().extract(md_path)

    assert full_text == "# Title\n\nHello world."
    assert len(word_page_map) == len(full_text.split())


def test_markdown_extractor_extract_ignores_mode_parameter(tmp_path):
    """mode is a PDF-specific concept (flat vs. blocks) — Markdown already
    has its own structure, so this must behave the same regardless of mode.
    """
    md_path = tmp_path / "notes.md"
    md_path.write_text("# Title\n\nHello world.")

    flat_result = MarkdownExtractor().extract(md_path, mode="flat")
    blocks_result = MarkdownExtractor().extract(md_path, mode="blocks")

    assert flat_result == blocks_result


# --- _markdown_section_map ---


def test_markdown_section_map_starts_at_one_before_any_header():
    text = "just a plain paragraph with no headers"
    assert _markdown_section_map(text) == [1] * len(text.split())


def test_markdown_section_map_increments_per_header():
    text = "intro\n\n# First\n\nbody one\n\n## Second\n\nbody two"
    words = text.split()
    result = _markdown_section_map(text)

    assert len(result) == len(words)
    # "intro" is before any header -> section 1; "# First" bumps to 2;
    # "## Second" bumps to 3.
    assert result[words.index("intro")] == 1
    assert result[words.index("First")] == 2
    assert result[words.index("Second")] == 3


def test_markdown_section_map_ignores_headers_inside_code_fences():
    """Regression test for a real risk: a documentation file with a shell/
    Python example containing a '#' comment must not be miscounted as
    starting a new section every time the example does.
    """
    text = "# Real Header\n\n```python\n# not a header, just a comment\nx = 1\n```\n\nmore text"
    words = text.split()
    result = _markdown_section_map(text)

    assert len(result) == len(words)
    assert result[words.index("Real")] == 2
    # Everything from the fenced comment onward stays in section 2 — no
    # bump from the "#" inside the code block.
    assert result[words.index("comment")] == 2
    assert result[words.index("more")] == 2


def test_markdown_section_map_length_matches_full_text_split_with_real_document():
    """The invariant chunk_document() depends on: len(word_page_map) ==
    len(full_text.split()), verified against a realistic multi-section file.
    """
    text = (
        "# Title\n\nIntro paragraph here.\n\n"
        "## Section A\n\nSome words in section A.\n\n"
        "```python\n# this is a comment, not a header\ndef foo():\n    pass\n```\n\n"
        "## Section B\n\nMore words in section B.\n"
    )
    result = _markdown_section_map(text)
    assert len(result) == len(text.split())
