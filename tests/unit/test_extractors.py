"""Tests for ingestion.extractors: Extractor ABC, PDFExtractor, MarkdownExtractor, get_extractor."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

from ingestion.extractors import (
    EXTRACTOR_REGISTRY,
    SUPPORTED_EXTENSIONS,
    DocxExtractor,
    MarkdownExtractor,
    PDFExtractor,
    RtfExtractor,
    _markdown_headers_and_sections,
    _markdown_section_map,
    get_extractor,
    normalize_extensions,
)


def _write_docx(path: Path, paragraphs: list[tuple[str, bool]]) -> None:
    """Writes a real .docx file — (text, is_centered) per paragraph."""
    document = Document()
    for text, centered in paragraphs:
        paragraph = document.add_paragraph(text)
        if centered:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.save(path)


def _write_rtf(path: Path, body: str) -> None:
    """Writes a minimal, real RTF file wrapping ``body`` (already RTF-escaped if needed)."""
    path.write_text(
        r"{\rtf1\ansi\ansicpg1250\deff0 {\fonttbl{\f0 Times New Roman;}}"
        rf"\pard {body}\par }}",
        encoding="latin-1",
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


# --- get_extractor & registry ---


def test_extractor_registry_and_supported_extensions():
    assert ".pdf" in EXTRACTOR_REGISTRY
    assert ".md" in EXTRACTOR_REGISTRY
    assert ".markdown" in EXTRACTOR_REGISTRY
    assert ".docx" in EXTRACTOR_REGISTRY
    assert ".rtf" in EXTRACTOR_REGISTRY
    assert EXTRACTOR_REGISTRY[".pdf"] is PDFExtractor
    assert EXTRACTOR_REGISTRY[".md"] is MarkdownExtractor
    assert EXTRACTOR_REGISTRY[".docx"] is DocxExtractor
    assert EXTRACTOR_REGISTRY[".rtf"] is RtfExtractor
    assert EXTRACTOR_REGISTRY[".markdown"] is MarkdownExtractor
    assert SUPPORTED_EXTENSIONS == frozenset(EXTRACTOR_REGISTRY.keys())


def test_normalize_extensions_normalizes_case_and_adds_dots():
    raw = ["md", ".PDF", "  .markdown  ", "", "  "]
    result = normalize_extensions(raw)
    assert result == frozenset({".md", ".pdf", ".markdown"})


def test_get_extractor_returns_pdf_extractor_for_pdf():
    assert isinstance(get_extractor(Path("doc.pdf")), PDFExtractor)


def test_get_extractor_is_case_insensitive():
    assert isinstance(get_extractor(Path("doc.PDF")), PDFExtractor)


def test_get_extractor_raises_on_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported document type: '.txt'"):
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


def test_pdf_extractor_extract_delegates_to_extract_document_text(
    monkeypatch, tmp_path
):
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


def test_markdown_headers_and_sections_hierarchical_stack():
    text = (
        "Intro text\n\n"
        "# Chapter 1\n\n"
        "Chapter 1 content\n\n"
        "## Section 1.1\n\n"
        "Section 1.1 content\n\n"
        "### Detail 1.1.1\n\n"
        "Detail content\n\n"
        "## Section 1.2\n\n"
        "Section 1.2 content\n\n"
        "# Chapter 2\n\n"
        "Chapter 2 content\n"
    )
    words = text.split()
    sections, headers = _markdown_headers_and_sections(text)

    assert len(sections) == len(words)
    assert len(headers) == len(words)

    # Intro before any header has empty breadcrumb and section 1
    assert headers[words.index("Intro")] == ""
    assert sections[words.index("Intro")] == 1

    # Chapter 1
    assert headers[words.index("Chapter")] == "# Chapter 1"
    assert sections[words.index("Chapter")] == 2

    # Section 1.1
    assert headers[words.index("Section")] == "# Chapter 1 > ## Section 1.1"

    # Detail 1.1.1
    assert (
        headers[words.index("Detail")]
        == "# Chapter 1 > ## Section 1.1 > ### Detail 1.1.1"
    )

    # Section 1.2 popped Detail 1.1.1 back to Section level
    idx_1_2 = words.index("1.2")
    assert headers[idx_1_2] == "# Chapter 1 > ## Section 1.2"

    # Chapter 2 popped all Chapter 1 descendants
    idx_ch2 = words.index("Chapter", words.index("1.2"))
    assert headers[idx_ch2] == "# Chapter 2"


def test_markdown_extractor_extract_with_headers(tmp_path):
    md_path = tmp_path / "test.md"
    md_path.write_text("# Overview\n\nSome text.\n\n## Sub\n\nMore text.")

    extractor = MarkdownExtractor()
    full_text, _sections, headers = extractor.extract_with_headers(md_path)

    assert full_text == "# Overview\n\nSome text.\n\n## Sub\n\nMore text."
    assert headers is not None
    assert len(headers) == len(full_text.split())
    assert headers[0] == "# Overview"
    assert headers[-1] == "# Overview > ## Sub"


def test_pdf_extractor_extract_with_headers_returns_none_for_headers(
    monkeypatch, tmp_path
):
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dummy")
    page = _fake_page("pdf text")
    _open_fake_pdf(monkeypatch, [page])

    extractor = PDFExtractor()
    full_text, page_map, headers = extractor.extract_with_headers(pdf_path)

    assert full_text == "pdf text"
    assert page_map == [1, 1]
    assert headers is None


# --- DocxExtractor ---


def test_get_extractor_returns_docx_extractor_for_docx():
    assert isinstance(get_extractor(Path("file.docx")), DocxExtractor)


def test_docx_extractor_validate_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        DocxExtractor().validate(tmp_path / "missing.docx")


def test_docx_extractor_validate_raises_on_empty_document(tmp_path):
    docx_path = tmp_path / "empty.docx"
    _write_docx(docx_path, [])
    with pytest.raises(ValueError, match="no extractable text"):
        DocxExtractor().validate(docx_path)


def test_docx_extractor_validate_raises_on_corrupted_file(tmp_path):
    docx_path = tmp_path / "corrupted.docx"
    docx_path.write_bytes(b"not actually a docx")
    with pytest.raises(ValueError, match="not a valid DOCX"):
        DocxExtractor().validate(docx_path)


def test_docx_extractor_validate_passes_for_real_content(tmp_path):
    docx_path = tmp_path / "real.docx"
    _write_docx(docx_path, [("Some real content.", False)])
    DocxExtractor().validate(docx_path)  # must not raise


def test_docx_extractor_extract_joins_paragraphs(tmp_path):
    docx_path = tmp_path / "doc.docx"
    _write_docx(
        docx_path,
        [
            ("First paragraph.", False),
            ("Second paragraph.", False),
        ],
    )

    full_text, word_section_map = DocxExtractor().extract(docx_path)

    assert full_text == "First paragraph. Second paragraph."
    assert word_section_map == [1, 1, 1, 1]


def test_docx_extractor_extract_ignores_blank_paragraphs(tmp_path):
    docx_path = tmp_path / "doc.docx"
    _write_docx(docx_path, [("Real text.", False), ("   ", False)])

    full_text, _ = DocxExtractor().extract(docx_path)

    assert full_text == "Real text."


def test_docx_extractor_extract_collapses_embedded_whitespace(tmp_path):
    docx_path = tmp_path / "doc.docx"
    _write_docx(docx_path, [("Word1   \n  Word2", False)])

    full_text, _ = DocxExtractor().extract(docx_path)

    assert full_text == "Word1 Word2"


def test_docx_extractor_extract_ignores_mode_parameter(tmp_path):
    docx_path = tmp_path / "doc.docx"
    _write_docx(docx_path, [("Text.", False)])

    flat_text, flat_sections = DocxExtractor().extract(docx_path, mode="flat")
    blocks_text, blocks_sections = DocxExtractor().extract(docx_path, mode="blocks")

    assert flat_text == blocks_text
    assert flat_sections == blocks_sections


def test_docx_extractor_section_index_increments_on_centered_paragraph(tmp_path):
    docx_path = tmp_path / "doc.docx"
    _write_docx(
        docx_path,
        [
            ("Preamble.", False),
            ("ítélete", True),
            ("Body text one.", False),
            ("Indokolás", True),
            ("Body text two.", False),
        ],
    )

    full_text, word_section_map = DocxExtractor().extract(docx_path)

    words = full_text.split()
    sections_by_word = dict(zip(words, word_section_map, strict=True))
    assert sections_by_word["Preamble."] == 1
    assert sections_by_word["ítélete"] == 2
    assert sections_by_word["one."] == 2
    assert sections_by_word["Indokolás"] == 3
    assert sections_by_word["two."] == 3


def test_docx_extractor_extract_with_headers_tracks_most_recent_centered_title(
    tmp_path,
):
    docx_path = tmp_path / "doc.docx"
    _write_docx(
        docx_path,
        [
            ("Court name.", False),
            ("ítélete", True),
            ("Body text one.", False),
            ("Indokolás", True),
            ("Body text two.", False),
        ],
    )

    full_text, _sections, headers = DocxExtractor().extract_with_headers(docx_path)

    assert headers is not None
    assert len(headers) == len(full_text.split())
    words = full_text.split()
    headers_by_word = dict(zip(words, headers, strict=True))
    assert headers_by_word["Court"] == ""
    assert headers_by_word["ítélete"] == "ítélete"
    assert headers_by_word["one."] == "ítélete"
    assert headers_by_word["two."] == "Indokolás"


# --- RtfExtractor ---


def test_get_extractor_returns_rtf_extractor_for_rtf():
    assert isinstance(get_extractor(Path("file.rtf")), RtfExtractor)


def test_rtf_extractor_validate_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        RtfExtractor().validate(tmp_path / "missing.rtf")


def test_rtf_extractor_validate_raises_on_non_rtf_content(tmp_path):
    rtf_path = tmp_path / "fake.rtf"
    rtf_path.write_text("This is just plain text, not RTF.")
    with pytest.raises(ValueError, match="does not look like a valid RTF"):
        RtfExtractor().validate(rtf_path)


def test_rtf_extractor_validate_raises_on_empty_body(tmp_path):
    rtf_path = tmp_path / "empty.rtf"
    _write_rtf(rtf_path, "")
    with pytest.raises(ValueError, match="no extractable text"):
        RtfExtractor().validate(rtf_path)


def test_rtf_extractor_validate_passes_for_real_content(tmp_path):
    rtf_path = tmp_path / "real.rtf"
    _write_rtf(rtf_path, "Some real content.")
    RtfExtractor().validate(rtf_path)  # must not raise


def test_rtf_extractor_extract_strips_control_words(tmp_path):
    rtf_path = tmp_path / "doc.rtf"
    _write_rtf(rtf_path, r"\qc Centered title\par Body text here.")

    full_text, word_page_map = RtfExtractor().extract(rtf_path)

    assert "Centered title" in full_text
    assert "Body text here." in full_text
    assert "\\qc" not in full_text
    assert len(word_page_map) == len(full_text.split())


def test_rtf_extractor_extract_decodes_hex_escaped_hungarian_characters(tmp_path):
    """Regression/documentation test for a real striprtf quirk: the space
    immediately after a hex-escaped character (``\\'e9 ``) is sometimes
    swallowed, merging it with the next word ("t\\'e9 l" -> "té l", not
    "t é l"). Confirmed against the library directly, not assumed — this
    only matters for letter-spaced text (the exact convention real
    corpus documents use for centered section titles, e.g. "í t é l e t
    e t :"), and RtfExtractor doesn't do header detection at all yet (see
    its docstring), so a cosmetic spacing slip here doesn't affect
    anything this extractor is actually relied on for today.
    """
    rtf_path = tmp_path / "doc.rtf"
    _write_rtf(rtf_path, r"\'ed t\'e9 l e t")

    full_text, _ = RtfExtractor().extract(rtf_path)

    assert full_text.strip() == "í té l e t"


def test_rtf_extractor_extract_word_page_map_is_uniformly_one(tmp_path):
    rtf_path = tmp_path / "doc.rtf"
    _write_rtf(rtf_path, "one two three four")

    full_text, word_page_map = RtfExtractor().extract(rtf_path)

    assert word_page_map == [1] * len(full_text.split())


def test_rtf_extractor_extract_ignores_mode_parameter(tmp_path):
    rtf_path = tmp_path / "doc.rtf"
    _write_rtf(rtf_path, "Some text.")

    flat_text, flat_pages = RtfExtractor().extract(rtf_path, mode="flat")
    blocks_text, blocks_pages = RtfExtractor().extract(rtf_path, mode="blocks")

    assert flat_text == blocks_text
    assert flat_pages == blocks_pages


def test_rtf_extractor_extract_with_headers_returns_none_for_headers(tmp_path):
    rtf_path = tmp_path / "doc.rtf"
    _write_rtf(rtf_path, "Some text.")

    full_text, word_page_map, headers = RtfExtractor().extract_with_headers(rtf_path)

    assert full_text.strip() == "Some text."
    assert word_page_map == [1] * len(full_text.split())
    assert headers is None
