"""PDF text extraction module.

Provides functions to extract plain text from PDF documents using pdfplumber.
This is the single, canonical implementation used by both the diagnostic
script (``scripts/extract_text.py``) and the ingestion pipeline
(``ingestion/ingest.py``).

Key exports:
    extract_pages          -- Extract page-level text dicts from a PDF file.
    extract_document_text  -- Extract one document-level string (see below)
                               plus a word-to-page-number map, for chunking
                               that isn't blind to page boundaries.
    is_scanned_pdf         -- Detect whether a PDF is image-based (no text layer).
"""

import statistics
from itertools import pairwise
from pathlib import Path

#: A line-to-line vertical gap larger than this multiple of a page's median
#: line spacing is treated as a paragraph break in PDF_EXTRACTION_MODE="blocks".
#: Empirically tuned against two real documents (see PLAN.md "Ismert
#: korlátok") — not perfect on every document, since pdfplumber's coordinate
#: system is per-page, so a paragraph break that happens to fall exactly at
#: a page boundary can never be detected this way (nothing to compare the
#: gap against on the other side of the boundary).
PARAGRAPH_GAP_MULTIPLIER = 1.8


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text from every page of a PDF file using pdfplumber.

    pdfplumber uses x/y character coordinates to reconstruct proper reading
    order, which produces cleaner output for tabular content (e.g. tax
    statements, contracts) compared to pypdf.

    Args:
        pdf_path: Path to the PDF file to process.

    Returns:
        A list of page dicts, each containing:
            - ``page_number`` (int): 1-based page index.
            - ``text`` (str): Extracted text (empty string for image pages).
            - ``char_count`` (int): Number of characters extracted.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not point to an existing file.
    """
    import pdfplumber

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # extract_text() returns None for scanned/image-only pages
            text = page.extract_text() or ""
            # Malformed font/encoding data in some PDFs makes pdfplumber emit
            # literal NUL characters for un-decodable glyphs. Postgres TEXT
            # columns reject NUL outright ("string literal cannot contain
            # NUL (0x00) characters"), so this must be stripped here, at the
            # extraction source, rather than defensively re-checked by every
            # downstream consumer (chunking, storage).
            text = text.replace("\x00", "")
            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "char_count": len(text),
                }
            )

    return pages


def extract_document_text(pdf_path: Path, mode: str = "flat") -> tuple[str, list[int]]:
    """Extract a whole document as one string, plus a per-word page-number map.

    Concatenating pages *before* chunking (rather than chunking each page
    separately, as :func:`extract_pages`-based callers used to) fixes a real
    problem: a paragraph that runs across a page break used to become two
    independent, truncated chunks, with no overlap between them.

    Args:
        pdf_path: Path to the PDF file to process.
        mode: ``"flat"`` (default) joins each page's plain-text words with a
            single space — fast, but paragraph structure is lost. ``"blocks"``
            additionally detects paragraph breaks from word coordinates and
            preserves them as ``"\\n\\n"``, so a structure-aware chunking
            strategy (see :class:`ingestion.chunker.LangChainChunkingStrategy`)
            can split on them.

    Returns:
        A ``(full_text, word_page_map)`` tuple. ``word_page_map[i]`` is the
        1-based page number that ``full_text.split()[i]`` came from — the two
        are always the same length, even in ``"blocks"`` mode where
        ``full_text`` also contains ``"\\n\\n"`` markers: those are appended
        onto the *previous* word's own string rather than added as separate
        list entries, so ``str.split()`` (which treats any whitespace run,
        including embedded newlines, as a single separator) never produces
        an extra token for them.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not point to an existing file.
        ValueError: If ``mode`` is not ``"flat"`` or ``"blocks"``.
    """
    if mode == "flat":
        pages = extract_pages(pdf_path)
        word_texts: list[str] = []
        word_page_map: list[int] = []
        for page in pages:
            words = page["text"].split()
            word_texts.extend(words)
            word_page_map.extend([page["page_number"]] * len(words))
        return " ".join(word_texts), word_page_map

    if mode == "blocks":
        return _extract_blocks_text(pdf_path)

    raise ValueError(f"Unknown PDF_EXTRACTION_MODE: '{mode}'. Valid options are: 'flat', 'blocks'.")


def _extract_blocks_text(pdf_path: Path) -> tuple[str, list[int]]:
    """Coordinate-based variant of :func:`extract_document_text` (mode="blocks").

    Reasoning for the threshold and the page-boundary limitation lives on
    :data:`PARAGRAPH_GAP_MULTIPLIER`.
    """
    import pdfplumber

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    word_texts: list[str] = []
    word_page_map: list[int] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            if not words:
                continue

            unique_tops = sorted({round(w["top"], 1) for w in words})
            gaps = [b - a for a, b in pairwise(unique_tops)]
            median_gap = statistics.median(gaps) if gaps else 0
            threshold = median_gap * PARAGRAPH_GAP_MULTIPLIER if median_gap else float("inf")

            prev_top: float | None = None
            for w in words:
                text = w["text"].replace("\x00", "")
                if not text:
                    continue
                if prev_top is not None and (w["top"] - prev_top) > threshold and word_texts:
                    word_texts[-1] += "\n\n"
                word_texts.append(text)
                word_page_map.append(page_number)
                prev_top = w["top"]
            # No paragraph-break check across the page boundary itself:
            # pdfplumber's "top" coordinate resets per page, so the gap
            # between the last word of this page and the first word of the
            # next isn't meaningful to compare against this page's threshold.

    return " ".join(word_texts), word_page_map


def is_scanned_pdf(pages: list[dict]) -> bool:
    """Return True if the PDF appears to be image-based (no extractable text).

    A PDF is considered scanned when it has at least one page and every page
    returns zero characters. In this case OCR (e.g. Tesseract) would be
    required to extract content.

    An empty ``pages`` list (e.g. a 0-page or corrupted PDF) is deliberately
    *not* considered scanned: ``all()`` on an empty iterable is vacuously
    ``True``, which would otherwise misreport an empty/corrupt file as a
    scanning problem. Callers should check for an empty ``pages`` list
    separately if that case needs its own handling.

    Args:
        pages: The list of page dicts returned by :func:`extract_pages`.

    Returns:
        True if the PDF has pages but no text was found on any of them,
        False otherwise (including when ``pages`` is empty).
    """
    if not pages:
        return False
    return all(p["char_count"] == 0 for p in pages)
