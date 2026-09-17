"""PDF text extraction module.

Provides functions to extract plain text from PDF documents using pdfplumber.
This is the single, canonical implementation used by both the diagnostic
script (``scripts/extract_text.py``) and the ingestion pipeline
(``ingestion/ingest.py``).

Key exports:
    extract_pages   -- Extract page-level text dicts from a PDF file.
    is_scanned_pdf  -- Detect whether a PDF is image-based (no text layer).
"""

from pathlib import Path


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
