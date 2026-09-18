"""Document extraction diagnostic script.

Purpose:
    A hand-runnable tool to inspect what get_extractor() extracts from a
    given document (PDF or Markdown — format detected from the extension).
    Useful to quickly verify a new document has usable content before
    adding it to the knowledge base.

    The actual extraction logic lives in ``ingestion/extractors.py`` (and,
    for PDFs, ``ingestion/pdf_loader.py`` underneath it). This script only
    adds CLI handling and a human-readable console report, grouped by
    page/section using the same ``word_page_map`` ``add_document()`` itself
    relies on — so this works identically for any supported format.

Usage:
    # Pass the document path as a CLI argument:
    python scripts/extract_text.py /path/to/document.pdf
    python scripts/extract_text.py /path/to/notes.md

    # Or set TEST_DOC_PATH in .env and run without arguments:
    python scripts/extract_text.py
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is importable (needed for running as a script)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.format_utils import truncate

# ---------------------------------------------------------------------------
# TEST_DOC_PATH is intentionally NOT hardcoded here.
# Personal or client-specific filenames must never be committed to git.
# Provide the path either via CLI argument or via TEST_DOC_PATH in .env.
# ---------------------------------------------------------------------------
PREVIEW_CHARS = 500  # How many characters to preview per page/section


def print_report(doc_path: Path, full_text: str, word_page_map: list[int]) -> None:
    """Print a human-readable extraction report to stdout, grouped by page/section.

    Args:
        doc_path: The document file that was processed.
        full_text: The whole document's text, as returned by
            :meth:`ingestion.extractors.Extractor.extract`.
        word_page_map: Page (PDF) or header-section (Markdown) number per
            word in ``full_text.split()`` — same alignment
            :func:`ingestion.chunker.chunk_document` relies on.
    """
    words = full_text.split()

    print("=" * 60)
    print(f"File      : {doc_path.name}")
    print(f"Pages/sections: {len(set(word_page_map))}")
    print(f"Total words: {len(words)}")
    print(f"Total chars: {len(full_text)}")
    print("=" * 60)

    if not words:
        print("\n⚠️  WARNING: No text found.")
        print("   For a PDF, this usually means it's scanned (image-based) —")
        print("   OCR (e.g. Tesseract) would be required to extract text.")
        return

    # Group words by page/section, preserving the order they first appear in.
    section_order: list[int] = []
    section_words: dict[int, list[str]] = {}
    for word, page in zip(words, word_page_map, strict=True):
        if page not in section_words:
            section_words[page] = []
            section_order.append(page)
        section_words[page].append(word)

    for page in section_order:
        text = " ".join(section_words[page])
        print(f"\n--- Page/section {page} ({len(text)} chars) ---")
        print(truncate(text, PREVIEW_CHARS))
        if len(text) > PREVIEW_CHARS:
            print(f"  ... [{len(text) - PREVIEW_CHARS} more characters]")


def main() -> None:
    """Entry point: resolve the target document and run the extraction report.

    Resolution order for the target file:
    1. CLI argument: ``python scripts/extract_text.py /path/to/file.pdf``
    2. ``TEST_DOC_PATH`` environment variable (set in .env)
    3. Exits with a helpful error if neither is provided.
    """
    from config import settings
    from ingestion.extractors import get_extractor

    if len(sys.argv) > 1:
        doc_path = Path(sys.argv[1])
    elif settings.TEST_DOC_PATH:
        doc_path = Path(settings.TEST_DOC_PATH)
    else:
        print("Error: No document path provided.")
        print("  Usage: python scripts/extract_text.py /path/to/document.pdf")
        print("  Or set TEST_DOC_PATH in your .env file.")
        sys.exit(1)

    print(f"\n📄 Extracting text from: {doc_path}\n")

    extractor = get_extractor(doc_path)
    full_text, word_page_map = extractor.extract(doc_path)
    print_report(doc_path, full_text, word_page_map)


if __name__ == "__main__":
    main()
