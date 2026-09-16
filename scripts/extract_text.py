"""PDF text extraction diagnostic script.

Purpose:
    A hand-runnable tool to inspect what pdfplumber extracts from a given PDF.
    Useful to quickly verify whether a new document has a native text layer
    before adding it to the knowledge base.

    The actual extraction logic lives in ``ingestion/pdf_loader.py``.
    This script only adds CLI handling and a human-readable console report.

Usage:
    # Pass the PDF path as a CLI argument:
    python scripts/extract_text.py /path/to/document.pdf

    # Or set TEST_PDF_PATH in .env and run without arguments:
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

# ---------------------------------------------------------------------------
# TEST_PDF_PATH is intentionally NOT hardcoded here.
# Personal or client-specific filenames must never be committed to git.
# Provide the path either via CLI argument or via TEST_PDF_PATH in .env.
# ---------------------------------------------------------------------------
PREVIEW_CHARS = 500  # How many characters to preview per page


def print_report(pdf_path: Path, pages: list[dict]) -> None:
    """Print a human-readable extraction report to stdout.

    Args:
        pdf_path: The PDF file that was processed.
        pages: The list of page dicts returned by
            :func:`ingestion.pdf_loader.extract_pages`.
    """
    from ingestion.pdf_loader import is_scanned_pdf

    total_chars = sum(p["char_count"] for p in pages)

    print("=" * 60)
    print(f"File      : {pdf_path.name}")
    print(f"Pages     : {len(pages)}")
    print(f"Total chars: {total_chars}")
    print("=" * 60)

    if is_scanned_pdf(pages):
        print("\n⚠️  WARNING: No text found in any page.")
        print("   This PDF is likely scanned (image-based).")
        print("   OCR (e.g. Tesseract) would be required to extract text.")
        return

    for page in pages:
        print(f"\n--- Page {page['page_number']} ({page['char_count']} chars) ---")
        if page["char_count"] == 0:
            print("  (empty page — possibly an image or blank)")
        else:
            preview = page["text"].strip()[:PREVIEW_CHARS]
            print(preview)
            if page["char_count"] > PREVIEW_CHARS:
                print(f"  ... [{page['char_count'] - PREVIEW_CHARS} more characters]")


def main() -> None:
    """Entry point: resolve the target PDF and run the extraction report.

    Resolution order for the target file:
    1. CLI argument: ``python scripts/extract_text.py /path/to/file.pdf``
    2. ``TEST_PDF_PATH`` environment variable (set in .env)
    3. Exits with a helpful error if neither is provided.
    """
    from config import settings
    from ingestion.pdf_loader import extract_pages

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    elif settings.TEST_PDF_PATH:
        pdf_path = Path(settings.TEST_PDF_PATH)
    else:
        print("Error: No PDF path provided.")
        print("  Usage: python scripts/extract_text.py /path/to/document.pdf")
        print("  Or set TEST_PDF_PATH in your .env file.")
        sys.exit(1)

    print(f"\n📄 Extracting text from: {pdf_path}\n")

    pages = extract_pages(pdf_path)
    print_report(pdf_path, pages)


if __name__ == "__main__":
    main()
