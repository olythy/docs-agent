"""Document ingestion pipeline.

Orchestrates the full add_document flow:
    1. Extract text page by page from a PDF (``pdf_loader``).
    2. Split each page's text into overlapping word-based chunks (``chunker``).
    3. Embed every chunk with the active embedding driver (``drivers.embedding``).
    4. Store chunks + embeddings in the ``document_chunks`` Postgres table.

This module exposes a single public function: :func:`add_document`.

Usage::

    from ingestion.ingest import add_document
    add_document("/path/to/document.pdf")
"""

from pathlib import Path

from config import settings
from drivers.embedding import get_embedding_driver
from ingestion.chunker import chunk_pages
from ingestion.pdf_loader import extract_pages, is_scanned_pdf
from store import VectorStore


def add_document(file_path: str | Path) -> None:
    """Ingest a PDF document into the RAG knowledge base.

    This is the main tool exposed to the agent. It runs the full pipeline:
    extract → chunk → embed → store.

    Args:
        file_path: Path to the PDF file to ingest (str or Path).

    Raises:
        FileNotFoundError: If the PDF does not exist at ``file_path``.
        RuntimeError: If the database connection is not configured.
        ValueError: If the PDF has no pages, or is scanned (no extractable
            text layer).
    """
    pdf_path = Path(file_path)
    source_file = pdf_path.name

    print(f"[ingest] Starting ingestion: {source_file}")

    # Step 1: Extract text from every page
    pages = extract_pages(pdf_path)
    if not pages:
        raise ValueError(
            f"'{source_file}' has no pages. The file may be empty or corrupted."
        )
    if is_scanned_pdf(pages):
        raise ValueError(
            f"'{source_file}' appears to be a scanned PDF with no text layer. "
            "OCR support is not implemented in this version."
        )
    print(f"[ingest] Extracted text from {len(pages)} page(s).")

    # Step 2: Chunk the extracted text
    chunks = chunk_pages(pages, source_file=source_file)
    print(f"[ingest] Created {len(chunks)} chunk(s) "
          f"(~{settings.CHUNK_SIZE} words each, {settings.CHUNK_OVERLAP} overlap).")

    # Step 3: Embed all chunks in one batched call
    driver = get_embedding_driver()
    texts = [c["content"] for c in chunks]
    print(f"[ingest] Embedding with driver='{settings.EMBEDDING_DRIVER}' ...")
    embeddings = driver.embed_batch(texts)
    print(f"[ingest] Embeddings ready. Dimension: {len(embeddings[0])}.")

    # Step 4: Store in Postgres
    inserted = VectorStore().save(chunks, embeddings)
    print(f"[ingest] Stored {inserted} row(s) in document_chunks. Done! ✅")
