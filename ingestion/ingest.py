"""Document ingestion pipeline.

Orchestrates the full add_document flow:
    1. Validate the source document (``ingestion.extractors.Extractor.validate``).
    2. Extract the whole document as one string + a word-to-page/section map
       (``Extractor.extract``) — concatenating pages *before* chunking is
       what avoids truncating a paragraph that spans a page break.
    3. Split the document into chunks (``chunker.chunk_document``), per
       ``settings.CHUNKING_STRATEGY``.
    4. Embed every chunk with the active embedding driver (``drivers.embedding``).
    5. Store chunks + embeddings in the ``document_chunks`` Postgres table.

This module exposes a single public function: :func:`add_document`.

Usage::

    from ingestion.ingest import add_document
    add_document("/path/to/document.pdf")
    add_document("/path/to/notes.md")
"""

from pathlib import Path

from config import settings
from drivers.embedding import get_embedding_driver
from ingestion.chunker import chunk_document, get_chunk_overflow_strategy
from ingestion.extractors import get_extractor
from store import VectorStore


def add_document(file_path: str | Path) -> None:
    """Ingest a document into the RAG knowledge base.

    This is the main tool exposed to the agent. It runs the full pipeline:
    extract → chunk → embed → store. The document's format is detected from
    its extension (see :func:`ingestion.extractors.get_extractor`).

    Args:
        file_path: Path to the source document (str or Path).

    Raises:
        FileNotFoundError: If the file does not exist at ``file_path``.
        RuntimeError: If the database connection is not configured, or if
            the active embedding driver's dimension doesn't match the
            existing document_chunks.embedding column.
        ValueError: If the file's extension is unsupported, or the file has
            no extractable content (e.g. empty, or a scanned PDF with no
            text layer).
    """
    doc_path = Path(file_path)
    source_file = doc_path.name

    print(f"[ingest] Starting ingestion: {source_file}")

    # Step 1: Pick the extractor for this file's format and validate it
    extractor = get_extractor(doc_path)
    extractor.validate(doc_path)

    # Step 2: Get the driver up front — CHUNKING_STRATEGY=langchain needs it
    # (token limit/counting) *during* chunking, not just for embedding after.
    driver = get_embedding_driver()
    store = VectorStore()
    store.assert_dimension_matches(driver.dimension)

    # Step 3: Concatenate the whole document, then chunk it document-wide
    full_text, word_page_map = extractor.extract(doc_path, mode=settings.PDF_EXTRACTION_MODE)
    chunks = chunk_document(full_text, word_page_map, source_file=source_file, driver=driver)
    print(f"[ingest] Created {len(chunks)} chunk(s) via CHUNKING_STRATEGY="
          f"'{settings.CHUNKING_STRATEGY}'.")

    pre_overflow_count = len(chunks)
    chunks = get_chunk_overflow_strategy().apply(chunks, driver)
    if len(chunks) != pre_overflow_count:
        print(
            f"[ingest] CHUNK_OVERFLOW_STRATEGY=split corrected "
            f"{pre_overflow_count} chunk(s) into {len(chunks)}."
        )

    # Step 4: Embed all chunks in one batched call
    texts = [c["content"] for c in chunks]
    print(f"[ingest] Embedding with driver='{settings.EMBEDDING_DRIVER}' ...")
    embeddings = driver.embed_batch(texts)
    print(f"[ingest] Embeddings ready. Dimension: {len(embeddings[0])}.")

    # Step 5: Store in Postgres
    inserted = store.save(chunks, embeddings)
    print(f"[ingest] Stored {inserted} row(s) in document_chunks. Done! ✅")
