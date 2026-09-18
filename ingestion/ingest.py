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

import logging
from pathlib import Path

from config import settings
from drivers.embedding import get_embedding_driver
from ingestion.chunker import chunk_document, get_chunk_overflow_strategy
from ingestion.extractors import get_extractor
from store import VectorStore

# Progress logging, not print(): add_document() is called from mcp_server.py
# over an MCP stdio transport, where stray stdout writes can corrupt the
# JSON-RPC protocol stream — confirmed empirically (a print() here broke a
# real client's message parsing). logging defaults to stderr, which is safe
# for every caller (CLI scripts, agent.py, mcp_server.py alike).
logger = logging.getLogger(__name__)


def add_document(file_path: str | Path, force: bool = False) -> None:
    """Ingest a document into the RAG knowledge base.

    This is the main tool exposed to the agent. It runs the full pipeline:
    extract → chunk → embed → store. The document's format is detected from
    its extension (see :func:`ingestion.extractors.get_extractor`).

    Args:
        file_path: Path to the source document (str or Path).
        force: Skip the already-ingested check and ingest anyway. Since
            there's no way to identify/replace a document's *previous*
            chunks (they aren't keyed by content hash), this adds a second,
            duplicate copy rather than updating the existing one — only use
            this if that's genuinely what's wanted.

    Raises:
        FileNotFoundError: If the file does not exist at ``file_path``.
        RuntimeError: If the database connection is not configured, or if
            the active embedding driver's dimension doesn't match the
            existing document_chunks.embedding column.
        ValueError: If the file's extension is unsupported, the file has no
            extractable content (e.g. empty, or a scanned PDF with no text
            layer), or (unless ``force=True``) a document with this same
            filename is already in the knowledge base — found the hard way:
            re-ingesting the same file twice (nothing here prevented it)
            silently doubled its chunks, which then crowded out other,
            genuinely relevant chunks from a real query's top-k results.
    """
    doc_path = Path(file_path)
    source_file = doc_path.name

    logger.info("[ingest] Starting ingestion: %s", source_file)

    # Step 1: Pick the extractor for this file's format and validate it
    extractor = get_extractor(doc_path)
    extractor.validate(doc_path)

    # Step 2: Get the driver up front — CHUNKING_STRATEGY=langchain needs it
    # (token limit/counting) *during* chunking, not just for embedding after.
    driver = get_embedding_driver()
    store = VectorStore()
    store.assert_dimension_matches(driver.dimension)

    if not force and store.has_chunks_from_source(source_file):
        raise ValueError(
            f"'{source_file}' is already in the knowledge base. Pass "
            "force=True to ingest it again anyway (this adds a duplicate "
            "copy of its chunks, it does not replace the existing ones)."
        )

    # Step 3: Concatenate the whole document, then chunk it document-wide
    full_text, word_page_map = extractor.extract(doc_path, mode=settings.PDF_EXTRACTION_MODE)
    chunks = chunk_document(full_text, word_page_map, source_file=source_file, driver=driver)
    logger.info(
        "[ingest] Created %d chunk(s) via CHUNKING_STRATEGY='%s'.",
        len(chunks),
        settings.CHUNKING_STRATEGY,
    )

    pre_overflow_count = len(chunks)
    chunks = get_chunk_overflow_strategy().apply(chunks, driver)
    if len(chunks) != pre_overflow_count:
        logger.info(
            "[ingest] CHUNK_OVERFLOW_STRATEGY=split corrected %d chunk(s) into %d.",
            pre_overflow_count,
            len(chunks),
        )

    # Step 4: Embed all chunks in one batched call
    texts = [c["content"] for c in chunks]
    logger.info("[ingest] Embedding with driver='%s' ...", settings.EMBEDDING_DRIVER)
    embeddings = driver.embed_batch(texts)
    logger.info("[ingest] Embeddings ready. Dimension: %d.", len(embeddings[0]))

    # Step 5: Store in Postgres
    inserted = store.save(chunks, embeddings)
    logger.info("[ingest] Stored %d row(s) in document_chunks. Done! ✅", inserted)
