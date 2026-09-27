"""Document ingestion pipeline.

Orchestrates the document ingestion flow:
    - ``add_document``: Ingest a single file (PDF or Markdown).
    - ``add_directory``: Batch-ingest all qualifying files in a directory.

This module exposes two public functions: :func:`add_document` and :func:`add_directory`.

Usage::

    from ingestion.ingest import add_document, add_directory
    add_document("/path/to/document.pdf")
    add_document("/path/to/notes.md")
    add_directory("/path/to/docs/", recursive=True)
"""

import logging
from collections.abc import Collection
from pathlib import Path

from config import settings
from drivers.embedding import get_embedding_driver
from ingestion.chunker import chunk_document, get_chunk_overflow_strategy
from ingestion.extractors import (
    SUPPORTED_EXTENSIONS,
    get_extractor,
    normalize_extensions,
)
from store import VectorStore

# Progress logging, not print(): add_document() is called from mcp_server.py
# over an MCP stdio transport, where stray stdout writes can corrupt the
# JSON-RPC protocol stream — confirmed empirically (a print() here broke a
# real client's message parsing). logging defaults to stderr, which is safe
# for every caller (CLI scripts, agent.py, mcp_server.py alike).
logger = logging.getLogger(__name__)


def add_document(
    file_path: str | Path,
    force: bool = False,
    store: VectorStore | None = None,
) -> None:
    """Ingest a document into the RAG knowledge base.

    This is the main tool exposed to the agent. It runs the full pipeline:
    extract → chunk → embed → store. The document's format is detected from
    its extension (see :func:`ingestion.extractors.get_extractor`).

    Args:
        file_path: Path to the source document (str or Path).
        force: Skip the already-ingested check and ingest anyway. Since
            this replaces any existing chunks from this file rather than
            creating duplicates.
        store: Optional :class:`store.VectorStore` instance. If omitted,
            instantiates a fresh one.

    Raises:
        FileNotFoundError: If the file does not exist at ``file_path``.
        RuntimeError: If the database connection is not configured, or if
            the active embedding driver's dimension doesn't match the
            existing document_chunks.embedding column.
        ValueError: If the file's extension is unsupported, the file has no
            extractable content (e.g. empty, or a scanned PDF with no text
            layer), or (unless ``force=True``) a document with this same
            filename is already in the knowledge base.
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
    store = store if store is not None else VectorStore()
    store.assert_dimension_matches(driver.dimension)

    if not force and store.has_chunks_from_source(source_file):
        raise ValueError(
            f"'{source_file}' is already in the knowledge base. Pass "
            "force=True to re-ingest it (this replaces any existing chunks "
            "from this file)."
        )
    if force:
        deleted = store.delete_chunks_from_source(source_file)
        if isinstance(deleted, int) and deleted > 0:
            logger.info(
                "[ingest] Replaced %d existing chunk(s) for '%s'.",
                deleted,
                source_file,
            )

    # Step 3: Concatenate the whole document, then chunk it document-wide
    try:
        full_text, word_page_map, word_header_map = extractor.extract_with_headers(
            doc_path, mode=settings.PDF_EXTRACTION_MODE
        )
    except (TypeError, ValueError):
        full_text, word_page_map = extractor.extract(
            doc_path, mode=settings.PDF_EXTRACTION_MODE
        )
        word_header_map = None
    chunks = chunk_document(
        full_text,
        word_page_map,
        source_file=source_file,
        driver=driver,
        word_header_map=word_header_map,
    )
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
    embeddings = driver.embed_documents(texts)
    logger.info("[ingest] Embeddings ready. Dimension: %d.", len(embeddings[0]))

    # Step 5: Store in Postgres
    inserted = store.save(chunks, embeddings)
    logger.info("[ingest] Stored %d row(s) in document_chunks. Done! ✅", inserted)


def add_directory(
    dir_path: str | Path,
    recursive: bool = True,
    force: bool = False,
    allowed_extensions: Collection[str] | None = None,
) -> dict:
    """Batch-ingest all qualifying documents from a directory into the knowledge base.

    Scans ``dir_path`` for files with allowed extensions, ignoring hidden files
    and directories (names starting with '.'). For each qualifying document,
    attempts ingestion via :func:`add_document`.

    Unlike :func:`add_document`, which raises immediately when a document is
    already present (unless ``force=True``) or invalid, ``add_directory`` is
    designed for batch resilience: it records skipped or failed files and
    continues processing the rest of the directory, returning an overall summary.

    Args:
        dir_path: Path to the directory (str or Path).
        recursive: Whether to search subdirectories recursively (default: True).
        force: If True, replaces existing chunks for all files instead of
            skipping them.
        allowed_extensions: Optional collection of permitted extensions (e.g.
            ``{".md"}``). If omitted, defaults to the system configuration
            (``settings.parsed_ingest_extensions & SUPPORTED_EXTENSIONS``).

    Returns:
        A dict with the batch ingestion summary:
            - ``"ingested"``: list of successfully ingested file paths.
            - ``"skipped"``: list of files skipped because they are already present.
            - ``"failed"``: list of dicts with ``"file"`` and ``"error"`` message.
            - ``"total_found"``: total count of qualifying files discovered.

    Raises:
        FileNotFoundError: If ``dir_path`` does not exist.
        NotADirectoryError: If ``dir_path`` is not a directory.
    """
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")

    if allowed_extensions is not None:
        normalized = normalize_extensions(allowed_extensions)
        effective_allowed = normalized & SUPPORTED_EXTENSIONS
        unsupported = normalized - SUPPORTED_EXTENSIONS
        if unsupported:
            logger.warning(
                "[ingest] Extension(s) %s have no registered extractor and will be ignored.",
                sorted(unsupported),
            )
    else:
        effective_allowed = settings.parsed_ingest_extensions & SUPPORTED_EXTENSIONS
        unsupported = settings.parsed_ingest_extensions - SUPPORTED_EXTENSIONS
        if unsupported:
            logger.warning(
                "[ingest] Extension(s) %s in INGEST_EXTENSIONS have no registered extractor and will be ignored.",
                sorted(unsupported),
            )

    iterator = path.rglob("*") if recursive else path.glob("*")
    files: list[Path] = []
    for item in iterator:
        if not item.is_file():
            continue
        # Skip hidden files or files inside hidden subdirectories (.git, .venv, etc.)
        if any(part.startswith(".") for part in item.relative_to(path).parts):
            continue
        if item.suffix.lower() in effective_allowed:
            files.append(item)

    files.sort()

    summary: dict = {
        "ingested": [],
        "skipped": [],
        "failed": [],
        "total_found": len(files),
    }

    if not files:
        logger.info("[ingest] No supported documents found in %s", path)
        return summary

    driver = get_embedding_driver()
    store = VectorStore()
    store.assert_dimension_matches(driver.dimension)

    for doc_file in files:
        source_name = doc_file.name
        if not force and store.has_chunks_from_source(source_name):
            logger.info(
                "[ingest] Skipping '%s' (already in knowledge base)", source_name
            )
            summary["skipped"].append(str(doc_file))
            continue

        try:
            add_document(doc_file, force=force, store=store)
            summary["ingested"].append(str(doc_file))
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            logger.warning("[ingest] Failed to ingest '%s': %s", doc_file, exc)
            summary["failed"].append({"file": str(doc_file), "error": str(exc)})

    logger.info(
        "[ingest] Finished directory '%s': %d ingested, %d skipped, %d failed.",
        path.name,
        len(summary["ingested"]),
        len(summary["skipped"]),
        len(summary["failed"]),
    )
    return summary
