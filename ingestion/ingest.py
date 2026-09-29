"""Document ingestion pipeline.

Orchestrates the document ingestion flow:
    - ``add_document``: Ingest a single file (PDF, Markdown, DOCX, or RTF).
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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from config import settings
from drivers.embedding import get_embedding_driver
from ingestion.chunker import chunk_document, get_chunk_overflow_strategy
from ingestion.extractors import (
    SUPPORTED_EXTENSIONS,
    get_extractor,
    normalize_extensions,
)
from ingestion.hash import compute_file_hash
from logger import LogAction, get_logger
from store import VectorStore

# Progress logging, not print(): add_document() is called from mcp_server.py
# over an MCP stdio transport, where stray stdout writes can corrupt the
# JSON-RPC protocol stream — confirmed empirically (a print() here broke a
# real client's message parsing). logging defaults to stderr, which is safe
# for every caller (CLI scripts, agent.py, mcp_server.py alike).
logger = logging.getLogger(__name__)


class IngestAction(Enum):
    """What add_document/add_directory should do next for one candidate file.

    See :func:`_resolve_ingest_action` for how this is decided.
    """

    SKIP_UNCHANGED = "skip_unchanged"
    ALIAS_EXISTING_CONTENT = "alias_existing_content"
    REPLACE_PREVIOUS_VERSION = "replace_previous_version"
    FORCE_REPLACE_DUPLICATE = "force_replace_duplicate"
    INSERT_NEW = "insert_new"


@dataclass(frozen=True)
class IngestDecision:
    """The resolved action, plus which content_hash's chunks (if any) to delete first."""

    action: IngestAction
    hash_to_delete: str | None = None


def _resolve_ingest_action(
    existing_hash: str | None,
    hash_already_stored: bool,
    content_hash: str,
    force: bool,
) -> IngestDecision:
    """Decides what to do with one candidate file, given two already-run lookups.

    Pure decision logic, no I/O of its own — extracted out of add_document/
    add_directory so the four dedup/versioning branches (skip an unchanged
    re-ingest, alias identical content under a new path, replace a
    previous version, or insert fresh) are each independently testable,
    instead of being interleaved with extraction/chunking/embedding.

    Args:
        existing_hash: The content_hash currently stored for this file's
            source path, or None if this source path has no chunks yet
            (:meth:`store.VectorStore.get_hash_by_source`).
        hash_already_stored: Whether any chunk anywhere already has this
            exact content_hash, regardless of source path
            (:meth:`store.VectorStore.has_content_hash`).
        content_hash: This file's own SHA-256 content hash.
        force: Whether the caller passed ``force=True``.

    Returns:
        An :class:`IngestDecision` telling the caller which of the five
        :class:`IngestAction` cases applies, and which hash's chunks (if
        any) must be deleted before inserting the new ones.
    """
    if not force:
        if existing_hash == content_hash:
            return IngestDecision(IngestAction.SKIP_UNCHANGED)
        if hash_already_stored and existing_hash is None:
            return IngestDecision(IngestAction.ALIAS_EXISTING_CONTENT)

    if existing_hash:
        return IngestDecision(
            IngestAction.REPLACE_PREVIOUS_VERSION, hash_to_delete=existing_hash
        )
    if force and hash_already_stored:
        return IngestDecision(
            IngestAction.FORCE_REPLACE_DUPLICATE, hash_to_delete=content_hash
        )

    return IngestDecision(IngestAction.INSERT_NEW)


def add_document(
    file_path: str | Path,
    force: bool = False,
    store: VectorStore | None = None,
    source_path: str | None = None,
) -> None:
    """Ingest a document into the RAG knowledge base.

    This is the main tool exposed to the agent. It runs the full pipeline:
    extract → chunk → embed → store. The document's format is detected from
    its extension (see :func:`ingestion.extractors.get_extractor`).

    Performs content-addressable integrity and deduplication checks via SHA-256
    hash:
        - If the document path exists with an identical content hash, skips it
          (unless ``force=True``).
        - If the document path exists with a *different* content hash (a new version),
          it cleanly replaces the old chunks with the new ones.
        - If an identical content hash already exists under a different path,
          skips to avoid polluting the retrieval index with duplicates.

    Args:
        file_path: Path to the source document (str or Path).
        force: Skip the already-ingested checks and re-ingest anyway. When True,
            replaces existing chunks for this document.
        store: Optional :class:`store.VectorStore` instance. If omitted,
            instantiates a fresh one.
        source_path: Optional logical path identity of the document (e.g.
            relative path in a directory tree). Defaults to ``str(file_path)``.

    Raises:
        FileNotFoundError: If the file does not exist at ``file_path``.
        RuntimeError: If the database connection is not configured, or if
            the active embedding driver's dimension doesn't match the
            existing document_chunks.embedding column.
        ValueError: If the file's extension is unsupported, the file has no
            extractable content, or (unless ``force=True``) the document/content
            is already present in the knowledge base.
    """
    doc_path = Path(file_path)
    if not doc_path.exists():
        raise FileNotFoundError(f"File not found: {doc_path}")

    source_file = doc_path.name
    effective_source_path = source_path if source_path is not None else str(doc_path)
    content_hash = compute_file_hash(doc_path)

    logger.info(
        "[ingest] Starting ingestion: %s (%s)", effective_source_path, content_hash[:8]
    )

    # Step 1: Pick the extractor for this file's format and validate it
    extractor = get_extractor(doc_path)
    extractor.validate(doc_path)

    # Step 2: Get the driver up front — CHUNKING_STRATEGY=langchain needs it
    # (token limit/counting) *during* chunking, not just for embedding after.
    driver = get_embedding_driver()
    store = store if store is not None else VectorStore()

    with store:
        store.assert_dimension_matches(driver.dimension)

        existing_hash = store.get_hash_by_source(effective_source_path)
        hash_already_stored = store.has_content_hash(content_hash)
        decision = _resolve_ingest_action(
            existing_hash, hash_already_stored, content_hash, force
        )

        if decision.action is IngestAction.SKIP_UNCHANGED:
            raise ValueError(
                f"'{effective_source_path}' is already in the knowledge base with "
                "identical content. Pass force=True to re-index it anyway."
            )
        if decision.action is IngestAction.ALIAS_EXISTING_CONTENT:
            # Content already exists under another source path — register alias without re-embedding
            store.add_source_alias(content_hash, effective_source_path)
            logger.info(
                "[ingest] Content already indexed (hash %s). Added '%s' as alias.",
                content_hash[:8],
                effective_source_path,
            )
            return

        if decision.hash_to_delete:
            deleted = store.delete_chunks_by_hash(decision.hash_to_delete)
            if deleted > 0 and decision.action is IngestAction.REPLACE_PREVIOUS_VERSION:
                logger.info(
                    "[ingest] Replaced %d existing chunk(s) for previous version of '%s' (hash %s).",
                    deleted,
                    effective_source_path,
                    decision.hash_to_delete[:8],
                )
            elif deleted > 0:
                logger.info(
                    "[ingest] Force-removed %d existing chunk(s) for hash %s.",
                    deleted,
                    decision.hash_to_delete[:8],
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
            source_path=effective_source_path,
            content_hash=content_hash,
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
        texts = [c.content for c in chunks]
        logger.info(
            "[ingest] Embedding with driver='%s' ...", settings.EMBEDDING_DRIVER
        )
        embeddings = driver.embed_documents(texts)
        logger.info("[ingest] Embeddings ready. Dimension: %d.", len(embeddings[0]))

        # Step 5: Store in Postgres
        inserted = store.save(chunks, embeddings)
        logger.info("[ingest] Stored %d row(s) in document_chunks. Done! ✅", inserted)
        get_logger().log(
            LogAction.DOCUMENT_INGESTED,
            {
                "path": effective_source_path,
                "chunks_count": inserted,
                "content_hash": content_hash[:8],
                "embedding_driver": settings.EMBEDDING_DRIVER,
            },
        )


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
        "updated": [],
        "aliased": [],
        "skipped": [],
        "failed": [],
        "total_found": len(files),
    }

    if not files:
        logger.info("[ingest] No supported documents found in %s", path)
        return summary

    driver = get_embedding_driver()
    store = VectorStore()

    with store:
        store.assert_dimension_matches(driver.dimension)

        for doc_file in files:
            rel_path = str(doc_file.relative_to(path))
            try:
                content_hash = compute_file_hash(doc_file)
                existing_hash = store.get_hash_by_source(rel_path)
                hash_already_stored = store.has_content_hash(content_hash)
                decision = _resolve_ingest_action(
                    existing_hash, hash_already_stored, content_hash, force
                )

                if decision.action is IngestAction.SKIP_UNCHANGED:
                    logger.info("[ingest] Skipping '%s' (unchanged)", rel_path)
                    summary["skipped"].append(str(doc_file))
                    continue
                if decision.action is IngestAction.ALIAS_EXISTING_CONTENT:
                    store.add_source_alias(content_hash, rel_path)
                    logger.info(
                        "[ingest] Aliased '%s' to existing content (hash %s)",
                        rel_path,
                        content_hash[:8],
                    )
                    summary["aliased"].append(str(doc_file))
                    continue

                add_document(doc_file, force=force, store=store, source_path=rel_path)
                if existing_hash and existing_hash != content_hash:
                    summary["updated"].append(str(doc_file))
                else:
                    summary["ingested"].append(str(doc_file))
            except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
                logger.warning("[ingest] Failed to ingest '%s': %s", doc_file, exc)
                summary["failed"].append({"file": str(doc_file), "error": str(exc)})

    logger.info(
        "[ingest] Finished directory '%s': %d ingested, %d updated, %d aliased, %d skipped, %d failed.",
        path.name,
        len(summary["ingested"]),
        len(summary["updated"]),
        len(summary["aliased"]),
        len(summary["skipped"]),
        len(summary["failed"]),
    )
    return summary
