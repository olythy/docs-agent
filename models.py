"""Core, framework-free data shapes shared across the ingestion and retrieval pipelines.

Replaces the loose ``dict`` shape chunks used to be passed around as
(``{"content": ..., "metadata": {...}}``) with typed, strictly-annotated
dataclasses — cheap insurance against typo'd dict keys as the corpus (and
its metadata: header paths, section breadcrumbs, page numbers) grows more
complex, without introducing a full Entity/Use-Case layered architecture
the project's linear pipeline doesn't need (see AGENTS.md).

Key exports:
    ChunkMetadata  -- Everything stored in document_chunks.metadata (JSONB).
    Chunk          -- One chunk before storage: content + ChunkMetadata.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata attached to one chunk, mirroring the document_chunks.metadata JSONB column.

    ``source_path``/``sources``/``content_hash``/``header_path`` are
    optional because :func:`ingestion.chunker.chunk_pages` (the legacy
    per-page chunker, still used by ``scripts/eval_cli.py``'s inspect/
    extract tooling and its own tests) only ever sets
    ``source_file``/``page_number``/``chunk_index`` — the production path,
    :func:`ingestion.chunker.chunk_document`, sets all of them.

    Attributes:
        source_file: Basename of the source document (e.g. ``"invoice.pdf"``).
        page_number: Which page (PDF) or header-based section index
            (Markdown) this chunk mostly came from, or ``None`` if
            undeterminable (e.g. an empty page map).
        chunk_index: 0-based position of this chunk within its document.
        source_path: Logical path identity of the source document (e.g.
            ``"finance/2024/report.pdf"``), used for dedup/versioning
            lookups (see ``ingestion.ingest._resolve_ingest_action``).
        sources: Every source path whose content hashes to this chunk's
            content (see ``store.VectorStore.add_source_alias``).
        content_hash: Hexadecimal SHA-256 digest of the whole source
            document this chunk was produced from.
        header_path: Hierarchical Markdown heading breadcrumb (e.g.
            ``"# Chapter 1 > ## Section 1.1"``), if any.
        document_identifiers: Code-like identifier tokens (case numbers,
            invoice numbers, ...) found near the start of the source
            document (see ``ingestion.chunker.extract_document_identifiers``),
            embedded into every chunk's content so a later identifier-based
            query can find them regardless of which chunk actually holds
            the relevant content -- not just chunk 0, where the document's
            header line originally appeared.
        document_date: The source document's own date of issue (e.g. a
            Hungarian court decision's "kelt" date), normalized to
            ISO-8601 (``"YYYY-MM-DD"``), or ``None`` if the extractor
            couldn't find one (see
            ``ingestion.chunker.extract_document_date`` -- confirmed live
            this misses on roughly 10% of real corpus documents).
            Deliberately *not* embedded into chunk content -- see
            ``docs/decisions.md``'s 2026-10-03 entry for why: unlike
            ``document_identifiers``, nothing does a literal text match
            against a date, so it only needs to be a filterable/
            query-time field, and embedding it would push near-duplicate
            boilerplate documents' vectors even closer together.
    """

    source_file: str
    page_number: int | None
    chunk_index: int
    source_path: str | None = None
    sources: tuple[str, ...] = ()
    content_hash: str | None = None
    header_path: str | None = None
    document_identifiers: tuple[str, ...] = ()
    document_date: str | None = None

    def to_dict(self) -> dict:
        """Convert to the JSON-serializable dict shape stored in Postgres.

        Only includes optional fields that are actually set, matching the
        dict this project built by hand before this dataclass existed — so
        the JSONB column's shape (and every existing row) stays unchanged.
        """
        data: dict = {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }
        if self.source_path is not None:
            data["source_path"] = self.source_path
        if self.sources:
            data["sources"] = list(self.sources)
        if self.content_hash is not None:
            data["content_hash"] = self.content_hash
        if self.header_path is not None:
            data["header_path"] = self.header_path
        if self.document_identifiers:
            data["document_identifiers"] = list(self.document_identifiers)
        if self.document_date is not None:
            data["document_date"] = self.document_date
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkMetadata":
        """Reconstruct from the dict shape stored in Postgres (or built by hand elsewhere)."""
        return cls(
            source_file=data["source_file"],
            page_number=data.get("page_number"),
            chunk_index=data["chunk_index"],
            source_path=data.get("source_path"),
            sources=tuple(data.get("sources", ())),
            content_hash=data.get("content_hash"),
            header_path=data.get("header_path"),
            document_identifiers=tuple(data.get("document_identifiers", ())),
            document_date=data.get("document_date"),
        )


@dataclass(frozen=True)
class Chunk:
    """One chunk before storage — what the chunker and overflow strategy produce.

    Attributes:
        content: The chunk's text, ready for embedding.
        metadata: This chunk's :class:`ChunkMetadata`.
    """

    content: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        """Convert to the ``{"content": ..., "metadata": {...}}`` dict shape ``store.VectorStore.save`` writes."""
        return {"content": self.content, "metadata": self.metadata.to_dict()}


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk after retrieval — what ``store.VectorStore.search``/``search_fulltext`` return.

    Attributes:
        id: The row's ``document_chunks.id`` primary key — lets callers
            (e.g. :func:`query.hybrid.reciprocal_rank_fusion`) identify the
            *same* chunk across separate result sets (vector vs. keyword
            search), since two different rows could coincidentally share
            identical text.
        content: The chunk's text.
        metadata: This chunk's :class:`ChunkMetadata`.
        score: Similarity score. Scale depends on the source — cosine
            similarity (0-1) from ``search()``, ``ts_rank`` (unbounded)
            from ``search_fulltext()`` — never compare the two directly,
            only their *ranks* (exactly what RRF fusion does).
    """

    id: int
    content: str
    metadata: ChunkMetadata
    score: float

    def to_dict(self) -> dict:
        """Convert to the ``{"id": ..., "content": ..., "metadata": {...}, "score": ...}`` dict shape used before this dataclass existed."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "score": self.score,
        }
