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
    RetrievedChunk -- One chunk returned by a search, with its id and score.
    Document, MetaKey, MetaValue, MetaStatus -- The structured-metadata layer
        (see docs/structured-metadata-design.md): a document's identity, the
        key catalog, an extracted value with its evidence, and what is known
        about a (document, key) pair.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum


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
        hub_score: This chunk's average cosine similarity to its nearest
            neighbors in the *whole corpus*, independent of any specific
            query -- set by ``store.VectorStore.compute_hub_scores()`` (a
            post-ingest batch pass), ``None`` until that's run. A high
            score means the chunk's embedding sits in a "generic"/central
            region of the embedding space (many other chunks look similar
            to it); used to penalize generic chunks at query time (see
            ``query.retrieval.steps.candidates.CslsReorderStep``) without ever excluding them
            outright -- see ``docs/decisions.md`` for why a hard exclusion
            threshold was tried first and rejected.
        document_summary: A short, LLM-generated, fact-focused summary of
            the whole source document (see
            ``ingestion.summarize.generate_document_summary``), extracted
            once per document and embedded into *every* chunk's content
            (same "embed into every chunk" mechanism as
            ``document_identifiers``, since the point is to give the
            embedding a distinguishing signal regardless of which chunk a
            later query happens to match). Confirmed live (see
            docs/decisions.md): must ask for the document's own
            case-specific facts explicitly, not a generic topic
            restatement -- a generic summary makes near-duplicate
            documents' embeddings *more* alike, not less.
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
    hub_score: float | None = None
    document_summary: str | None = None

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
        if self.hub_score is not None:
            data["hub_score"] = self.hub_score
        if self.document_summary is not None:
            data["document_summary"] = self.document_summary
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
            hub_score=data.get("hub_score"),
            document_summary=data.get("document_summary"),
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
            (e.g. :func:`query.retrieval.hybrid.reciprocal_rank_fusion`) identify the
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


class ValueType(StrEnum):
    """The type of a metadata key's values; selects which ``value_*`` column holds them.

    ``IDENTIFIER`` is stored exactly like ``TEXT`` (``value_text``): the type tells the
    system that the text is an identifier, to be compared by the rule in
    :mod:`metadata.identifiers`."""

    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    BOOL = "bool"
    IDENTIFIER = "identifier"  # text stored as written, compared in a normalised form


class KeyStatus(StrEnum):
    """Lifecycle of a catalog key; only ``APPROVED`` keys are usable in queries."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    RETIRED = "retired"


class MetaSource(StrEnum):
    """Where a metadata value came from."""

    DETERMINISTIC = "deterministic"
    LLM = "llm"
    SIDECAR = "sidecar"


class MetaState(StrEnum):
    """What is known about a (document, key) pair.

    ``PRESENT`` extracted and verified; ``CONFIRMED_ABSENT`` the document was
    examined and does not state it; ``UNVERIFIED`` extracted but the evidence
    did not check out (never used by queries); ``NOT_ATTEMPTED`` no attempt yet.
    A count over a key must report everything that is not ``PRESENT`` or
    ``CONFIRMED_ABSENT`` as unknown.
    """

    PRESENT = "present"
    CONFIRMED_ABSENT = "confirmed_absent"
    UNVERIFIED = "unverified"
    NOT_ATTEMPTED = "not_attempted"


@dataclass(frozen=True)
class Document:
    """A document's identity: the root every extracted value hangs off.

    Attributes:
        content_hash: SHA-256 of the whole document (the primary key).
        source_file: Basename of the source file.
        summary: The per-document summary, if one was generated.
        ingested_at: When the document was (re)ingested, if known.
    """

    content_hash: str
    source_file: str
    summary: str | None = None
    ingested_at: datetime | None = None


@dataclass(frozen=True)
class DocumentSelection:
    """The documents a structured filter selects, as a sub-select (not a list).

    A restricted search puts ``document_id IN (<sql>)`` in its WHERE clause, so the
    database does the filtering and no list of documents ever travels: a list
    capped at a few thousand hashes cannot express "the 800,000 invoices of 2024".

    Attributes:
        sql: A ``SELECT d.id FROM documents d WHERE ...`` query with ``%s``
            placeholders, built by the plan compiler (never from user text).
        params: Its bound parameters, in order.
    """

    sql: str
    params: tuple = ()


class TypeStatus(StrEnum):
    """Whether a document type may be used.

    ``PROPOSED`` types (suggested by a classifier, not yet reviewed) are not
    usable in queries; ``RETIRED`` types were withdrawn or merged into another.
    """

    PROPOSED = "proposed"
    APPROVED = "approved"
    RETIRED = "retired"


@dataclass(frozen=True)
class DocumentType:
    """One kind of document the system knows (a court decision, an invoice, ...).

    Attributes:
        type: English snake_case identifier; what documents and keys refer to.
        name: Human-readable name.
        description: What the classifier and the planner read to decide whether a
            document or a question belongs to this type.
        status: Whether the type is usable.
    """

    type: str
    name: str
    description: str
    status: TypeStatus = TypeStatus.PROPOSED


@dataclass(frozen=True)
class MetaKey:
    """One entry of the key catalog for a document type.

    Attributes:
        doc_type: The kind of document the key applies to.
        key: English snake_case name.
        value_type: Which typed column holds its values.
        description: What the extractor reads to decide whether the key fits.
        example: An illustrative value.
        allowed_values: For categorical keys, the canonical English tokens.
        multi_valued: Whether several rows per document are expected.
        status: ``PROPOSED`` keys are not usable in queries.
        version: Bumped when the description changes, so stale values can be found.
    """

    doc_type: str
    key: str
    value_type: ValueType
    description: str
    example: str | None = None
    allowed_values: tuple[str, ...] | None = None
    multi_valued: bool = False
    status: KeyStatus = KeyStatus.PROPOSED
    version: int = 1


@dataclass(frozen=True)
class MetaValue:
    """One extracted value of one key of one document, with its evidence.

    Exactly one of the ``value_*`` fields is set (the database enforces it).

    Attributes:
        content_hash: The document.
        key: The catalog key.
        key_version: The key description version that produced this value.
        source: Where it came from.
        ordinal: Position among a multi-valued key's rows (0 for single-valued).
        qualifiers: Role / party / instance etc., as the catalog allows.
        evidence: The verbatim quote that shows the value.
        evidence_chunk_index: Which chunk of the document the quote is in.
        page: Page or section number of that chunk, if known.
    """

    content_hash: str
    key: str
    key_version: int
    source: MetaSource
    value_text: str | None = None
    value_number: Decimal | None = None
    value_date: date | None = None
    value_bool: bool | None = None
    unit: str | None = None
    ordinal: int = 0
    qualifiers: dict[str, str] = field(default_factory=dict)
    evidence: str | None = None
    evidence_chunk_index: int | None = None
    page: int | None = None


@dataclass(frozen=True)
class MetaStatus:
    """What is known about one (document, key) pair; see :class:`MetaState`."""

    content_hash: str
    key: str
    state: MetaState
    key_version: int
