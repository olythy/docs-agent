"""Runs metadata extraction over the documents that still need it.

The orchestrator behind ``scripts/meta_cli.py extract-meta``: for each document
with pending keys it fetches the chunks, lets the right source extract each key,
verifies LLM-sourced values against their quotes, and records both the values and
a status per key -- including *absent* and *unverified*, so a later count can say
"+K unknown" instead of silently dropping documents. It is idempotent and
resumable: it only touches (document, key) pairs without a status at the key's
current version.

Key exports:
    MetaExtractionRunner -- The orchestrator.
    RunReport            -- What a run did.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from ingestion.chunker import strip_chunk_prefixes
from metadata.conversion import to_meta_value
from metadata.date_parsers import DateParser
from metadata.evidence import EvidenceSelector
from metadata.sources import MetaSource, SourceResult
from metadata.verification import EvidenceVerifier
from models import (
    Document,
    KeyStatus,
    MetaKey,
    MetaState,
    MetaStatus,
    MetaValue,
    RetrievedChunk,
    ValueType,
)


class DocumentRepository(Protocol):
    """The slice of :class:`document_store.DocumentStore` the runner uses."""

    def list_keys(
        self, doc_type: str, status: KeyStatus | None = None
    ) -> list[MetaKey]: ...
    def documents_needing(
        self, keys: list[MetaKey], limit: int | None = None, seed: int | None = None
    ) -> list[Document]: ...
    def get_statuses(self, content_hash: str) -> dict[str, MetaStatus]: ...
    def replace_values(
        self, content_hash: str, key: str, values: list[MetaValue]
    ) -> None: ...
    def set_status(self, status: MetaStatus) -> None: ...
    def upsert_key(self, key: MetaKey) -> None: ...


class ChunkRepository(Protocol):
    """The slice of :class:`store.VectorStore` the runner uses."""

    def get_document_chunks(self, content_hash: str) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class RunReport:
    """What a run did, counted over (document, key) outcomes.

    Attributes:
        documents: Documents processed.
        present: Keys extracted and (where required) verified.
        confirmed_absent: Keys the extractor looked for and did not find.
        unverified: Keys whose value could not be confirmed against its quote.
        failed: Keys left pending because the source gave no usable answer.
        proposed_keys: New keys the extractor suggested (stored as ``proposed``).
    """

    documents: int = 0
    present: int = 0
    confirmed_absent: int = 0
    unverified: int = 0
    failed: int = 0
    proposed_keys: int = 0


class MetaExtractionRunner:
    """Extracts the pending metadata keys of the documents that need them.

    Args:
        documents: Where documents, the catalog, values and status live.
        chunks: Where a document's chunks are read from.
        selector: Chooses which chunks an LLM source is shown.
        sources: Candidate sources, in priority order; a key goes to the first
            source that supports it.
        date_parser: How dates are written in this corpus (for verification).
        on_progress: Called after each document with ``(done, total)``.
    """

    def __init__(
        self,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        selector: EvidenceSelector,
        sources: Sequence[MetaSource],
        date_parser: DateParser | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self._documents = documents
        self._chunks = chunks
        self._selector = selector
        self._sources = list(sources)
        self._date_parser = date_parser
        self._on_progress = on_progress

    def run(
        self, doc_type: str, limit: int | None = None, seed: int | None = None
    ) -> RunReport:
        """Extract every pending approved key for up to ``limit`` documents.

        Args:
            doc_type: The catalog to use.
            limit: Process at most this many documents (for a trial run).
            seed: With ``limit``, take a random sample fixed by this seed instead
                of the first documents by file name (which is rarely
                representative: files are named by court).

        Returns:
            The counts of what happened.
        """
        keys = self._documents.list_keys(doc_type, KeyStatus.APPROVED)
        todo = self._documents.documents_needing(keys, limit, seed)
        report = RunReport()
        for index, document in enumerate(todo, start=1):
            report = self._merge(report, self._process(document, keys, doc_type))
            if self._on_progress:
                self._on_progress(index, len(todo))
        return report

    def _process(
        self, document: Document, keys: list[MetaKey], doc_type: str
    ) -> RunReport:
        statuses = self._documents.get_statuses(document.content_hash)
        pending = [
            k
            for k in keys
            if k.key not in statuses or statuses[k.key].key_version < k.version
        ]
        raw = self._chunks.get_document_chunks(document.content_hash)
        if not raw:
            return RunReport()
        bodies = [
            replace(c, content=strip_chunk_prefixes(c.content, c.metadata)) for c in raw
        ]

        counts = {
            "present": 0,
            "absent": 0,
            "unverified": 0,
            "failed": 0,
            "proposed": 0,
        }
        for source, source_keys in self._assign(pending):
            shown = (
                self._selector.select(bodies, source_keys)
                if source.needs_verification
                else bodies
            )
            result = source.extract(shown, source_keys)
            verifier = (
                EvidenceVerifier("\n".join(c.content for c in shown), self._date_parser)
                if source.needs_verification
                else None
            )
            for key in source_keys:
                self._record(document, key, source, result, shown, verifier, counts)
            counts["proposed"] += self._store_proposals(result, doc_type)
        return RunReport(
            documents=1,
            present=counts["present"],
            confirmed_absent=counts["absent"],
            unverified=counts["unverified"],
            failed=counts["failed"],
            proposed_keys=counts["proposed"],
        )

    def _assign(self, keys: list[MetaKey]) -> list[tuple[MetaSource, list[MetaKey]]]:
        """Give each key to the first source that supports it."""
        groups: dict[int, list[MetaKey]] = {}
        for key in keys:
            for position, source in enumerate(self._sources):
                if source.supports(key):
                    groups.setdefault(position, []).append(key)
                    break
        return [(self._sources[p], ks) for p, ks in groups.items()]

    def _record(
        self,
        document: Document,
        key: MetaKey,
        source: MetaSource,
        result: SourceResult,
        shown: list[RetrievedChunk],
        verifier: EvidenceVerifier | None,
        counts: dict[str, int],
    ) -> None:
        if result.failed:
            counts["failed"] += 1  # no status: it stays pending and is retried next run
            return
        found = [c for c in result.candidates if c.key == key.key]
        if not key.multi_valued:
            found = found[:1]
        if not found:
            self._set(document, key, MetaState.CONFIRMED_ABSENT)
            counts["absent"] += 1
            return
        values: list[MetaValue] = []
        for ordinal, candidate in enumerate(found):
            page = next(
                (
                    c.metadata.page_number
                    for c in shown
                    if c.metadata.chunk_index == candidate.evidence_chunk_index
                ),
                None,
            )
            typed = to_meta_value(
                key, candidate, document.content_hash, source.kind, ordinal, page
            )
            confirmed = typed is not None and (
                verifier is None
                or verifier.verify(key, candidate.value, candidate.evidence).verified
            )
            if typed is None or not confirmed:
                self._documents.replace_values(document.content_hash, key.key, [])
                self._set(document, key, MetaState.UNVERIFIED)
                counts["unverified"] += 1
                return
            values.append(typed)
        self._documents.replace_values(document.content_hash, key.key, values)
        self._set(document, key, MetaState.PRESENT)
        counts["present"] += 1

    def _set(self, document: Document, key: MetaKey, state: MetaState) -> None:
        self._documents.set_status(
            MetaStatus(document.content_hash, key.key, state, key.version)
        )

    def _store_proposals(self, result: SourceResult, doc_type: str) -> int:
        """Keep a suggested key as ``proposed`` (never usable in a query until approved)."""
        known = {k.key for k in self._documents.list_keys(doc_type)}
        stored = 0
        for proposal in result.proposed_keys:
            if proposal.key in known:
                continue
            try:
                value_type = ValueType(proposal.value_type)
            except ValueError:
                value_type = ValueType.TEXT
            self._documents.upsert_key(
                MetaKey(
                    doc_type,
                    proposal.key,
                    value_type,
                    proposal.description or "(proposed)",
                )
            )
            known.add(proposal.key)
            stored += 1
        return stored

    @staticmethod
    def _merge(a: RunReport, b: RunReport) -> RunReport:
        return RunReport(
            documents=a.documents + b.documents,
            present=a.present + b.present,
            confirmed_absent=a.confirmed_absent + b.confirmed_absent,
            unverified=a.unverified + b.unverified,
            failed=a.failed + b.failed,
            proposed_keys=a.proposed_keys + b.proposed_keys,
        )
