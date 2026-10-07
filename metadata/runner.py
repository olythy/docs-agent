"""Runs metadata extraction over the documents that still need it.

The orchestrator behind ``scripts/meta_cli.py extract-meta``: for each document
with pending keys it fetches the chunks, lets the right source extract each key,
verifies LLM-sourced values against their quotes, and records both the values and
a status per key -- including *absent* and *unverified*, so a later count can say
"+K unknown" instead of silently dropping documents. It is idempotent and
resumable: it only touches (document, key) pairs without a status at the key's
current version.

The two API calls per document (choosing the evidence, asking the model) dominate the
run time and only wait for the network, so ``workers`` > 1 runs them for several
documents at once. The database is **never** touched from a worker: the main thread
reads a document's chunks, hands the API work to a worker, and writes the result when it
comes back, so a worker count changes the speed and nothing else (the stored values and
the report are the same for any count).

Key exports:
    MetaExtractionRunner -- The orchestrator.
    RunReport            -- What a run did.
"""

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
from retry_policy import TransientAPIError

logger = logging.getLogger(__name__)


class DocumentRepository(Protocol):
    """The slice of :class:`document_store.DocumentStore` the runner uses."""

    def list_keys(
        self, doc_type: str, status: KeyStatus | None = None
    ) -> list[MetaKey]: ...
    def documents_needing(
        self,
        keys: list[MetaKey],
        doc_type: str,
        limit: int | None = None,
        seed: int | None = None,
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
        failed_files: The documents whose model or rerank call kept failing after
            every retry (their keys are counted in ``failed`` and stay pending).
    """

    documents: int = 0
    present: int = 0
    confirmed_absent: int = 0
    unverified: int = 0
    failed: int = 0
    proposed_keys: int = 0
    failed_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Plan:
    """What a worker needs for one document, read beforehand on the main thread."""

    document: Document
    bodies: list[RetrievedChunk]
    groups: list[tuple[MetaSource, list[MetaKey]]]


@dataclass(frozen=True)
class _Outcome:
    """What the API work for one document came to: the extractions, or why it gave up."""

    extractions: "list[_Extraction]"
    error: TransientAPIError | None = None


@dataclass(frozen=True)
class _Extraction:
    """What one source returned for one group of keys of a document."""

    source: MetaSource
    keys: list[MetaKey]
    shown: list[RetrievedChunk]
    result: SourceResult
    verifier: EvidenceVerifier | None


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
        workers: How many documents' API calls run at once (default 1: one after
            the other). A provider's rate limit is handled by the drivers' retries.

    Raises:
        ValueError: If ``workers`` is less than 1.
    """

    def __init__(
        self,
        documents: DocumentRepository,
        chunks: ChunkRepository,
        selector: EvidenceSelector,
        sources: Sequence[MetaSource],
        date_parser: DateParser | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        workers: int = 1,
    ) -> None:
        if workers < 1:
            raise ValueError(f"workers must be at least 1, got {workers}")
        self._workers = workers
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
            doc_type: The catalog to use; only documents of this type are processed.
            limit: Process at most this many documents (for a trial run).
            seed: With ``limit``, take a random sample fixed by this seed instead
                of the first documents by file name (which is rarely
                representative: files are named by court).

        Returns:
            The counts of what happened.
        """
        keys = self._documents.list_keys(doc_type, KeyStatus.APPROVED)
        todo = self._documents.documents_needing(keys, doc_type, limit, seed)
        if self._workers > 1:
            return self._run_concurrently(todo, keys, doc_type)
        report = RunReport()
        for index, document in enumerate(todo, start=1):
            report = self._merge(report, self._process(document, keys, doc_type))
            if self._on_progress:
                self._on_progress(index, len(todo))
        return report

    def _run_concurrently(
        self, todo: list[Document], keys: list[MetaKey], doc_type: str
    ) -> RunReport:
        """Like the plain loop, with the API work of several documents in flight.

        A document's chunks are read on this thread just before it is handed to a
        worker, and only ``2 * workers`` documents are in flight, so memory does not
        grow with the corpus. Results are written here, in the order they finish.
        An unexpected error in a worker (a bug, a bad login) stops the run, as it does
        without workers: what was already written stays, and the next run continues
        from there. A provider that keeps failing for one document does not (see
        :meth:`_extract`).
        """
        report = RunReport()
        done = 0
        in_flight: dict[Future[_Outcome], _Plan] = {}
        remaining = iter(todo)
        exhausted = False

        def finished_one() -> None:
            nonlocal done
            done += 1
            if self._on_progress:
                self._on_progress(done, len(todo))

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            while True:
                while not exhausted and len(in_flight) < 2 * self._workers:
                    document = next(remaining, None)
                    if document is None:
                        exhausted = True
                        break
                    plan = self._prepare(document, keys)
                    if plan is None:  # nothing stored for this document
                        finished_one()
                        continue
                    in_flight[pool.submit(self._extract, plan)] = plan
                if not in_flight:
                    return report
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    plan = in_flight.pop(future)
                    try:
                        outcome = future.result()
                    except BaseException:
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise
                    report = self._merge(report, self._commit(plan, outcome, doc_type))
                    finished_one()

    def _process(
        self, document: Document, keys: list[MetaKey], doc_type: str
    ) -> RunReport:
        """One document, start to finish, on this thread."""
        plan = self._prepare(document, keys)
        if plan is None:
            return RunReport()
        return self._commit(plan, self._extract(plan), doc_type)

    def _prepare(self, document: Document, keys: list[MetaKey]) -> "_Plan | None":
        """Read what a document needs (database reads; never called from a worker)."""
        statuses = self._documents.get_statuses(document.content_hash)
        pending = [
            k
            for k in keys
            if k.key not in statuses or statuses[k.key].key_version < k.version
        ]
        raw = self._chunks.get_document_chunks(document.content_hash)
        if not raw:
            return None
        bodies = [
            replace(c, content=strip_chunk_prefixes(c.content, c.metadata)) for c in raw
        ]
        return _Plan(document, bodies, self._assign(pending))

    def _extract(self, plan: "_Plan") -> "_Outcome":
        """The API work for one document: choose the evidence and ask the sources.

        Touches no database, so it is safe to run on a worker thread. A provider that
        keeps failing after every retry (:class:`retry_policy.TransientAPIError`) is
        a failure of *this document*, not of the run: it is returned as such, counted
        and named in the report, and the document stays pending for the next run.
        Any other error is not caught: it is a bug or a bad login, and hiding it
        would make a broken run look like a slow one.
        """
        extractions = []
        try:
            for source, source_keys in plan.groups:
                shown = (
                    self._selector.select(plan.bodies, source_keys)
                    if source.needs_verification
                    else plan.bodies
                )
                verifier = (
                    EvidenceVerifier(
                        "\n".join(c.content for c in shown), self._date_parser
                    )
                    if source.needs_verification
                    else None
                )
                extractions.append(
                    _Extraction(
                        source,
                        source_keys,
                        shown,
                        source.extract(shown, source_keys),
                        verifier,
                    )
                )
        except TransientAPIError as error:
            return _Outcome([], error)
        return _Outcome(extractions)

    def _commit(self, plan: "_Plan", outcome: "_Outcome", doc_type: str) -> RunReport:
        """Record what the sources returned (database writes; main thread only)."""
        if outcome.error is not None:
            logger.warning(
                "[extract] %s: the provider kept failing (%s); nothing was stored "
                "for it and the next run will try it again.",
                plan.document.source_file,
                outcome.error,
            )
            return RunReport(
                documents=1,
                failed=sum(len(keys) for _, keys in plan.groups),
                failed_files=(plan.document.source_file,),
            )
        counts = {
            "present": 0,
            "absent": 0,
            "unverified": 0,
            "failed": 0,
            "proposed": 0,
        }
        for e in outcome.extractions:
            for key in e.keys:
                self._record(
                    plan.document, key, e.source, e.result, e.shown, e.verifier, counts
                )
            counts["proposed"] += self._store_proposals(e.result, doc_type)
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
            failed_files=a.failed_files + b.failed_files,
        )
