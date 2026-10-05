"""Runs document-type classification over the documents that have no type yet.

The orchestrator behind ``scripts/meta_cli.py classify-documents``: for each
unclassified document it builds the text the classifier sees (the summary and the
opening passages), asks the classifier, verifies the quote it gave, and records
the type. It is idempotent and resumable (only documents without a type are
touched) and it never hides a failure: an unusable answer, a quote that is not in
the text, or an unknown type leaves the document unclassified and is counted, so
the next run retries it.

A proposed new type is stored as ``proposed``; the document points at it, but
queries cannot use it until a person approves it.

Key exports:
    ClassificationRunner -- The orchestrator.
    ClassificationReport -- What a run did.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ingestion.chunker import strip_chunk_prefixes
from metadata.classifier import Classification, TypeClassifier
from metadata.verification import EvidenceVerifier
from models import Document, DocumentType, RetrievedChunk, TypeStatus

#: How much of a document's opening the classifier is shown.
HEAD_CHARS = 2000

_TYPE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class TypeRepository(Protocol):
    """The slice of :class:`document_store.DocumentStore` the runner uses."""

    def list_types(self, status: TypeStatus | None = None) -> list[DocumentType]: ...
    def get_type(self, type_name: str) -> DocumentType | None: ...
    def upsert_type(self, doc_type: DocumentType) -> None: ...
    def unclassified_documents(
        self, limit: int | None = None, seed: int | None = None
    ) -> list[Document]: ...
    def set_document_type(self, content_hash: str, type_name: str) -> bool: ...


class ChunkRepository(Protocol):
    """The slice of :class:`store.VectorStore` the runner uses."""

    def get_document_chunks(self, content_hash: str) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class ClassificationReport:
    """What a run did.

    Attributes:
        documents: Documents looked at.
        classified: Documents given an existing type.
        proposed: Documents pointed at a *proposed* type (new, or proposed earlier).
        new_types: Types created as ``proposed`` during the run.
        failed: Documents left unclassified, with the reason counted below.
        reasons: ``reason -> count`` of the failures (an unusable answer, a quote
            not found in the text, an unknown or retired type, no text).
    """

    documents: int = 0
    classified: int = 0
    proposed: int = 0
    new_types: int = 0
    failed: int = 0
    reasons: tuple[tuple[str, int], ...] = ()


class ClassificationRunner:
    """Gives every unclassified document a type.

    Args:
        documents: Where types and documents live.
        chunks: Where a document's chunks are read from.
        classifier: Decides a document's type.
        head_chars: How much of the document's opening the classifier sees.
        on_progress: Called after each document with ``(done, total)``.
    """

    def __init__(
        self,
        documents: TypeRepository,
        chunks: ChunkRepository,
        classifier: TypeClassifier,
        head_chars: int = HEAD_CHARS,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self._documents = documents
        self._chunks = chunks
        self._classifier = classifier
        self._head_chars = head_chars
        self._on_progress = on_progress

    def run(
        self, limit: int | None = None, seed: int | None = None
    ) -> ClassificationReport:
        """Classify up to ``limit`` documents that have no type.

        Args:
            limit: At most this many documents (for a trial run).
            seed: With ``limit``, a random sample fixed by this seed instead of the
                first documents by file name.
        """
        todo = self._documents.unclassified_documents(limit, seed)
        counts = {"classified": 0, "proposed": 0, "new_types": 0}
        reasons: dict[str, int] = {}
        for index, document in enumerate(todo, start=1):
            outcome = self._process(document)
            if outcome in counts:
                counts[outcome] += 1
            elif outcome == "new_proposed":
                counts["proposed"] += 1
                counts["new_types"] += 1
            else:
                reasons[outcome] = reasons.get(outcome, 0) + 1
            if self._on_progress:
                self._on_progress(index, len(todo))
        return ClassificationReport(
            documents=len(todo),
            classified=counts["classified"],
            proposed=counts["proposed"],
            new_types=counts["new_types"],
            failed=sum(reasons.values()),
            reasons=tuple(sorted(reasons.items())),
        )

    def _process(self, document: Document) -> str:
        """Classify one document; return an outcome name or a failure reason."""
        text = self._head(document)
        if not text:
            return "no text to read"
        usable = [
            t
            for t in self._documents.list_types()
            if t.status is not TypeStatus.RETIRED
        ]
        result = self._classifier.classify(text, usable)
        if result.failed:
            return "unusable answer"
        if not EvidenceVerifier(text).contains(result.evidence):
            return "quote not found in the text"
        if result.type is not None:
            return self._assign(document, result.type, usable)
        return self._assign_proposal(document, result, usable)

    def _head(self, document: Document) -> str:
        """The summary and the opening passages, up to ``head_chars`` of the latter."""
        chunks = sorted(
            self._chunks.get_document_chunks(document.content_hash),
            key=lambda c: c.metadata.chunk_index,
        )
        body = ""
        for chunk in chunks:
            if len(body) >= self._head_chars:
                break
            body += strip_chunk_prefixes(chunk.content, chunk.metadata) + "\n"
        body = body[: self._head_chars].strip()
        if not body:
            return ""
        return f"{document.summary}\n\n{body}" if document.summary else body

    def _assign(
        self, document: Document, type_name: str, usable: list[DocumentType]
    ) -> str:
        chosen = next((t for t in usable if t.type == type_name), None)
        if chosen is None:
            return "unknown or retired type"
        self._documents.set_document_type(document.content_hash, chosen.type)
        return "proposed" if chosen.status is TypeStatus.PROPOSED else "classified"

    def _assign_proposal(
        self, document: Document, result: Classification, usable: list[DocumentType]
    ) -> str:
        proposal = result.proposal
        assert proposal is not None  # a non-failed result has a type or a proposal
        if not _TYPE_NAME.match(proposal.type):
            return "proposed type is not English snake_case"
        if any(t.type == proposal.type for t in usable):
            return self._assign(document, proposal.type, usable)
        if self._documents.get_type(proposal.type) is not None:
            return "unknown or retired type"  # a retired type is not revived
        self._documents.upsert_type(
            DocumentType(
                proposal.type, proposal.name, proposal.description, TypeStatus.PROPOSED
            )
        )
        self._documents.set_document_type(document.content_hash, proposal.type)
        return "new_proposed"
