"""Resolving the identifiers a question names to the documents that carry them.

A question can name a case number ("what did the court decide in 4.P.20.409/2023/4").
The metadata already holds each document's identifiers (keys of type ``identifier``), so
the question's identifier can be turned into the documents it belongs to **before** the
retrieval, and the retrieval can then run only over those: nothing has to be pinned
ahead of relevance, and a number that belongs to no document can be said plainly.

The resolver does not decide what to do with the answer. It reports, per identifier, the
documents (none, one or several: a number can be a case number in one document and an
invoice number in another, and nothing here picks between them; the ranking decides by
content) and which identifiers resolved to nothing.

Key exports:
    ResolvedIdentifiers -- What the identifiers resolved to.
    IdentifierSource    -- Where documents are looked up.
    IdentifierResolver  -- The resolver.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from metadata.identifiers import normalize_identifier


@dataclass(frozen=True)
class ResolvedIdentifiers:
    """What the identifiers of a question resolved to.

    Attributes:
        documents: Each identifier as it was asked, with the ids of the documents that
            carry it (sorted; empty when none does).
    """

    documents: Mapping[str, tuple[int, ...]]

    @property
    def document_ids(self) -> tuple[int, ...]:
        """Every resolved document, once, sorted (the union over the identifiers)."""
        return tuple(sorted({d for ids in self.documents.values() for d in ids}))

    @property
    def unresolved(self) -> tuple[str, ...]:
        """The identifiers that no document carries, in the order they were asked."""
        return tuple(i for i, ids in self.documents.items() if not ids)

    @property
    def ambiguous(self) -> tuple[str, ...]:
        """The identifiers that more than one document carries."""
        return tuple(i for i, ids in self.documents.items() if len(ids) > 1)


class IdentifierSource(Protocol):
    """The slice of :class:`document_store.DocumentStore` the resolver uses."""

    def documents_with_identifiers(
        self, wanted: Sequence[str]
    ) -> list[tuple[int, int]]: ...


class IdentifierResolver:
    """Finds the documents that carry the identifiers a question names.

    Args:
        source: Where documents are looked up (their identifiers are compared by the rule of
            :mod:`metadata.identifiers`).
    """

    def __init__(self, source: IdentifierSource) -> None:
        self._source = source

    def resolve(self, identifiers: Sequence[str]) -> ResolvedIdentifiers:
        """Resolve ``identifiers``, one lookup for all of them.

        Args:
            identifiers: As they were written in the question. One that is nothing but
                punctuation cannot name anything and resolves to no document.

        Returns:
            Every identifier given, in order, with its documents.
        """
        normalised = {raw: normalize_identifier(raw) for raw in identifiers}
        wanted = list(dict.fromkeys(n for n in normalised.values() if n))
        found: dict[str, set[int]] = defaultdict(set)
        if wanted:  # nothing to look up for punctuation alone, so no round trip
            for position, document_id in self._source.documents_with_identifiers(
                wanted
            ):
                found[wanted[position - 1]].add(document_id)
        return ResolvedIdentifiers(
            {raw: tuple(sorted(found.get(n, ()))) for raw, n in normalised.items()}
        )
