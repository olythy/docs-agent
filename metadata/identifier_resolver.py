"""Resolving the identifiers a question names to the documents that carry them.

A question can name a case number ("what did the court decide in 4.P.20.409/2023/4").
The metadata already holds each document's identifiers (keys of type ``identifier``), so
the question's identifier can be turned into the documents it belongs to **before** the
retrieval, and the retrieval can then run only over those: nothing has to be pinned
ahead of relevance, and a number that belongs to no document can be said plainly.

It looks in up to three steps, each only for an identifier the earlier ones found nothing for:

1. the identifier as written (equal after normalisation, or continued by a suffix);
2. the identifier as a *part* of a stored one, because people often give only the end of a
   composite identifier and leave out the leading series or office (see
   :func:`metadata.identifiers.identifier_contains`);
3. the identifier with its separators ignored, because it may be written with other
   separators than the stored one ("P.20103.2022.19" for "4.P.20.103/2022/19"; see
   :func:`metadata.identifiers.identifier_compact_contains`).

Only an identifier that says enough (:func:`metadata.identifiers.is_specific_enough`) is
looked for in steps 2 and 3, and what was found there is reported by how it was found.

The resolver does not decide what to do with the answer. It reports, per identifier, the
documents (none, one or several: a number can be a case number in one document and an
invoice number in another, and nothing here picks between them; the ranking decides by
content), which identifiers resolved only partially, and which to nothing.

Key exports:
    ResolvedIdentifiers -- What the identifiers resolved to.
    IdentifierSource    -- Where documents are looked up.
    IdentifierResolver  -- The resolver.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from metadata.identifiers import (
    compact_pattern,
    is_specific_enough,
    normalize_identifier,
    partial_pattern,
)


class MatchKind(StrEnum):
    """How an identifier was found when it was not found as written."""

    PARTIAL = "partial"  # the end or a middle part of a stored identifier
    COMPACT = "compact"  # equal once separators are ignored


@dataclass(frozen=True)
class ResolvedIdentifiers:
    """What the identifiers of a question resolved to.

    Attributes:
        documents: Each identifier as it was asked, with the ids of the documents that
            carry it (sorted; empty when none does).
        kinds: For an identifier (as asked) that was not found as written, how it was found.
    """

    documents: Mapping[str, tuple[int, ...]]
    kinds: Mapping[str, MatchKind] = field(default_factory=dict)

    def _of(self, kind: MatchKind) -> tuple[str, ...]:
        return tuple(i for i in self.documents if self.kinds.get(i) is kind)

    @property
    def partial(self) -> tuple[str, ...]:
        """The identifiers resolved only as a part of a stored one, in the order asked."""
        return self._of(MatchKind.PARTIAL)

    @property
    def compact(self) -> tuple[str, ...]:
        """The identifiers resolved only with their separators ignored."""
        return self._of(MatchKind.COMPACT)

    @property
    def approximate(self) -> tuple[str, ...]:
        """Every identifier that was not found as written but was found."""
        return tuple(i for i in self.documents if i in self.kinds)

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

    def documents_matching_identifier_patterns(
        self, patterns: Sequence[str], ignoring_separators: bool = False
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
        found = self._exact(wanted)

        # An identifier not found as written may be the end of a longer one ...
        specific = [n for n in wanted if not found[n] and is_specific_enough(n)]
        partially = self._by_pattern(specific, partial_pattern, False)
        # ... or be written with other separators.
        remaining = [n for n in specific if not partially[n]]
        compactly = self._by_pattern(remaining, compact_pattern, True)

        documents: dict[str, tuple[int, ...]] = {}
        kinds: dict[str, MatchKind] = {}
        for raw, n in normalised.items():
            ids = found.get(n) or partially.get(n) or compactly.get(n) or set()
            documents[raw] = tuple(sorted(ids))
            if not found.get(n) and partially.get(n):
                kinds[raw] = MatchKind.PARTIAL
            elif not found.get(n) and compactly.get(n):
                kinds[raw] = MatchKind.COMPACT
        return ResolvedIdentifiers(documents, kinds)

    def _exact(self, wanted: list[str]) -> dict[str, set[int]]:
        """Step 1: the identifiers as written."""
        found: dict[str, set[int]] = defaultdict(set)
        if wanted:  # nothing to look up for punctuation alone, so no round trip
            for position, document_id in self._source.documents_with_identifiers(
                wanted
            ):
                found[wanted[position - 1]].add(document_id)
        return found

    def _by_pattern(
        self, wanted: list[str], make_pattern, ignoring_separators: bool
    ) -> dict[str, set[int]]:
        """A later step: the identifiers matched by a pattern each."""
        found: dict[str, set[int]] = defaultdict(set)
        if wanted:
            patterns = [make_pattern(n) for n in wanted]
            rows = self._source.documents_matching_identifier_patterns(
                patterns, ignoring_separators
            )
            for position, document_id in rows:
                found[wanted[position - 1]].add(document_id)
        return found
