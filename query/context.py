"""The context that travels through the retrieval pipeline.

A question about documents is answered by a short chain of steps (search, fuse, rerank,
select ...) that share one frozen :class:`RetrievalContext`: a fixed part (the facts, the
metadata filter) and named *slots* that the steps fill. A slot is named for what it
holds, not for the phase that produced it; each slot has one writing step, and a new
need adds a slot. The steps' contract is in :mod:`query.step`.

Key exports:
    Slot, CHUNK_SLOTS -- The named parts of the context.
    RetrievalContext    -- The context.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum

from models import RetrievedChunk
from query.facts import QueryFacts


class Slot(StrEnum):
    """The named parts of the context a step can fill (each value is a field name)."""

    QUERY_VECTOR = "query_vector"
    DENSE_POOL = "dense_pool"
    KEYWORD_POOL = "keyword_pool"
    RANKED = "ranked"
    PINS = "pins"
    SELECTED = "selected"


#: The slots that hold passages (the others hold a vector and a set of ids).
CHUNK_SLOTS = frozenset(
    {Slot.DENSE_POOL, Slot.KEYWORD_POOL, Slot.RANKED, Slot.SELECTED}
)


@dataclass(frozen=True)
class RetrievalContext:
    """Everything the steps share. Frozen: a step returns a changed copy.

    Attributes:
        facts: The facts of the question.
        metadata_filter: A key/value restriction on chunk metadata, passed to the
            searches that accept it.
        query_vector: The embedded question (may be supplied by the caller).
        dense_pool: Candidates from the vector search (later widened and reordered).
        keyword_pool: Candidates from the full-text search.
        ranked: The candidates in the order the ranking steps leave them.
        pins: Ids of chunks that must survive the cut (exact identifier matches).
        selected: The final context passed to the answer.
    """

    facts: QueryFacts
    metadata_filter: Mapping[str, object] | None = None
    query_vector: tuple[float, ...] | None = None
    dense_pool: tuple[RetrievedChunk, ...] | None = None
    keyword_pool: tuple[RetrievedChunk, ...] | None = None
    ranked: tuple[RetrievedChunk, ...] | None = None
    pins: frozenset[int] | None = None
    selected: tuple[RetrievedChunk, ...] | None = None

    def value(self, slot: Slot) -> object:
        """The content of ``slot`` (``None`` while no step has filled it)."""
        return getattr(self, slot.value)

    def with_slots(self, **slots: object) -> "RetrievalContext":
        """A copy with the given slots (by field name) replaced."""
        return replace(self, **slots)  # type: ignore[arg-type]
