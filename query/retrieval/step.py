"""The contract of a step of the retrieval pipeline.

A step does one thing to a :class:`query.retrieval.context.RetrievalContext`: it reads some slots
and fills others, and either continues with the changed context (:class:`Continue`) or
refuses the question (:class:`Halt`). It declares the slots it ``requires`` and
``provides``, so a chain that cannot work (a step reads what no earlier step wrote) is
refused when it is built, not found at run time.

A step never reads ``Settings``: its parameters and collaborators come in through its
constructor.

Key exports:
    RetrievalStep       -- The contract every step implements.
    Continue, Halt    -- What a step returns.
    AuxRecord         -- A side result a step reports (a year pool, the identifier matches).
    StepName          -- Stable names of the steps (the trace and the funnel use them).
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from models import RetrievedChunk
from query.outcome import Declined
from query.retrieval.context import RetrievalContext, Slot


class StepName(StrEnum):
    """Names of the steps that reproduce the original retrieval."""

    EMBED_QUERY = "embed_query"
    DENSE_SEARCH = "dense_search"
    RELEVANCE_GATE = "relevance_gate"
    YEAR_DENSE_WIDENING = "year_dense_widening"
    CSLS_REORDER = "csls_reorder"
    KEYWORD_SEARCH = "keyword_search"
    YEAR_KEYWORD_WIDENING = "year_keyword_widening"
    RRF_FUSION = "rrf_fusion"
    RERANK = "rerank"
    RERANK_SCORE_GATE = "rerank_score_gate"
    LISTWISE_RERANK = "listwise_rerank"
    TOP_K_SELECTION = "top_k_selection"


@dataclass(frozen=True)
class AuxRecord:
    """A side result a step reports for the trace, outside the context.

    Attributes:
        label: What it is (e.g. ``year_pool``, ``identifier_matches``).
        chunks: The chunks.
    """

    label: str
    chunks: tuple[RetrievedChunk, ...]


@dataclass(frozen=True)
class Continue:
    """A step finished; the chain goes on with ``context``.

    Attributes:
        context: The changed context.
        records: Side results for the trace.
        notes: Small facts about the step (e.g. whether a gate passed).
    """

    context: RetrievalContext
    records: tuple[AuxRecord, ...] = ()
    notes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Halt:
    """A step refused; the chain stops and the question is declined.

    Attributes:
        declined: The refusal.
        records: Side results for the trace.
        notes: Small facts about the step.
    """

    declined: Declined
    records: tuple[AuxRecord, ...] = ()
    notes: Mapping[str, object] = field(default_factory=dict)


StepResult = Continue | Halt


class RetrievalStep(ABC):
    """One step of the retrieval pipeline.

    Subclasses set the three descriptive attributes and implement :meth:`run`.

    Attributes:
        name: Stable name (see :class:`StepName`).
        requires: Slots it reads; an earlier step must provide them.
        provides: Slots it fills (it must fill all of them when it continues).
    """

    name: str
    requires: frozenset[Slot] = frozenset()
    provides: frozenset[Slot] = frozenset()

    @abstractmethod
    def run(self, context: RetrievalContext) -> StepResult:
        """Do the step.

        Args:
            context: What the earlier steps left.

        Returns:
            :class:`Continue` with the changed context, or :class:`Halt` to refuse.
        """
