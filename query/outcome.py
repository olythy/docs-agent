"""How a retrieval ends: with chunks to answer from, or with an explicit refusal.

An empty list used to stand for "nothing relevant", whichever stage had decided it.
Here a refusal is a value that says *which stage* refused and *why*, so the caller,
the explain record and the tests never have to guess. Wording for the user is not
decided here (a renderer will map a reason to its text).

Key exports:
    DeclineReason -- Why a question was not answered.
    Declined      -- A refusal, attributed to a stage.
    Answerable    -- Chunks to answer from.
"""

from dataclasses import dataclass
from enum import StrEnum

from models import RetrievedChunk


class DeclineReason(StrEnum):
    """Why a question was not answered."""

    COULD_NOT_INTERPRET = "could_not_interpret"
    NOT_SUPPORTED = "not_supported"
    NO_MATCHING_DOCUMENTS = "no_matching_documents"
    NOT_RELEVANT = "not_relevant"  # the cosine relevance gate
    RERANK_REJECTED = "rerank_rejected"  # the reranker's score gate


@dataclass(frozen=True)
class Declined:
    """A refusal.

    Attributes:
        reason: Why.
        stage: The step or phase that refused (e.g. ``relevance_gate``, ``planning``).
        detail: What the user should be told in addition (e.g. the planner's reason).
        note: A caveat that goes with the refusal.
    """

    reason: DeclineReason
    stage: str
    detail: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class Answerable:
    """Chunks an answer can be written from, best first.

    Attributes:
        chunks: The final context.
    """

    chunks: tuple[RetrievedChunk, ...]
