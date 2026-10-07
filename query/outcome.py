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

#: Said when no chunk was relevant enough. Callers (the agent layer, the eval's decline
#: detection) test for this wording, so it is one constant.
NO_RESULTS_MESSAGE = (
    "I could not find relevant information about this in the provided documents."
)

#: Said when the request is of a kind the system cannot do yet; the planner's reason
#: (in the question's language) follows it.
NOT_SUPPORTED_MESSAGE = (
    "This kind of question is not supported yet, so I will not guess at an answer."
)

#: Said when the planner cannot turn the question into a valid plan.
COULD_NOT_INTERPRET_MESSAGE = (
    "I could not interpret this question well enough to answer it from the "
    "structured data, and I did not want to guess."
)


def no_matching_documents_message(explanation: str, note: str | None = None) -> str:
    """Said when the question's filters select no document.

    The wording that ``query.decline_detection`` recognises ("no documents match").

    Args:
        explanation: The executed filter in words.
        note: A caveat that goes with it (documents that could not be checked).
    """
    return f"No documents match the filter ({explanation})." + (
        f" {note}" if note else ""
    )


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


class RefusalRenderer:
    """Turns a :class:`Declined` into the sentence a person is told.

    The wording is the original system's, word for word: the eval's decline detection and
    the adversarial questions depend on it. Both retrieval gates say the same thing to the
    person (nothing relevant); they differ only in the explain record.
    """

    def render(self, declined: Declined) -> str:
        """The refusal as text."""
        match declined.reason:
            case DeclineReason.COULD_NOT_INTERPRET:
                return COULD_NOT_INTERPRET_MESSAGE
            case DeclineReason.NOT_SUPPORTED:
                if declined.detail:
                    return f"{NOT_SUPPORTED_MESSAGE} ({declined.detail})"
                return NOT_SUPPORTED_MESSAGE
            case DeclineReason.NO_MATCHING_DOCUMENTS:
                return no_matching_documents_message(
                    declined.detail or "", declined.note
                )
            case DeclineReason.NOT_RELEVANT | DeclineReason.RERANK_REJECTED:
                return NO_RESULTS_MESSAGE
