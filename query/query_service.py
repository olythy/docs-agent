"""Answering a question end to end: decide, then carry the decision out.

:class:`QueryService` is the one entry point of the new pipeline. It owns no logic of its
own about *what* to do (that is the decider's) nor *how* (the retrieval, the answerers and the
refusal renderer do it); it connects them in the one order there is:

    facts -> decision -> refuse | answer exactly | retrieve -> (refuse | write the answer)

Every answer comes back with an :class:`Explain`: the decision that was taken, what the
retrieval did, who refused. Nothing the person is told is only in a log.

Key exports:
    Answer       -- The text, with its explanation.
    Explain      -- How the answer came about.
    QueryService -- Answers a question.
"""

import logging
from dataclasses import dataclass, replace
from typing import Protocol

from metadata.plan import QueryPlan
from models import RetrievedChunk
from query.answering import GroundedAnswer
from query.decision import AnswerExactly, Decider, Decision, ReadDocuments, Refuse
from query.facts import QueryFactsReader
from query.outcome import Answerable, Declined, RefusalRenderer
from query.service import RetrievalRequest, RetrievalResult
from store import VectorStore

logger = logging.getLogger(__name__)


class Retrieves(Protocol):
    """Reads the best chunks (:class:`query.service.RetrievalService`)."""

    def retrieve(
        self, request: RetrievalRequest, store: VectorStore
    ) -> RetrievalResult: ...


class AnswersExactly(Protocol):
    """Answers from the metadata alone (:class:`query.answering.ExactAnswerer`)."""

    def answer(self, question: str, plan: QueryPlan) -> str: ...


class AnswersFromChunks(Protocol):
    """Writes an answer from chunks (:class:`query.answering.GroundedAnswerer`)."""

    def answer(
        self, question: str, chunks: tuple[RetrievedChunk, ...]
    ) -> GroundedAnswer: ...


@dataclass(frozen=True)
class Explain:
    """How an answer came about.

    Attributes:
        decision: What was decided for the question.
        retrieval: What the retrieval did; ``None`` when no documents were read.
        declined: Who refused and why; ``None`` when the question was answered.
        model_refused: The model itself answered with its refusal sentence.
    """

    decision: Decision
    retrieval: RetrievalResult | None = None
    declined: Declined | None = None
    model_refused: bool = False


@dataclass(frozen=True)
class Answer:
    """What the person is told, and how it came about.

    Attributes:
        text: The answer, with any caveat of the scope appended.
        explain: How it came about.
    """

    text: str
    explain: Explain


class QueryService:
    """Answers a question.

    Args:
        facts_reader: Reads the identifiers and years of the question.
        decider: Decides how the question is answered.
        retrieval: Reads the best chunks of the documents in scope.
        exact: Answers from the metadata alone (``None`` when no planner decides, since
            then no question is answered exactly).
        grounded: Writes an answer from the chunks read.
        refusals: Words a refusal.
    """

    def __init__(
        self,
        facts_reader: QueryFactsReader,
        decider: Decider,
        retrieval: Retrieves,
        exact: AnswersExactly | None,
        grounded: AnswersFromChunks,
        refusals: RefusalRenderer,
    ) -> None:
        self._facts_reader = facts_reader
        self._decider = decider
        self._retrieval = retrieval
        self._exact = exact
        self._grounded = grounded
        self._refusals = refusals

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        profile: str | None = None,
        store: VectorStore | None = None,
    ) -> Answer:
        """Answer ``question``.

        Args:
            question: The user's question.
            top_k: Overrides ``RETRIEVAL_TOP_K``.
            min_score: Overrides ``RETRIEVAL_MIN_SCORE``.
            profile: Reads with this profile instead of the one the decision chose (for
                comparing profiles; it changes nothing about the decision itself).
            store: The chunk store (default: the configured one).

        Raises:
            RuntimeError: If no document type is approved (the planner was switched on
                before ``load-catalog``), or the embedding dimension does not match the store.
            metadata.plan.PlanError: If an exact plan no longer fits the catalog.
        """
        decision = self._decider.decide(self._facts_reader.read(question))
        match decision:
            case Refuse(declined=declined):
                return Answer(
                    self._refusals.render(declined),
                    Explain(decision, declined=declined),
                )
            case AnswerExactly(plan=plan):
                if self._exact is None:
                    raise RuntimeError(
                        "An exact answer was decided but no ExactAnswerer is set."
                    )
                return Answer(self._exact.answer(question, plan), Explain(decision))
            case ReadDocuments():
                if profile is not None:
                    decision = replace(decision, profile=profile)
                return self._read(question, decision, top_k, min_score, store)

    def _read(
        self,
        question: str,
        decision: ReadDocuments,
        top_k: int | None,
        min_score: float | None,
        store: VectorStore | None,
    ) -> Answer:
        """Retrieve inside the decision's scope and write the answer."""
        result = self._retrieval.retrieve(
            RetrievalRequest(
                question,
                profile=decision.profile,
                top_k=top_k,
                min_score=min_score,
                scope=decision.scope,
            ),
            store if store is not None else VectorStore(),
        )
        note = result.scope.note
        outcome = result.outcome
        if isinstance(outcome, Declined):
            text, explain = (
                self._refusals.render(outcome),
                Explain(decision, result, declined=outcome),
            )
        else:
            assert isinstance(outcome, Answerable)
            grounded = self._grounded.answer(question, outcome.chunks)
            text = grounded.text
            explain = Explain(decision, result, model_refused=grounded.refused)
        return Answer(f"{text}\n\n{note}" if note else text, explain)
