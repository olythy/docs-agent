"""Answering a question in words: exactly from the metadata, or from the chunks read.

Three classes with one job each, which the decision (``query.decision``) only chooses between:

* :class:`GroundedAnswerer` writes an answer from the chunks read, under an explicit
  :class:`AnswerPolicy`, and says whether the model refused (the prompt is
  ``drivers.llm._build_prompt``, byte for byte).
* :class:`ExactAnswerer` carries out an exact plan (a count, a list, a sum, an overview) and
  words the result. It does the SQL and the wording; it decides nothing.
* :class:`ResultPhraser` words an exact result in the question's own language and checks the
  numbers: the model only *words* the answer, every figure must appear in it unchanged, or the
  answer falls back to the plain facts (:func:`render_result`) so a mis-copied number is never
  presented as exact.

Key exports:
    AnswerPolicy     -- How the model is asked to answer (whether dates are shown).
    GroundedAnswer   -- The text, and whether it is the model's own refusal.
    AnswerObserver   -- Told after every answer the model wrote.
    GroundedAnswerer -- Writes an answer from the chunks read.
    ExactAnswerer -- Executes an exact plan and words the result.
    ResultPhraser -- Phrases an exact result, and checks the figures.
    render_result -- The exact result as plain facts.
"""

import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from drivers.llm import REFUSAL_SENTENCE, AnswerDriver, _build_prompt
from metadata.executor import PlanResult
from metadata.plan import Operation, QueryPlan
from models import RetrievedChunk

logger = logging.getLogger(__name__)

#: How many documents/groups are shown in a rendered answer.
_SHOWN = 50


@dataclass(frozen=True)
class AnswerPolicy:
    """How the model is asked to answer.

    Attributes:
        expose_document_date: Show each excerpt's document date.
    """

    expose_document_date: bool


@dataclass(frozen=True)
class GroundedAnswer:
    """What the model said about the chunks.

    Attributes:
        text: The reply, unchanged.
        refused: The reply is the model's own refusal sentence.
    """

    text: str
    refused: bool


class AnswerObserver(Protocol):
    """Told after the model wrote an answer (the answering counterpart of a step observer)."""

    def on_answer(
        self,
        question: str,
        chunks: tuple[RetrievedChunk, ...],
        answer: "GroundedAnswer",
        seconds: float,
    ) -> None:
        """Called with the question, the chunks it was written from, the answer and how
        long the model took."""
        ...


class GroundedAnswerer:
    """Writes an answer from the chunks read.

    Args:
        driver: The language model.
        policy: How it is asked to answer.
        max_tokens: Maximum tokens to generate.
        observers: Told after every answer (the audit log); the answerer itself knows
            nothing about log files.
        clock: Returns seconds, to time the model call (injected for tests).
    """

    def __init__(
        self,
        driver: AnswerDriver,
        policy: AnswerPolicy,
        max_tokens: int = 1024,
        observers: Sequence[AnswerObserver] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._driver = driver
        self._policy = policy
        self._max_tokens = max_tokens
        self._observers = tuple(observers)
        self._clock = clock

    def answer(
        self, question: str, chunks: tuple[RetrievedChunk, ...]
    ) -> GroundedAnswer:
        """Ask the model to answer ``question`` from ``chunks`` (best first)."""
        system_prompt, user_message = _build_prompt(
            question,
            list(chunks),
            expose_document_date=self._policy.expose_document_date,
        )
        started = self._clock()
        text = self._driver.generate(system_prompt, user_message, self._max_tokens)
        seconds = self._clock() - started
        refused = text.strip().strip("'\"").lower().startswith(REFUSAL_SENTENCE.lower())
        answer = GroundedAnswer(text, refused)
        for observer in self._observers:
            observer.on_answer(question, chunks, answer, seconds)
        return answer


class PlanRunner(Protocol):
    """Runs a plan (:class:`metadata.executor.PlanExecutor`)."""

    def execute(self, plan: QueryPlan) -> PlanResult: ...


class Phraser(Protocol):
    """Turns an exact result into an answer in the question's language."""

    def phrase(self, question: str, plan: QueryPlan, result: PlanResult) -> str: ...


def render_result(plan: QueryPlan, result: PlanResult) -> str:
    """The exact result as plain facts (also the safe form of an answer).

    Args:
        plan: The executed plan.
        result: What it returned.
    """
    lines = [f"Executed filter: {result.explanation}"]
    unknown_note = (
        f"{result.unknown} further document(s) could not be decided (a filtered "
        "value is unverified or was not extracted), so the true figure may be "
        f"up to {result.unknown} higher."
        if result.unknown
        else None
    )
    if result.operation is Operation.SUM:
        lines.append(
            f"Total of {plan.sum_key}: {result.total} "
            f"(over {result.sum_documents} document(s) that state it)"
        )
    elif result.groups:
        lines.append(f"Matching documents: {result.count}")
        lines += [f"  {value}: {n}" for value, n in result.groups[:_SHOWN]]
    elif result.operation is Operation.COUNT:
        lines.append(f"Matching documents: {result.count}")
    else:
        lines.append(f"Matching documents: {result.count}")
        for row in result.documents[:_SHOWN]:
            _, source_file, *rest = row
            summary = f" -- {str(rest[0])[:300]}" if rest and rest[0] else ""
            lines.append(f"  {source_file}{summary}")
        if result.truncated:
            lines.append(f"  (showing {len(result.documents)} of {result.count})")
    if unknown_note:
        lines.append(unknown_note)
    if plan.residual:
        lines.append(
            f"Not applied (no key covers it, so the result is not narrowed by it): {plan.residual}"
        )
    return "\n".join(lines)


def _figures(result: PlanResult) -> list[str]:
    """The numbers an answer must reproduce exactly."""
    figures = [str(n) for n in (result.count, result.unknown or None) if n is not None]
    figures += [str(n) for _, n in result.groups[:_SHOWN]]
    if result.total is not None:
        figures.append(str(int(result.total)))
    return figures


def _plain_digits(text: str) -> str:
    """Drop thousands separators so "1 234" / "1.234" / "1,234" compare as 1234."""
    return re.sub(r"(?<=\d)[ ., ](?=\d{3}(?!\d))", "", text)


class ResultPhraser:
    """Phrases an exact result in the question's own language, and checks the numbers.

    The model only *words* the answer; every figure must appear in it unchanged. If
    one is missing the answer falls back to the plain facts, and a warning is
    logged, so a mis-copied number is never presented as exact.

    Args:
        llm: The driver used for the call.
    """

    def __init__(self, llm) -> None:
        self._llm = llm

    def phrase(self, question: str, plan: QueryPlan, result: PlanResult) -> str:
        facts = render_result(plan, result)
        prompt = (
            "Answer the question in the same language as the question, using ONLY "
            "the facts below. Copy every number exactly. If some documents could "
            "not be decided, or part of the question is not applied, say so "
            "plainly. Do not add anything the facts do not state.\n\n"
            f"QUESTION: {question}\n\nFACTS:\n{facts}"
        )
        reply = (
            self._llm.run_tool_calling_turn(
                [{"role": "user", "content": prompt}]
            ).content
            or ""
        ).strip()
        digits = _plain_digits(reply)
        missing = [f for f in _figures(result) if f not in digits]
        if not reply or missing:
            logger.warning(
                "[router] Phrased answer dropped figure(s) %s; returning the plain facts.",
                missing,
            )
            return facts
        return f"{reply}\n\n[Executed filter: {result.explanation}]"


class ExactAnswerer:
    """Carries out an exact plan and words the result.

    Args:
        executor: Runs the plan (SQL over the metadata, read-only).
        phraser: Words the result as an answer.

    Raises:
        metadata.plan.PlanError: From :meth:`answer`, if the plan does not fit the catalog
            (it is not caught here: a plan the planner validated that no longer fits is a
            bug or a changed catalog, not a question to answer).
    """

    def __init__(self, executor: PlanRunner, phraser: Phraser) -> None:
        self._executor = executor
        self._phraser = phraser

    def answer(self, question: str, plan: QueryPlan) -> str:
        """Execute ``plan`` and word what it found.

        Args:
            question: The user's question (the answer is in its language).
            plan: A count / list / sum / overview plan.
        """
        result = self._executor.execute(plan)
        logger.info(
            "[answer] Answered exactly (%s): %s",
            plan.operation.value,
            result.explanation,
        )
        return self._phraser.phrase(question, plan, result)
