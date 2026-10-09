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
    cited_labels  -- The excerpt labels an answer cites.
    render_sources -- The "Sources:" block of an answer's citations.
"""

import logging
import re
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from drivers.llm import REFUSAL_SENTENCE, AnswerDriver, _build_prompt, labelled_chunks
from metadata.executor import PlanResult
from metadata.plan import Operation, QueryPlan
from metadata.verification import EvidenceVerifier
from models import Citation, RetrievedChunk

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


#: A bracketed list of one- or two-digit excerpt labels: ``[2]``, ``[1, 3]``. Not a year
#: (``[2023]``) or a footnote of any length: the labels are the numbers 1..top_k.
_LABELS = re.compile(r"\[\s*(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\s*\]")


#: The line before the quotes of an answer, and one quote line: ``[2] "a passage"``.
_QUOTES_MARK = re.compile(r"(?:^|\n)[ \t]*QUOTES:[ \t]*\n?")
_QUOTE_LINE = re.compile(r'^\s*\[(\d{1,2})\]\s*["“„»«]?(.*?)["”»«]?\s*$')


def split_quotes(text: str) -> tuple[str, dict[int, str]]:
    """The answer proper and the passages the model quoted for the labels it cited.

    Args:
        text: The model's reply: the answer, then optionally a ``QUOTES:`` block with one
            ``[n] "passage"`` line per cited label.

    Returns:
        ``(answer without the block, {label: passage})``. A reply without the block has no
        quotes.
    """
    marks = list(_QUOTES_MARK.finditer(text))
    if not marks:
        return text.strip(), {}
    mark = marks[-1]
    quotes: dict[int, str] = {}
    for line in text[mark.end() :].splitlines():
        found = _QUOTE_LINE.match(line)
        if found and found.group(2).strip():
            quotes.setdefault(int(found.group(1)), found.group(2).strip())
    return text[: mark.start()].rstrip(), quotes


def strip_labels(text: str, labels: Collection[int]) -> str:
    """``text`` without the given labels.

    A bracket that loses all its labels goes, with the space before it; one that keeps some
    is rewritten without the others (``[1, 3]`` minus 1 is ``[3]``).
    """
    if not labels:
        return text

    def rewrite(found: re.Match[str]) -> str:
        numbers = [
            n for n in re.split(r"\s*[,;]\s*", found.group(2)) if int(n) not in labels
        ]
        return f"{found.group(1)}[{', '.join(numbers)}]" if numbers else ""

    return re.sub(r"([ \t]*)\[\s*(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\s*\]", rewrite, text)


def cited_labels(text: str) -> list[int]:
    """The excerpt labels an answer cites, each once, in the order first used."""
    found: list[int] = []
    for group in _LABELS.findall(text):
        for number in re.split(r"\s*[,;]\s*", group):
            label = int(number)
            if label not in found:
                found.append(label)
    return found


@dataclass(frozen=True)
class GroundedAnswer:
    """What the model said about the chunks.

    Attributes:
        text: The reply, unchanged.
        refused: The reply is the model's own refusal sentence.
        citations: The chunks the reply cites by label whose quoted passage is really in the
            chunk, resolved by the system (none for a refusal).
        unknown_labels: Labels the reply cites that no excerpt carried: a citation to
            nothing, counted as a wrong citation.
        unverified_labels: Labels that named an excerpt but came with no quote, or a quote
            that is not in it: dropped from the visible answer, not claimed as citations.
        visible_text: The answer as shown: without the quotes block and without the labels
            that were not verified (``None``: the reply as it is).
    """

    text: str
    refused: bool
    citations: tuple[Citation, ...] = ()
    unknown_labels: tuple[int, ...] = ()
    unverified_labels: tuple[int, ...] = ()
    visible_text: str | None = None

    @property
    def visible(self) -> str:
        """What the person is shown."""
        return self.text if self.visible_text is None else self.visible_text


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
        if not refused:
            answer = self._with_citations(answer, chunks)
        for observer in self._observers:
            observer.on_answer(question, chunks, answer, seconds)
        return answer

    @staticmethod
    def _with_citations(
        answer: GroundedAnswer, chunks: tuple[RetrievedChunk, ...]
    ) -> GroundedAnswer:
        """``answer`` with each label it cites resolved and its quote checked in the chunk.

        A label counts as a citation only if the model quoted a passage for it and the
        passage is in the chunk shown under that label (the same check as the metadata
        layer's evidence quotes). Anything else is taken out of the visible answer.
        """
        body, quotes = split_quotes(answer.text)
        shown = labelled_chunks(list(chunks))
        citations: list[Citation] = []
        unknown: list[int] = []
        unverified: list[int] = []
        for label in cited_labels(body):
            if not 1 <= label <= len(shown):
                unknown.append(label)
                continue
            chunk = shown[label - 1]
            quote = quotes.get(label)
            if quote and EvidenceVerifier(chunk.content).contains(quote):
                citations.append(
                    Citation(
                        label,
                        chunk.id,
                        chunk.metadata.source_file,
                        chunk.metadata.chunk_index,
                        quote,
                    )
                )
            else:
                unverified.append(label)
        return replace(
            answer,
            citations=tuple(citations),
            unknown_labels=tuple(unknown),
            unverified_labels=tuple(unverified),
            visible_text=strip_labels(body, {*unknown, *unverified}),
        )


def _short(quote: str, limit: int = 160) -> str:
    """A quote cut to ``limit`` characters (with an ellipsis) for the sources block."""
    return quote if len(quote) <= limit else quote[: limit - 1].rstrip() + "…"


def render_sources(citations: Sequence[Citation]) -> str:
    """The "Sources:" block listing the chunks an answer cites (empty when it cites none)."""
    if not citations:
        return ""
    lines = [
        f"[{c.label}] {c.source_file}, section {c.chunk_index}"
        + (f': "{_short(c.quote)}"' if c.quote else "")
        for c in citations
    ]
    return "Sources:\n" + "\n".join(lines)


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
    unstated_note = (
        f"{result.unstated} further document(s) do not state a value a filter asked "
        "for (for example no date), so they are not in the figures above, but may "
        "belong to them."
        if result.unstated
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
        lines += _grouping_gaps(result)
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
    lines += _by_year_lines(plan, result)
    if unknown_note:
        lines.append(unknown_note)
    if unstated_note:
        lines.append(unstated_note)
    if plan.residual:
        lines.append(
            f"Not applied (no key covers it, so the result is not narrowed by it): {plan.residual}"
        )
    if plan.caveat:
        lines.append(f"Read with care: {plan.caveat}")
    lines += [f"Reliability of {key}: {note}" for key, note in result.reliability]
    return "\n".join(lines)


def _by_year_lines(plan: QueryPlan, result: PlanResult) -> list[str]:
    """The count year by year (and by the grouping key within each year), as plain lines.

    Written by code, never left to the model: a table of this size is exactly where a
    copied number goes wrong.
    """
    if plan.by_year is None:
        return []
    if not result.yearly:
        return [f"By year ({plan.by_year}): no matching document has a value for it."]
    within: dict[int, list[str]] = {}
    for year, value, n in result.by_period:
        within.setdefault(year, []).append(f"{value} {n}")
    lines = [f"By year ({plan.by_year}):"]
    for year, n in result.yearly:
        detail = f" ({', '.join(within[year])})" if year in within else ""
        lines.append(f"  {year}: {n}{detail}")
    return lines


_UNGROUPED_REASON = {
    "unverified": "its value could not be verified",
    "confirmed_absent": "the document does not state it",
    "not_attempted": "it has not been extracted yet",
}


def _grouping_gaps(result: PlanResult) -> list[str]:
    """Lines saying which matching documents the shown groups do not cover, and why."""
    lines = [
        f"  (in no group, {_UNGROUPED_REASON.get(reason, reason)}): {n}"
        for reason, n in result.ungrouped
    ]
    if result.groups_overlap:
        lines.append(
            "  Note: a document can be in more than one group, so the groups can add "
            f"up to more than {result.count}."
        )
    else:
        shown = sum(n for _, n in result.groups[:_SHOWN])
        if shown < result.grouped_documents:
            lines.append(
                f"  ({result.grouped_documents - shown} more document(s) are in "
                "smaller groups that are not listed.)"
            )
    return lines


def _figures(result: PlanResult) -> list[str]:
    """The numbers an answer must reproduce exactly."""
    figures = [
        str(n)
        for n in (result.count, result.unknown or None, result.unstated or None)
        if n is not None
    ]
    figures += [str(n) for _, n in result.groups[:_SHOWN]]
    figures += [str(n) for _, n in result.ungrouped]
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
        year_note = (
            "A year-by-year table is added after your answer by the system: do not "
            "repeat it, only say in a sentence what it shows. "
            if plan.by_year is not None and result.yearly
            else ""
        )
        prompt = (
            "Answer the question in the same language as the question, using ONLY "
            "the facts below. Copy every number exactly. If some documents could "
            "not be decided, or part of the question is not applied, say so "
            f"plainly. {year_note}Do not add anything the facts do not state.\n\n"
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
        caveat = f"\n[Read with care: {plan.caveat}]" if plan.caveat else ""
        reliability = "".join(
            f"\n[Reliability of {key}: {note}]" for key, note in result.reliability
        )
        by_year = _by_year_lines(plan, result)
        table = "\n\n" + "\n".join(by_year) if by_year else ""
        return (
            f"{reply}{table}\n\n[Executed filter: {result.explanation}]"
            f"{caveat}{reliability}"
        )


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
