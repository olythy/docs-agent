"""Deciding how a question is answered: exactly from metadata, or by reading documents.

The router sits in front of ``query_knowledge_base``. It asks the query planner
(``metadata/planner.py``) what kind of question this is:

* a **count / list / sum / overview** question is answered *exactly* from the
  structured metadata and never reaches the retriever (top-k chunk retrieval
  cannot count), but only when the plan covers the whole question: a plan with
  a *residual* (a condition no key covers) cannot be exact and is read like a
  lookup instead;
* a **lookup** question ("what did the court decide in ...") still goes to the
  normal retrieval pipeline, restricted to the documents the planner's filters
  select (no filters: unrestricted);
* a question that names a **case/document identifier** always goes to the normal
  retrieval, without asking the planner, because an identifier is a pinpoint
  lookup the pipeline already handles.

Nothing here is silent. A counted answer states the filter that was executed and
how many documents could not be decided; a lookup restricted by a filter says how
many documents could not be checked against it; a part of the question no key
covers is named as not applied; and a question the planner cannot turn into a
valid plan gets a plain "could not interpret" instead of a guess.

Key exports:
    Routing            -- The decision: an answer, or a document set to read.
    QueryRouter        -- Makes the decision.
    ResultPhraser      -- Phrases an exact result in the question's language.
    render_result      -- The exact result as plain facts.
    get_query_router   -- Builds the router.
"""

import logging
import re
from dataclasses import dataclass, replace
from typing import Protocol

from metadata.executor import PlanExecutor, PlanResult
from metadata.plan import Operation, QueryPlan
from metadata.planner import (
    CatalogSource,
    PlanningFailed,
    QueryPlanner,
    ValueSource,
    collect_known_values,
    load_catalogs,
)
from models import DocumentSelection
from store import extract_identifier_tokens

logger = logging.getLogger(__name__)

#: Returned when the planner cannot turn the question into a valid plan.
COULD_NOT_INTERPRET_MESSAGE = (
    "I could not interpret this question well enough to answer it from the "
    "structured data, and I did not want to guess."
)

#: How many documents/groups are shown in a rendered answer.
_SHOWN = 50


@dataclass(frozen=True)
class Routing:
    """How a question is to be answered.

    Attributes:
        answer: The complete answer, when the question was answered exactly from
            the metadata. ``None`` means "read documents".
        selection: The documents the normal retrieval is restricted to, as a
            sub-select (see :class:`models.DocumentSelection`); ``None`` means
            unrestricted.
        note: A caveat to append to whatever answer the retrieval produces
            (e.g. how many documents could not be checked against the filter).
    """

    answer: str | None = None
    selection: DocumentSelection | None = None
    note: str | None = None


class CatalogReader(CatalogSource, ValueSource, Protocol):
    """The slice of :class:`document_store.DocumentStore` the router reads."""


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


class QueryRouter:
    """Decides whether a question is answered from metadata or by reading documents.

    Args:
        planner: Turns the question into a plan.
        executor: Runs a plan.
        catalog: Where the document types and their keys are read (the planner
            chooses among the approved types; nothing is configured).
        phraser: Words exact results as answers.
    """

    def __init__(
        self,
        planner: QueryPlanner,
        executor: PlanRunner,
        catalog: CatalogReader,
        phraser: Phraser,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._catalog = catalog
        self._phraser = phraser

    def route(self, question: str) -> Routing:
        """Decide how to answer ``question``.

        Raises:
            RuntimeError: If no document type is approved (the router was switched
                on before ``load-catalog``).
        """
        if extract_identifier_tokens(question):
            logger.info("[router] Identifier in the question: lookup, planner skipped.")
            return Routing()
        catalogs = load_catalogs(self._catalog)
        try:
            plan = self._planner.plan(
                question, catalogs, collect_known_values(self._catalog, catalogs)
            )
        except PlanningFailed as failure:
            logger.warning("[router] Could not plan %r: %s", question, failure.reason)
            return Routing(answer=COULD_NOT_INTERPRET_MESSAGE)

        if plan.residual and plan.operation is not Operation.LOOKUP:
            # Part of the question is covered by no key, so a count/list/sum over
            # the keys alone would answer a different, easier question, and
            # present it as exact. The documents have to be read; the filters
            # still narrow which ones.
            logger.info(
                "[router] %s plan has a residual (%r): reading the filtered documents instead.",
                plan.operation.value,
                plan.residual,
            )
            plan = replace(
                plan, operation=Operation.LOOKUP, group_by=None, sum_key=None
            )
        if plan.operation is Operation.LOOKUP:
            return self._restricted_lookup(plan)
        result = self._executor.execute(plan)
        logger.info(
            "[router] Answered exactly (%s): %s",
            plan.operation.value,
            result.explanation,
        )
        return Routing(answer=self._phraser.phrase(question, plan, result))

    def _restricted_lookup(self, plan: QueryPlan) -> Routing:
        if plan.doc_type is None and not plan.filters:
            return Routing()  # names no type and no restriction: read everything
        result = self._executor.execute(plan)
        note = (
            f"Note: {result.unknown} document(s) could not be checked against the "
            f"filter ({result.explanation}) and were not read."
            if result.unknown
            else None
        )
        if not result.count:
            return Routing(
                answer=(
                    f"No documents match the filter ({result.explanation})."
                    + (f" {note}" if note else "")
                )
            )
        return Routing(selection=result.selection, note=note)


def get_query_router() -> QueryRouter:
    """Build the router (the document types come from the database)."""
    from document_store import DocumentStore
    from drivers.llm import get_answer_driver
    from metadata.clock import SystemClock
    from metadata.compiler import PlanCompiler
    from metadata.date_ranges import DateRangeResolver
    from metadata.planner import LLMQueryPlanner

    clock = SystemClock()
    compiler = PlanCompiler(DateRangeResolver(clock))
    llm = get_answer_driver()
    store = DocumentStore()
    return QueryRouter(
        planner=LLMQueryPlanner(llm, compiler, clock),
        executor=PlanExecutor(store, compiler),
        catalog=store,
        phraser=ResultPhraser(llm),
    )
