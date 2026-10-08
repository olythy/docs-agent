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
* a request the system **cannot do yet** (documents *similar* to a named one) is
  said so plainly, instead of being sent to a search that can only return the
  named document and then answer "I could not find this information".

The planner always decides (when the router is on). An identifier in the question
is a *parameter* (which document the question is about), not an *intent*: a
question that names a case may ask about it (a lookup, the retrieval pins that
document), or ask for cases like it (not supported yet). It used to skip the
planner, which sent "five cases similar to X" to a retrieval that returned X alone.
A lookup that names an identifier is *not* restricted by the plan's filters: the
retrieval finds the named document itself, and a metadata restriction could only
exclude it (identifiers are stored in written variants, an anonymised document may
lack a court name).

Nothing here is silent. A counted answer states the filter that was executed and
how many documents could not be decided; a lookup restricted by a filter says how
many documents could not be checked against it; a part of the question no key
covers is named as not applied; and a question the planner cannot turn into a
valid plan gets a plain "could not interpret" instead of a guess.

Key exports:
    Routing            -- The decision: an answer, or a document set to read.
    QueryRouter        -- Makes the decision.
    as_routed          -- The plan as the router treats it (a residual means read, not count).
    get_query_router   -- Builds the router.
"""

import logging
from dataclasses import dataclass
from typing import Protocol

from metadata.executor import PlanExecutor
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
from query.answering import (
    Phraser,
    PlanRunner,
    ResultPhraser,
)
from query.decision import as_routed
from query.outcome import (
    COULD_NOT_INTERPRET_MESSAGE,
    NOT_SUPPORTED_MESSAGE,
    no_matching_documents_message,
)
from store import extract_identifier_tokens

logger = logging.getLogger(__name__)


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
        catalogs = load_catalogs(self._catalog)
        try:
            plan = self._planner.plan(
                question, catalogs, collect_known_values(self._catalog, catalogs)
            )
        except PlanningFailed as failure:
            logger.warning("[router] Could not plan %r: %s", question, failure.reason)
            return Routing(answer=COULD_NOT_INTERPRET_MESSAGE)

        if plan.operation is Operation.UNSUPPORTED:
            logger.info("[router] Not supported yet: %s", plan.reason)
            return Routing(answer=f"{NOT_SUPPORTED_MESSAGE} ({plan.reason})")

        routed = as_routed(plan)
        if routed is not plan:
            logger.info(
                "[router] %s plan has a residual (%r): reading the filtered documents instead.",
                plan.operation.value,
                plan.residual,
            )
        plan = routed
        if plan.operation is Operation.LOOKUP:
            if extract_identifier_tokens(question):
                # The question is about a named document: the retrieval finds it by
                # itself (it pins the identifier match). A metadata restriction would
                # only risk excluding it: an identifier is stored in written variants
                # ("4.P.20.409/2023/4-ítélet"), a court name may be missing for an
                # anonymised document. The restriction is for broad questions.
                logger.info(
                    "[router] Lookup anchored on an identifier: not restricted."
                )
                return Routing()
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
                answer=no_matching_documents_message(result.explanation, note)
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
