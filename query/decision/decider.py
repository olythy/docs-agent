"""The decision of a question: which way it goes, which documents, which steps.

A question is answered in one of three ways, and this module decides which, **without
carrying it out and without wording it**: the decision is a typed value (:class:`Decision`),
so it can be tested, shown and worded by whoever needs it.

* ``AnswerExactly``: a count / list / sum / overview over the metadata (SQL, no retrieval);
* ``Refuse``: it cannot be answered, with who refused and why (a ``Declined``);
* ``ReadDocuments``: read the best chunks of the documents of a :class:`~query.decision.scope.Scope`,
  with the steps of a named profile.

The retrieval does not decide *which documents* it ranks nor which steps it runs. Both are
decided here, before any ranking, and handed over: the scope (which documents; built by
``query.decision.scope``) and the profile name (which steps).

Key exports:
    Decision         -- ReadDocuments | AnswerExactly | Refuse.
    as_routed        -- The plan as treated: a residual means read, not count.
    ProfileSelector  -- Which profile (steps) a question to read gets.
    PlanningDecider  -- Decides with the query planner.
"""

import logging
from dataclasses import dataclass, replace
from typing import Protocol

from metadata.plan import Operation, QueryPlan
from metadata.planner import (
    CatalogSource,
    PlanningFailed,
    QueryPlanner,
    ValueSource,
    collect_known_values,
    load_catalogs,
)
from query.facts import QueryFacts
from query.outcome import Declined, DeclineReason

logger = logging.getLogger(__name__)

from query.decision.scope import Scope, ScopeResolver


def as_routed(plan: QueryPlan) -> QueryPlan:
    """The plan as the decider treats it: an exact plan with a residual is read instead.

    Part of the question is covered by no key, so a count / list / sum / overview over the
    keys alone would answer a different, easier question and present it as exact. The
    documents have to be read; the filters still narrow which ones. Anything else is
    returned unchanged (the same object).

    Args:
        plan: A plan as the planner produced it.
    """
    if plan.residual and plan.operation not in (
        Operation.LOOKUP,
        Operation.UNSUPPORTED,
    ):
        return replace(plan, operation=Operation.LOOKUP, group_by=None, sum_key=None)
    return plan


@dataclass(frozen=True)
class ReadDocuments:
    """Read the best chunks of some documents.

    Attributes:
        facts: The question's facts.
        plan: The planner's plan (``None`` when no planner was asked).
        profile: The name of the profile (the steps) to run.
        scope: Which documents the retrieval may look at.
    """

    facts: QueryFacts
    plan: QueryPlan | None
    profile: str
    scope: Scope


@dataclass(frozen=True)
class AnswerExactly:
    """Answer from the metadata alone: a count, a list, a sum or an overview.

    Attributes:
        facts: The question's facts.
        plan: The plan to execute.
    """

    facts: QueryFacts
    plan: QueryPlan


@dataclass(frozen=True)
class Refuse:
    """The question is not answered.

    Attributes:
        facts: The question's facts.
        declined: Who refused and why.
    """

    facts: QueryFacts
    declined: Declined


Decision = ReadDocuments | AnswerExactly | Refuse


class ProfileSelector:
    """Chooses the profile (the retrieval steps) a question to read gets.

    Today there is one rule: the configured default. A profile per kind of question is
    added here when a second profile has been measured; the retrieval never chooses.

    Args:
        default: The profile name to use (``best_chunks``).
    """

    def __init__(self, default: str) -> None:
        self._default = default

    def select(self, facts: QueryFacts, plan: QueryPlan | None) -> str:
        """The profile name for this question."""
        return self._default


class Decider(Protocol):
    """Decides how a question is answered."""

    def decide(self, facts: QueryFacts) -> Decision: ...


class PlanningDecider:
    """Asks the query planner what kind of question this is, and decides.

    The successor of ``QueryRouter.route`` without its execution and its wording: it
    returns the decision, and ``AnswerExactly`` is carried out and worded elsewhere.

    * the planner cannot produce a valid plan -> ``Refuse(COULD_NOT_INTERPRET)``;
    * ``unsupported`` (documents similar to a named one) -> ``Refuse(NOT_SUPPORTED)`` with the
      planner's reason;
    * ``survey`` (the practice or the outcomes across many documents the question does not
      name) -> ``Refuse(SURVEY_NOT_YET)``, until a profile that reads that many documents
      exists; a survey that names an identifier is read like a lookup;
    * an exact operation covering the whole question -> ``AnswerExactly``; one with a
      residual (a condition no key covers) is read like a lookup;
    * a lookup -> the scope (the named documents, or the filters' documents) and the profile.

    Args:
        planner: Turns the question into a plan.
        catalog: Where the document types and keys, and the stored values, are read.
        scopes: Builds the scope of a lookup.
        profiles: Chooses the profile of a question to read.

    Raises:
        RuntimeError: From :meth:`decide` if no document type is approved yet.
    """

    def __init__(
        self,
        planner: QueryPlanner,
        catalog: "CatalogReader",
        scopes: ScopeResolver,
        profiles: ProfileSelector,
    ) -> None:
        self._planner = planner
        self._catalog = catalog
        self._scopes = scopes
        self._profiles = profiles

    def decide(self, facts: QueryFacts) -> Decision:
        """Decide how ``facts.question`` is answered."""
        catalogs = load_catalogs(self._catalog)
        try:
            plan = self._planner.plan(
                facts.question, catalogs, collect_known_values(self._catalog, catalogs)
            )
        except PlanningFailed as failure:
            logger.warning(
                "[decide] Could not plan %r: %s", facts.question, failure.reason
            )
            return Refuse(
                facts,
                Declined(
                    DeclineReason.COULD_NOT_INTERPRET,
                    stage="planning",
                    detail=failure.reason,
                ),
            )
        if plan.operation is Operation.UNSUPPORTED:
            logger.info("[decide] Not supported yet: %s", plan.reason)
            return Refuse(
                facts,
                Declined(
                    DeclineReason.NOT_SUPPORTED, stage="planning", detail=plan.reason
                ),
            )
        if plan.operation is Operation.SURVEY:
            if facts.identifiers:
                # A question that names a document is read, whatever the planner thought:
                # refusing it would turn something the system answers well into a "not yet".
                logger.info(
                    "[decide] A survey that names %s: read like a lookup.",
                    ", ".join(facts.identifiers),
                )
                plan = replace(plan, operation=Operation.LOOKUP)
            else:
                logger.info("[decide] A survey across many documents: not built yet.")
                return Refuse(
                    facts,
                    Declined(DeclineReason.SURVEY_NOT_YET, stage="planning"),
                )
        plan = as_routed(plan)
        if plan.operation is not Operation.LOOKUP:
            return AnswerExactly(facts, plan)
        scope = self._scopes.resolve(plan, facts.identifiers)
        if isinstance(scope, Declined):
            return Refuse(facts, scope)
        return ReadDocuments(facts, plan, self._profiles.select(facts, plan), scope)


class CatalogReader(CatalogSource, ValueSource, Protocol):
    """The slice of :class:`document_store.DocumentStore` the decider reads."""
