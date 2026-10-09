"""The decision side of a question: which way it goes, which documents, which steps.

A question is answered in one of three ways, and this module decides which, **without
carrying it out and without wording it**: the decision is a typed value (:class:`Decision`),
so it can be tested, shown and worded by whoever needs it.

* ``AnswerExactly``: a count / list / sum / overview over the metadata (SQL, no retrieval);
* ``Refuse``: it cannot be answered, with who refused and why (a ``Declined``);
* ``ReadDocuments``: read the best chunks of the documents of a :class:`Scope`, with the steps
  of a named profile.

The retrieval ranks chunks; it does not decide *which documents* they come from, nor which
steps it runs. Both are decided here, before any ranking, and handed over: the scope (which
documents) and the profile name (which steps).

Two sources name documents, and they are combined by explicit rules:

* an **identifier** in the question ("what did the court decide in 4.P.20.409/2023/4") is
  resolved to the documents that carry it (:class:`metadata.identifier_resolver.IdentifierResolver`);
* the **planner's filters** (a court, a period) select documents by their metadata.

The rules, all stated in the notes and the explain record rather than applied silently:

* several identifiers -> the **union** of their documents (a question that compares two cases
  is about both);
* an identifier that resolves **wins over the planner's filters**: a document named by its
  number is a stronger signal than a court or year written alongside it, which may be a slip.
  When the filters disagree with the named documents the scope says so in its note;
* a question that names identifiers none of which resolves is read **unrestricted, and the
  planner's filters are not applied** (as the original router did): the named case may be a
  reference, or written in another style than the stored one, and the planner sometimes puts
  an identifier or a court into a lookup's filters, where a miss would turn a question that can
  be read into "no documents match". It is said in the note, not done silently;
* identifiers that did not resolve, or only resolved approximately (as a part, or with other
  separators), are named in the note, whatever else resolved;
* filters alone narrow, and a filter that selects no document is an explicit refusal.

Key exports:
    Scope            -- Which documents the retrieval may look at, and why.
    as_routed        -- The plan as treated: a residual means read, not count.
    ScopeResolver    -- Builds a scope from the plan and the question's identifiers.
    scope_of_source_file -- The scope of one named file, for a caller that already knows it.
    Decision         -- ReadDocuments | AnswerExactly | Refuse.
    ProfileSelector  -- Which profile (steps) a question to read gets.
    PlanningDecider  -- Decides with the query planner (the router's successor).
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from metadata.executor import Membership, PlanResult
from metadata.identifier_resolver import IdentifierResolver, ResolvedIdentifiers
from metadata.identifiers import normalize_identifier
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
from query.facts import QueryFacts
from query.outcome import Declined, DeclineReason

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scope:
    """Which documents the retrieval may look at.

    Attributes:
        selection: The documents as a sub-select, or ``None`` for no restriction.
        note: A caveat to show with the answer (a filter that did not apply, documents that
            could not be checked against it).
        identifiers: What the question's identifiers resolved to (kept for the explain
            record: which resolved, which did not).
    """

    selection: DocumentSelection | None = None
    note: str | None = None
    identifiers: ResolvedIdentifiers = field(
        default_factory=lambda: ResolvedIdentifiers({})
    )

    @property
    def names_several_documents(self) -> bool:
        """The question names more than one document (so each must be represented)."""
        return len(self.identifiers.document_ids) > 1


def scope_of_source_file(source_file: str) -> Scope:
    """The scope of one ingested file (a caller that already knows which document it means).

    Args:
        source_file: The file name as ingested.
    """
    return Scope(
        selection=DocumentSelection(
            "SELECT id FROM documents WHERE source_file = %s", (source_file,)
        )
    )


class PlanQueries(Protocol):
    """The slice of :class:`metadata.executor.PlanExecutor` the scope uses."""

    def execute(self, plan: QueryPlan) -> PlanResult: ...
    def members(self, plan: QueryPlan, document_ids: Sequence[int]) -> Membership: ...


def _identifier_notes(resolved: ResolvedIdentifiers) -> list[str]:
    """What a person should be told about how the question's identifiers resolved."""
    notes = []
    if resolved.unresolved:
        names = ", ".join(resolved.unresolved)
        notes.append(f"Note: no document carries the identifier(s) {names}.")
    if resolved.partial:
        names = ", ".join(resolved.partial)
        notes.append(
            f"Note: {names} matched only as a part of a longer identifier of a document."
        )
    if resolved.compact:
        names = ", ".join(resolved.compact)
        notes.append(
            f"Note: {names} matched a document only when the separators are ignored."
        )
    return notes


class ScopeResolver:
    """Builds the :class:`Scope` of a question that is to be read.

    Args:
        plans: Runs the planner's filters.
        identifiers: Resolves the question's identifiers to documents.
    """

    def __init__(self, plans: PlanQueries, identifiers: IdentifierResolver) -> None:
        self._plans = plans
        self._identifiers = identifiers

    def resolve(
        self, plan: QueryPlan, identifiers: Sequence[str] = ()
    ) -> Scope | Declined:
        """The scope of a lookup.

        Args:
            plan: The lookup plan (its type and filters; a plan with neither restricts
                nothing).
            identifiers: The identifiers the question names, as written.

        Returns:
            The :class:`Scope`, or a :class:`query.outcome.Declined` when the filters select
            no document.
        """
        resolved = self._identifiers.resolve(
            [
                i for i in identifiers if normalize_identifier(i)
            ]  # punctuation names nothing
        )
        if resolved.document_ids:
            return self._named(plan, resolved)
        if resolved.documents:  # identifiers were named, and none of them resolved
            return self._unresolved(plan, resolved)
        return self._filtered(plan, resolved)

    def _named(self, plan: QueryPlan, resolved: ResolvedIdentifiers) -> Scope:
        """The documents the question names; the filters only add a note if they disagree."""
        ids = resolved.document_ids
        notes = _identifier_notes(resolved)
        if plan.doc_type is not None or plan.filters:
            membership = self._plans.members(plan, ids)
            outside = len(ids) - len(membership.inside)
            if outside:
                notes.append(
                    f"Note: the question's filter ({membership.explanation}) was not applied: "
                    f"{outside} of the {len(ids)} document(s) named by identifier do not match it."
                )
        return Scope(
            selection=DocumentSelection(
                "SELECT id FROM documents WHERE id = ANY(%s)", (list(ids),)
            ),
            note=" ".join(notes) or None,
            identifiers=resolved,
        )

    def _unresolved(self, plan: QueryPlan, resolved: ResolvedIdentifiers) -> Scope:
        """Identifiers were named and none resolved: read unrestricted, and say so.

        The planner's filters are not applied (see the module docstring); if there were any,
        the note says they were left out.
        """
        notes = _identifier_notes(resolved)
        if plan.doc_type is not None or plan.filters:
            explanation = self._plans.members(plan, ()).explanation
            notes.append(
                f"Note: the question's filter ({explanation}) was not applied, because the "
                "question names an identifier that matched no document."
            )
        return Scope(note=" ".join(notes) or None, identifiers=resolved)

    def _filtered(
        self, plan: QueryPlan, resolved: ResolvedIdentifiers
    ) -> Scope | Declined:
        """The documents the planner's filters select (nothing named by an identifier)."""
        if plan.doc_type is None and not plan.filters:
            return Scope(identifiers=resolved)  # names no type and no restriction
        result = self._plans.execute(plan)
        note = (
            f"Note: {result.unknown} document(s) could not be checked against the "
            f"filter ({result.explanation}) and were not read."
            if result.unknown
            else None
        )
        if not result.count:
            return Declined(
                DeclineReason.NO_MATCHING_DOCUMENTS,
                stage="scope",
                detail=result.explanation,
                note=note,
            )
        return Scope(selection=result.selection, note=note, identifiers=resolved)


# ------------------------------------------------------------------------ the decision


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
