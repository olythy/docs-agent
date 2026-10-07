"""The decision side of a question: which documents may the retrieval look at?

The retrieval ranks chunks; it does not decide *which documents* they come from. That is
decided here, from the metadata, before any ranking: a :class:`Scope` is the answer, and the
retrieval receives only a store restricted to it.

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
* an identifier that resolves to nothing neither narrows nor refuses (it may be a reference
  to a case, or written in another style than the stored one); it is only recorded;
* filters alone narrow, and a filter that selects no document is an explicit refusal.

Key exports:
    Scope         -- Which documents the retrieval may look at, and why.
    ScopeResolver -- Builds a scope from the plan and the question's identifiers.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from metadata.executor import Membership, PlanResult
from metadata.identifier_resolver import IdentifierResolver, ResolvedIdentifiers
from metadata.plan import QueryPlan
from models import DocumentSelection
from query.outcome import Declined, DeclineReason


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


class PlanQueries(Protocol):
    """The slice of :class:`metadata.executor.PlanExecutor` the scope uses."""

    def execute(self, plan: QueryPlan) -> PlanResult: ...
    def members(self, plan: QueryPlan, document_ids: Sequence[int]) -> Membership: ...


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
        resolved = self._identifiers.resolve(identifiers)
        if resolved.document_ids:
            return self._named(plan, resolved)
        return self._filtered(plan, resolved)

    def _named(self, plan: QueryPlan, resolved: ResolvedIdentifiers) -> Scope:
        """The documents the question names; the filters only add a note if they disagree."""
        ids = resolved.document_ids
        note = None
        if plan.doc_type is not None or plan.filters:
            membership = self._plans.members(plan, ids)
            outside = len(ids) - len(membership.inside)
            if outside:
                note = (
                    f"Note: the question's filter ({membership.explanation}) was not applied: "
                    f"{outside} of the {len(ids)} document(s) named by identifier do not match it."
                )
        return Scope(
            selection=DocumentSelection(
                "SELECT id FROM documents WHERE id = ANY(%s)", (list(ids),)
            ),
            note=note,
            identifiers=resolved,
        )

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
