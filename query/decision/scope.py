"""Which documents a question may read: the scope, and the rules that build it.

The retrieval ranks chunks; it does not decide *which documents* they come from. That is
decided here, before any ranking, from two sources that name documents, combined by
explicit rules:

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
    Scope                -- Which documents the retrieval may look at, and why.
    ScopeResolver        -- Builds a scope from the plan and the question's identifiers.
    scope_of_source_file -- The scope of one named file, for a caller that already knows it.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from metadata.executor import Membership, PlanResult
from metadata.identifier_resolver import IdentifierResolver, ResolvedIdentifiers
from metadata.identifiers import normalize_identifier
from metadata.plan import QueryPlan
from models import DocumentSelection
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
