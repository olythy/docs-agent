"""The query plan: a strict, JSON-shaped description of what to count, list or sum.

An LLM planner turns a question into this structure; nothing here is SQL and
nothing here trusts the model. :func:`parse_plan` checks the *shape* (known fields,
allowed operations, consistent combinations) and raises :class:`PlanError` with a
message a planner can be re-prompted with. Checking the plan against the *catalog*
(do these keys exist, do these operators fit their types) and turning it into
parameterised SQL is the compiler's job (``metadata/compiler.py``).

A plan, as JSON:

    {"document_type": "court_decision", "operation": "count",
     "filters": [{"key": "issuing_body", "op": "eq", "value": "Debreceni Ítélőtábla"},
                 {"key": "decision_date", "op": "between",
                  "value": {"kind": "calendar", "year_offset": -1, "month": 10}}],
     "group_by": null, "sum_key": null, "limit": 50, "residual": null}

Key exports:
    Operation, FilterOp -- The closed sets of things a plan may ask for.
    Filter, QueryPlan   -- The parsed plan.
    parse_plan          -- Validate a raw dict into a QueryPlan.
    PlanError           -- A plan that is malformed or inconsistent.
"""

from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

DEFAULT_LIMIT = 50
MAX_LIMIT = 1000


class PlanError(ValueError):
    """A query plan that is malformed or self-contradictory."""


class Operation(StrEnum):
    """What the plan asks for.

    ``COUNT`` an exact number (optionally grouped by a categorical key); ``LIST``
    the matching documents; ``SUM`` the total of a number key; ``LOOKUP`` the set of
    matching documents, to restrict a normal retrieval to; ``OVERVIEW`` the matching
    documents with their summaries, for a synthesised overview; ``UNSUPPORTED`` a
    request this system cannot do yet (for example "list five cases similar to case
    X"): it is reported plainly, not answered by a search that cannot serve it.
    """

    LOOKUP = "lookup"
    LIST = "list"
    COUNT = "count"
    SUM = "sum"
    OVERVIEW = "overview"
    UNSUPPORTED = "unsupported"


class FilterOp(StrEnum):
    """The comparison a filter applies to a key's value."""

    EQ = "eq"
    NE = "ne"
    IN = "in"
    CONTAINS = "contains"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    BETWEEN = "between"


@dataclass(frozen=True)
class Filter:
    """One condition on one key.

    Attributes:
        key: A catalog key.
        op: The comparison.
        value: The comparison's operand, still raw JSON: its meaning depends on
            the key's type, which only the compiler (with the catalog) knows.
    """

    key: str
    op: FilterOp
    value: Any


@dataclass(frozen=True)
class QueryPlan:
    """A validated-in-shape query plan.

    Attributes:
        doc_type: The document type the plan is about (its keys are the usable ones
            and only its documents are considered); ``None`` for a ``lookup`` that
            names no type.
        operation: What to compute.
        filters: Conditions that must all hold (an AND).
        group_by: For ``COUNT``, a categorical key to break the count down by.
        sum_key: For ``SUM``, the number key to total.
        limit: How many documents/groups to return.
        residual: The part of the question no key covers, if any (not executed
            by SQL; a later step evaluates it over the matching documents).
        reason: For ``UNSUPPORTED``, what was asked that cannot be done yet, in the
            question's language.
    """

    doc_type: str | None
    operation: Operation
    filters: tuple[Filter, ...] = ()
    group_by: str | None = None
    sum_key: str | None = None
    limit: int = DEFAULT_LIMIT
    residual: str | None = None
    reason: str | None = None


_PLAN_FIELDS = {
    "document_type",
    "operation",
    "filters",
    "group_by",
    "sum_key",
    "limit",
    "residual",
    "reason",
}
_FILTER_FIELDS = {"key", "op", "value"}


def _enum[E: StrEnum](enum: type[E], raw: object, what: str) -> E:
    try:
        return enum(raw)
    except ValueError:
        allowed = ", ".join(member.value for member in enum)
        raise PlanError(f"unknown {what} {raw!r}; use one of: {allowed}") from None


def parse_plan(raw: object, doc_types: Collection[str]) -> QueryPlan:
    """Validate a raw plan (typically an LLM's JSON) into a :class:`QueryPlan`.

    Args:
        raw: The decoded JSON value.
        doc_types: The document types a plan may name (the usable ones; the model
            must choose among them, it cannot invent one).

    Returns:
        The plan.

    Raises:
        PlanError: If the plan is not an object, has unknown or missing fields,
            names no ``document_type`` (it must be one of ``doc_types`` or null),
            uses an unknown operation or operator, or combines options that do not
            go together (``group_by`` outside ``count``, ``sum_key`` outside ``sum``,
            ``sum`` without a key, a limit out of range, filters or a computation
            without a document type, an ``unsupported`` plan without a ``reason`` or
            with filters, a ``reason`` on any other operation).
    """
    if not isinstance(raw, dict):
        raise PlanError(f"a plan must be an object, got {type(raw).__name__}")
    unknown = set(raw) - _PLAN_FIELDS
    if unknown:
        raise PlanError(
            f"unknown plan field(s) {sorted(unknown)}; allowed: {sorted(_PLAN_FIELDS)}"
        )
    if "operation" not in raw:
        raise PlanError("a plan needs an 'operation'")
    operation = _enum(Operation, raw["operation"], "operation")
    doc_type = _document_type(raw, doc_types)
    if doc_type is None and operation not in (Operation.LOOKUP, Operation.UNSUPPORTED):
        raise PlanError(
            f"the '{operation.value}' operation needs a document_type "
            f"({', '.join(sorted(doc_types)) or 'none is known'}): an exact answer is "
            "only possible for one kind of document; use 'lookup' (with "
            '"document_type": null) when the question names no known kind'
        )

    raw_filters = raw.get("filters") or []
    if not isinstance(raw_filters, list):
        raise PlanError("'filters' must be a list of {key, op, value} objects")
    filters = tuple(_parse_filter(f, i) for i, f in enumerate(raw_filters))
    if doc_type is None and filters:
        raise PlanError(
            "filters are keys of a document type: name the document_type they belong to"
        )
    group_by, sum_key, reason = (
        raw.get("group_by"),
        raw.get("sum_key"),
        raw.get("reason"),
    )
    if operation is Operation.UNSUPPORTED:
        if not isinstance(reason, str) or not reason.strip():
            raise PlanError(
                "an 'unsupported' plan needs a 'reason': what was asked that cannot be "
                "done yet, in the question's language"
            )
        if filters:
            raise PlanError("an 'unsupported' plan has no filters")
    elif reason is not None:
        raise PlanError("'reason' only goes with the 'unsupported' operation")
    for name, value in (
        ("group_by", group_by),
        ("sum_key", sum_key),
        ("residual", raw.get("residual")),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise PlanError(f"'{name}' must be a non-empty string or null")
    if group_by is not None and operation is not Operation.COUNT:
        raise PlanError("'group_by' only goes with the 'count' operation")
    if sum_key is not None and operation is not Operation.SUM:
        raise PlanError("'sum_key' only goes with the 'sum' operation")
    if operation is Operation.SUM and sum_key is None:
        raise PlanError("the 'sum' operation needs a 'sum_key'")

    limit = raw.get("limit", DEFAULT_LIMIT)
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_LIMIT
    ):
        raise PlanError(f"'limit' must be an integer from 1 to {MAX_LIMIT}")

    return QueryPlan(
        doc_type=doc_type,
        operation=operation,
        filters=filters,
        group_by=group_by,
        sum_key=sum_key,
        limit=limit,
        residual=raw.get("residual"),
        reason=reason,
    )


def _document_type(raw: dict, doc_types: Collection[str]) -> str | None:
    """The plan's document type: one of ``doc_types``, or ``None`` (explicitly null)."""
    if "document_type" not in raw:
        raise PlanError(
            "a plan needs a 'document_type': one of "
            f"{', '.join(sorted(doc_types)) or '(none known)'}, or null"
        )
    chosen = raw["document_type"]
    if chosen is None:
        return None
    if not isinstance(chosen, str) or chosen not in doc_types:
        raise PlanError(
            f"unknown document_type {chosen!r}; use one of "
            f"{', '.join(sorted(doc_types)) or '(none known)'}, or null"
        )
    return chosen


def _parse_filter(raw: object, index: int) -> Filter:
    where = f"filters[{index}]"
    if not isinstance(raw, dict):
        raise PlanError(f"{where} must be an object")
    unknown = set(raw) - _FILTER_FIELDS
    if unknown:
        raise PlanError(
            f"{where}: unknown field(s) {sorted(unknown)}; allowed: {sorted(_FILTER_FIELDS)}"
        )
    for needed in ("key", "op", "value"):
        if needed not in raw:
            raise PlanError(f"{where} needs a '{needed}'")
    if not isinstance(raw["key"], str) or not raw["key"]:
        raise PlanError(f"{where}: 'key' must be a non-empty string")
    return Filter(
        raw["key"], _enum(FilterOp, raw["op"], f"{where} operator"), raw["value"]
    )
