"""Compiling a query plan into parameterised SQL, checked against the catalog.

Pure: it takes a plan and the catalog's keys and returns SQL text plus parameters; it
never touches a database and never calls an LLM. Three rules keep it safe:

* **No value, key name or user text is ever put into the SQL string.** Values,
  key names and patterns are bound parameters; the only text spliced in is a
  fixed fragment chosen from a closed set (operator, column name).
* **Everything is checked against the catalog first**: the key exists and is
  approved, the operator fits the key's type, a categorical value is one of the
  allowed tokens, a date is a valid spec.
* **A count also says what it could not decide.** Besides the exact matching set,
  it builds a second query for documents that *might* match but whose metadata for
  a filtered key is unverified or never extracted, so an answer can say
  "N, plus K unknown" instead of silently dropping them.

Key exports:
    PlanCompiler  -- Compiles a plan.
    CompiledQuery -- The SQL, its parameters, and a human-readable echo of the filter.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from metadata.date_ranges import DateRange, DateRangeResolver, DateSpecError
from metadata.plan import Filter, FilterOp, Operation, PlanError, QueryPlan
from models import DocumentSelection, KeyStatus, MetaKey, ValueType

_VALUE_COLUMN = {
    ValueType.TEXT: "value_text",
    ValueType.NUMBER: "value_number",
    ValueType.DATE: "value_date",
    ValueType.BOOL: "value_bool",
}

_ALLOWED_OPS = {
    ValueType.TEXT: {FilterOp.EQ, FilterOp.NE, FilterOp.IN, FilterOp.CONTAINS},
    ValueType.NUMBER: {
        FilterOp.EQ,
        FilterOp.NE,
        FilterOp.GT,
        FilterOp.GTE,
        FilterOp.LT,
        FilterOp.LTE,
        FilterOp.BETWEEN,
    },
    ValueType.DATE: {
        FilterOp.EQ,
        FilterOp.IN,
        FilterOp.GT,
        FilterOp.GTE,
        FilterOp.LT,
        FilterOp.LTE,
        FilterOp.BETWEEN,
    },
    ValueType.BOOL: {FilterOp.EQ},
}

_COMPARISON = {
    FilterOp.EQ: "=",
    FilterOp.NE: "<>",
    FilterOp.GT: ">",
    FilterOp.GTE: ">=",
    FilterOp.LT: "<",
    FilterOp.LTE: "<=",
}

_KNOWN_STATES = "('present', 'confirmed_absent')"

#: Sentence punctuation that an exact text match ignores at the end of a value. A model
#: copies a value out of the question, where a case number or a name is often followed by
#: the sentence's own full stop ("... 4.P.20.409/2023/4. számú ügy"): an exact match then
#: finds nothing (confirmed live: the single-document questions fell to 50-75%).
_TRAILING = ".,;:"


@dataclass(frozen=True)
class CompiledQuery:
    """The result of compiling a plan.

    Attributes:
        operation: What the plan asked for.
        sql: The main query (bound with ``params``).
        params: Its parameters.
        count_sql: A query for the exact number of matching documents (to report the
            total beside a truncated list or a grouped count, whose groups need not add
            up to it); ``None`` for a plain ``count`` or ``sum``, whose main query
            already is that.
        count_params: Parameters of ``count_sql``.
        unknown_sql: A query for the number of documents that might match but cannot
            be decided (a filtered key is unverified or never extracted); ``None``
            when the plan has no filters.
        unknown_params: Parameters of ``unknown_sql``.
        explanation: The filter in words, to show the user what was actually asked.
        selection: The matching documents as a sub-select, for restricting a
            retrieval to them (see :class:`models.DocumentSelection`).
    """

    operation: Operation
    sql: str
    params: tuple[Any, ...]
    count_sql: str | None
    count_params: tuple[Any, ...]
    unknown_sql: str | None
    unknown_params: tuple[Any, ...]
    explanation: str
    selection: DocumentSelection


class PlanCompiler:
    """Compiles :class:`metadata.plan.QueryPlan` objects.

    Args:
        resolver: Resolves date specs to ranges (it owns the notion of "today").
    """

    def __init__(self, resolver: DateRangeResolver) -> None:
        self._resolver = resolver

    def compile(self, plan: QueryPlan, keys: list[MetaKey]) -> CompiledQuery:
        """Validate ``plan`` against ``keys`` and build its queries.

        Args:
            plan: A plan that already passed :func:`metadata.plan.parse_plan`.
            keys: The catalog's keys for ``plan.doc_type`` (only approved ones are
                usable).

        Returns:
            The compiled queries.

        Raises:
            PlanError: If a key is unknown or not approved, an operator does not
                fit a key's type, a value is invalid, or a date spec is impossible.
        """
        usable = {
            k.key: k
            for k in keys
            if k.status is KeyStatus.APPROVED and k.doc_type == plan.doc_type
        }
        compiled = [self._filter(f, usable) for f in plan.filters]

        # A plan about a document type only considers documents of that type. The
        # test is written so an untyped document is plainly false (never NULL),
        # which keeps the ``NOT (match)`` of the unknown query well defined.
        type_sql, type_params = "", ()
        if plan.doc_type is not None:
            type_sql = "(d.document_type IS NOT NULL AND d.document_type = %s)"
            type_params = (plan.doc_type,)
        parts = [t for t in (type_sql,) if t] + [c.satisfied_sql for c in compiled]
        match_sql = " AND ".join(parts) or "TRUE"
        match_params = (
            *type_params,
            *(p for c in compiled for p in c.satisfied_params),
        )
        words = [c.words for c in compiled]
        explanation = (
            " AND ".join(words)
            if plan.doc_type is None
            else f"{plan.doc_type} documents"
            + (": " + " AND ".join(words) if words else "")
        ) or "all documents"

        # Documents that might match but cannot be decided: a filtered key that is
        # unverified or was never extracted, or no document type yet (it may be
        # this type). Nothing to decide for a plan with neither type nor filters.
        unknown_sql: str | None = None
        unknown_params: tuple[Any, ...] = ()
        if compiled or plan.doc_type is not None:
            could_parts = (
                ["(d.document_type IS NULL OR d.document_type = %s)"]
                if plan.doc_type is not None
                else []
            ) + [f"({c.satisfied_sql} OR NOT {c.known_sql})" for c in compiled]
            could = " AND ".join(could_parts)
            could_params = (
                *type_params,
                *(p for c in compiled for p in (*c.satisfied_params, *c.known_params)),
            )
            unknown_sql = (
                f"SELECT count(*) FROM documents d WHERE {could} AND NOT ({match_sql})"
            )
            unknown_params = (*could_params, *match_params)

        count_sql = f"SELECT count(*) FROM documents d WHERE {match_sql}"
        sql, params, needs_count = self._operation(
            plan, usable, match_sql, match_params
        )
        return CompiledQuery(
            operation=plan.operation,
            sql=sql,
            params=params,
            count_sql=count_sql if needs_count else None,
            count_params=match_params if needs_count else (),
            unknown_sql=unknown_sql,
            unknown_params=unknown_params,
            explanation=explanation,
            selection=DocumentSelection(
                f"SELECT d.id FROM documents d WHERE {match_sql}", match_params
            ),
        )

    # -------------------------------------------------------------- operations

    def _operation(
        self,
        plan: QueryPlan,
        usable: dict[str, MetaKey],
        match_sql: str,
        match_params: tuple[Any, ...],
    ) -> tuple[str, tuple[Any, ...], bool]:
        op = plan.operation
        if op is Operation.COUNT and plan.group_by is not None:
            key = self._key(plan.group_by, usable, "group_by")
            if key.value_type is ValueType.DATE:
                raise PlanError(
                    f"cannot group by the date key {key.key!r}; group by a text key"
                )
            column = _VALUE_COLUMN[key.value_type]
            sql = (
                f"SELECT m.{column}, count(DISTINCT d.id) FROM documents d "
                "JOIN document_meta m ON m.document_id = d.id AND m.key = %s "
                f"WHERE {match_sql} GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT %s"
            )
            return sql, (key.key, *match_params, plan.limit), True
        if op is Operation.COUNT:
            return (
                f"SELECT count(*) FROM documents d WHERE {match_sql}",
                match_params,
                False,
            )
        if op is Operation.SUM:
            assert plan.sum_key is not None  # parse_plan guarantees it
            key = self._key(plan.sum_key, usable, "sum_key")
            if key.value_type is not ValueType.NUMBER:
                raise PlanError(
                    f"cannot sum {key.key!r}: it is a {key.value_type.value} key, not a number"
                )
            sql = (
                "SELECT coalesce(sum(m.value_number), 0), count(DISTINCT d.id) FROM documents d "
                f"JOIN document_meta m ON m.document_id = d.id AND m.key = %s WHERE {match_sql}"
            )
            return sql, (key.key, *match_params), False
        if op is Operation.LOOKUP:
            # The documents themselves are not fetched: a retrieval is restricted
            # with the compiled selection, so only their number is needed.
            return (
                f"SELECT count(*) FROM documents d WHERE {match_sql}",
                match_params,
                False,
            )
        columns = "d.content_hash, d.source_file" + (
            ", d.summary" if op is Operation.OVERVIEW else ""
        )
        sql = f"SELECT {columns} FROM documents d WHERE {match_sql} ORDER BY d.source_file LIMIT %s"
        return sql, (*match_params, plan.limit), True

    # ----------------------------------------------------------------- filters

    @staticmethod
    def _key(name: str, usable: dict[str, MetaKey], where: str) -> MetaKey:
        key = usable.get(name)
        if key is None:
            raise PlanError(
                f"{where}: unknown or unapproved key {name!r}; usable keys: {sorted(usable)}"
            )
        return key

    def _filter(self, flt: Filter, usable: dict[str, MetaKey]) -> "_CompiledFilter":
        key = self._key(flt.key, usable, "filter")
        if flt.op not in _ALLOWED_OPS[key.value_type]:
            allowed = sorted(o.value for o in _ALLOWED_OPS[key.value_type])
            raise PlanError(
                f"operator {flt.op.value!r} does not fit {key.value_type.value} key {key.key!r}; use one of {allowed}"
            )
        column = f"m.{_VALUE_COLUMN[key.value_type]}"
        condition, params, words = self._condition(flt, key, column)
        exists = (
            "EXISTS (SELECT 1 FROM document_meta m WHERE m.document_id = d.id "
            f"AND m.key = %s AND {condition})"
        )
        known = (
            "EXISTS (SELECT 1 FROM document_meta_status s WHERE s.document_id = d.id "
            f"AND s.key = %s AND s.key_version >= %s AND s.state IN {_KNOWN_STATES})"
        )
        return _CompiledFilter(
            satisfied_sql=exists,
            satisfied_params=(key.key, *params),
            known_sql=known,
            known_params=(key.key, key.version),
            words=f"{key.key} {words}",
        )

    def _condition(
        self, flt: Filter, key: MetaKey, column: str
    ) -> tuple[str, tuple[Any, ...], str]:
        kind = key.value_type
        if kind is ValueType.TEXT:
            return self._text(flt, key, column)
        if kind is ValueType.NUMBER:
            return self._number(flt, column)
        if kind is ValueType.DATE:
            return self._date(flt, column)
        if not isinstance(flt.value, bool):
            raise PlanError(
                f"{key.key!r} is a bool key: the value must be true or false"
            )
        return f"{column} = %s", (flt.value,), f"is {str(flt.value).lower()}"

    @staticmethod
    def _text(
        flt: Filter, key: MetaKey, column: str
    ) -> tuple[str, tuple[Any, ...], str]:
        def check(value: object) -> str:
            if not isinstance(value, str) or not value.strip():
                raise PlanError(
                    f"{key.key!r}: a text value must be a non-empty string, got {value!r}"
                )
            if (
                key.allowed_values is not None
                and flt.op is not FilterOp.CONTAINS
                and value not in key.allowed_values
            ):
                raise PlanError(
                    f"{key.key!r}: {value!r} is not an allowed value; use one of {list(key.allowed_values)}"
                )
            return value

        if flt.op is FilterOp.IN:
            if not isinstance(flt.value, list) or not flt.value:
                raise PlanError(f"{key.key!r}: 'in' needs a non-empty list")
            values = [check(v) for v in flt.value]
            return (
                f"rtrim({column}, '{_TRAILING}') = ANY(%s)",
                ([v.rstrip(_TRAILING) for v in values],),
                f"in {values}",
            )
        value = check(flt.value)
        if flt.op is FilterOp.CONTAINS:
            value = value.rstrip(_TRAILING) or value
            escaped = (
                value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            return (
                f"{column} ILIKE %s ESCAPE '\\'",
                (f"%{escaped}%",),
                f"contains {value!r}",
            )
        symbol = _COMPARISON[flt.op]
        if flt.op is FilterOp.EQ:
            return (
                f"rtrim({column}, '{_TRAILING}') = %s",
                (value.rstrip(_TRAILING),),
                f"= {value!r}",
            )
        return f"{column} {symbol} %s", (value,), f"{symbol} {value!r}"

    @staticmethod
    def _number(flt: Filter, column: str) -> tuple[str, tuple[Any, ...], str]:
        def number(value: object) -> Decimal:
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise PlanError(f"a number filter needs a number, got {value!r}")
            try:
                return Decimal(str(value))
            except InvalidOperation:
                raise PlanError(f"{value!r} is not a number") from None

        if flt.op is FilterOp.BETWEEN:
            if not isinstance(flt.value, list) or len(flt.value) != 2:
                raise PlanError("'between' on a number needs [low, high]")
            low, high = number(flt.value[0]), number(flt.value[1])
            if low > high:
                raise PlanError(f"'between' needs low <= high, got [{low}, {high}]")
            return (
                f"{column} BETWEEN %s AND %s",
                (low, high),
                f"between {low} and {high}",
            )
        value = number(flt.value)
        symbol = _COMPARISON[flt.op]
        return f"{column} {symbol} %s", (value,), f"{symbol} {value}"

    def _date(self, flt: Filter, column: str) -> tuple[str, tuple[Any, ...], str]:
        if flt.op is FilterOp.IN:
            # Several separate periods ("2021 and 2023"): any one of them. Two filters
            # on the same key could never both hold, so this is one filter.
            if not isinstance(flt.value, list) or not flt.value:
                raise PlanError("a date 'in' needs a non-empty list of date specs")
            spans = [self._range(v) for v in flt.value]
            return (
                "(" + " OR ".join(f"{column} BETWEEN %s AND %s" for _ in spans) + ")",
                tuple(bound for span in spans for bound in (span.start, span.end)),
                "in [" + ", ".join(f"{span.start}..{span.end}" for span in spans) + "]",
            )
        span = self._range(flt.value)
        if flt.op in (FilterOp.EQ, FilterOp.BETWEEN):
            return (
                f"{column} BETWEEN %s AND %s",
                (span.start, span.end),
                f"between {span.start} and {span.end}",
            )
        # a comparison against a whole range: after it, from its start, before it, up to its end
        bound, symbol = {
            FilterOp.GT: (span.end, ">"),
            FilterOp.GTE: (span.start, ">="),
            FilterOp.LT: (span.start, "<"),
            FilterOp.LTE: (span.end, "<="),
        }[flt.op]
        return f"{column} {symbol} %s", (bound,), f"{symbol} {bound}"

    def _range(self, value: object) -> DateRange:
        if isinstance(value, str):  # shorthand for one day
            try:
                day = date.fromisoformat(value)
            except ValueError:
                raise PlanError(
                    f"{value!r} is not an ISO date (YYYY-MM-DD) or a date spec"
                ) from None
            return DateRange(day, day)
        try:
            return self._resolver.resolve(value)  # type: ignore[arg-type]
        except DateSpecError as exc:
            raise PlanError(f"invalid date: {exc}") from exc


@dataclass(frozen=True)
class _CompiledFilter:
    satisfied_sql: str
    satisfied_params: tuple[Any, ...]
    known_sql: str
    known_params: tuple[Any, ...]
    words: str
