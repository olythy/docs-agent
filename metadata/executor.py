"""Running a query plan end to end: compile, execute, shape the result.

The thin orchestrator between a validated plan and an answer's raw material. It
compiles the plan against the catalog, runs the queries through a read-only
store, and returns a :class:`PlanResult` that always carries the echo of the
filter and the number of documents that could not be decided, so whatever phrases
the answer cannot present a bare count as if it were certain.

Key exports:
    PlanExecutor -- Runs a plan.
    PlanResult   -- What came back.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from metadata.compiler import PlanCompiler
from metadata.plan import Operation, QueryPlan
from models import DocumentSelection, KeyStatus, MetaKey


class PlanStore(Protocol):
    """The slice of :class:`document_store.DocumentStore` the executor uses."""

    def list_keys(
        self, doc_type: str, status: KeyStatus | None = None
    ) -> list[MetaKey]: ...
    def execute_query(
        self, sql: str, params: tuple, timeout_ms: int = ...
    ) -> list[tuple]: ...


@dataclass(frozen=True)
class PlanResult:
    """The outcome of running a plan.

    Attributes:
        operation: What the plan asked for.
        explanation: The executed filter in words (show it to the user).
        count: The exact number of matching documents (``count`` and the totals of the
            list-like operations); ``None`` for ``sum``.
        unknown: Documents that might match but cannot be decided because a filtered
            key is unverified or was never extracted. A count must be read as
            ``count`` *plus up to* ``unknown``.
        documents: For list-like operations, ``(content_hash, source_file[, summary])``
            rows, at most the plan's limit.
        groups: For a grouped count, ``(value, count)`` pairs.
        total: The sum, for ``sum``.
        sum_documents: How many documents contributed to the sum.
        truncated: The list holds fewer documents than ``count``.
        selection: For ``lookup``, the matching documents as a sub-select to restrict
            a retrieval to (see :class:`models.DocumentSelection`); ``count`` is how
            many they are. The documents themselves are not fetched.
    """

    operation: Operation
    explanation: str
    count: int | None
    unknown: int
    documents: tuple[tuple[Any, ...], ...] = ()
    groups: tuple[tuple[Any, int], ...] = ()
    total: Decimal | None = None
    sum_documents: int | None = None
    truncated: bool = False
    selection: DocumentSelection | None = None


class PlanExecutor:
    """Compiles and runs plans.

    Args:
        store: Where the catalog lives and queries run.
        compiler: Turns a plan into SQL.
    """

    def __init__(self, store: PlanStore, compiler: PlanCompiler) -> None:
        self._store = store
        self._compiler = compiler

    def execute(self, plan: QueryPlan) -> PlanResult:
        """Run ``plan``.

        Raises:
            metadata.plan.PlanError: If the plan does not fit the catalog.
        """
        keys = self._store.list_keys(plan.doc_type, KeyStatus.APPROVED)
        query = self._compiler.compile(plan, keys)
        unknown = 0
        if query.unknown_sql is not None:
            unknown = self._store.execute_query(
                query.unknown_sql, query.unknown_params
            )[0][0]
        rows = self._store.execute_query(query.sql, query.params)

        op = plan.operation
        if op is Operation.COUNT and plan.group_by is None:
            return PlanResult(op, query.explanation, count=rows[0][0], unknown=unknown)
        if op is Operation.COUNT:
            assert query.count_sql is not None
            total_documents = self._store.execute_query(
                query.count_sql, query.count_params
            )[0][0]
            groups = tuple((value, n) for value, n in rows)
            return PlanResult(
                op,
                query.explanation,
                count=total_documents,
                unknown=unknown,
                groups=groups,
            )
        if op is Operation.LOOKUP:
            return PlanResult(
                op,
                query.explanation,
                count=rows[0][0],
                unknown=unknown,
                selection=query.selection,
            )
        if op is Operation.SUM:
            total, contributing = rows[0]
            return PlanResult(
                op,
                query.explanation,
                count=None,
                unknown=unknown,
                total=total,
                sum_documents=contributing,
            )
        assert query.count_sql is not None
        count = self._store.execute_query(query.count_sql, query.count_params)[0][0]
        return PlanResult(
            op, query.explanation, count=count, unknown=unknown,
            documents=tuple(tuple(r) for r in rows), truncated=len(rows) < count,
        )  # fmt: skip
