"""Turning a question into a query plan.

A :class:`QueryPlanner` reads a natural-language question and the catalog of
queryable keys and returns a :class:`metadata.plan.QueryPlan`. The planner is the
*router*: ``lookup`` means "this needs the normal retrieval pipeline", anything
else is answered exactly from the structured metadata.

:class:`LLMQueryPlanner` asks a language model for the plan as JSON. It never
trusts the reply: the plan is parsed and compiled against the catalog (both pure,
no database), and if that fails the model is shown the exact error once and asked
to correct it. If the second attempt also fails the planner raises
:class:`PlanningFailed`; there is no silent fallback to some default plan, because
a confident wrong count is worse than saying "I could not interpret this".

Key exports:
    QueryPlanner    -- The contract.
    LLMQueryPlanner -- The model-backed planner.
    PlanningFailed  -- The question could not be turned into a valid plan.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from drivers.llm import AnswerDriver
from llm_json import extract_json
from metadata.clock import Clock
from metadata.compiler import PlanCompiler
from metadata.plan import Operation, PlanError, QueryPlan, parse_plan
from models import DocumentType, KeyStatus, MetaKey, TypeStatus, ValueType

#: A text key with more distinct values than this is too free-form to list in a prompt.
MAX_LISTED_VALUES = 60


@dataclass(frozen=True)
class TypeCatalog:
    """A document type together with its approved keys: what a plan about it may use.

    Attributes:
        doc_type: The type (its name and description tell the planner what it is).
        keys: Its approved keys.
    """

    doc_type: DocumentType
    keys: list[MetaKey]


class CatalogSource(Protocol):
    """The slice of :class:`document_store.DocumentStore` that lists types and keys."""

    def list_types(self, status: TypeStatus | None = None) -> list[DocumentType]: ...
    def list_keys(
        self, doc_type: str, status: KeyStatus | None = None
    ) -> list[MetaKey]: ...


class ValueSource(Protocol):
    """The slice of :class:`document_store.DocumentStore` that lists stored values."""

    def distinct_text_values(
        self, key: str, limit: int, doc_type: str | None = None
    ) -> list[str] | None: ...


def load_catalogs(source: CatalogSource) -> list[TypeCatalog]:
    """Every approved document type with its approved keys.

    Raises:
        RuntimeError: If no document type is approved yet (nothing to plan over).
    """
    types = source.list_types(TypeStatus.APPROVED)
    if not types:
        raise RuntimeError(
            "no approved document types; run `meta_cli.py load-catalog` first"
        )
    return [TypeCatalog(t, source.list_keys(t.type, KeyStatus.APPROVED)) for t in types]


def collect_known_values(
    source: ValueSource, catalogs: Sequence[TypeCatalog]
) -> dict[str, dict[str, list[str]]]:
    """The exact stored values of every low-cardinality free-text key, per type.

    A planner that has not seen the stored spellings guesses them (and splits a
    combined name into two). Keys with an allowed-value list already say what they
    take, and keys with many distinct values are left out. Values are read per
    document type, because two types may have a key of the same name.

    Args:
        source: Where the stored values are read.
        catalogs: The types and their approved keys.

    Returns:
        ``type -> key -> values``.
    """
    known: dict[str, dict[str, list[str]]] = {}
    for catalog in catalogs:
        for key in catalog.keys:
            if key.value_type is ValueType.TEXT and not key.allowed_values:
                values = source.distinct_text_values(
                    key.key, MAX_LISTED_VALUES, catalog.doc_type.type
                )
                if values:
                    known.setdefault(catalog.doc_type.type, {})[key.key] = values
    return known


class PlanningFailed(RuntimeError):
    """The question could not be turned into a valid plan.

    Attributes:
        reason: The last validation error (or why the reply was unusable).
        reply: The model's last reply, for diagnostics.
    """

    def __init__(self, reason: str, reply: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.reply = reply


class QueryPlanner(ABC):
    """Turns a question into a plan over a catalog."""

    @abstractmethod
    def plan(
        self,
        question: str,
        catalogs: Sequence[TypeCatalog],
        known_values: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    ) -> QueryPlan:
        """Return a plan for ``question``.

        Args:
            question: The user's question, in any language.
            catalogs: The document types the plan may be about, each with its
                approved keys; the planner chooses the type.
            known_values: Exact stored values of low-cardinality text keys per type
                (see :func:`collect_known_values`), so a filter can copy them.

        Raises:
            PlanningFailed: If no valid plan could be produced.
        """


class LLMQueryPlanner(QueryPlanner):
    """Asks a language model for a plan, validates it, and retries once on an error.

    Args:
        llm: The driver used for the call.
        compiler: Validates a plan against the catalog (the same one that later
            compiles it, so a plan that passes here cannot fail there).
        clock: Today's date, given to the model so it can tell what "now" is
            (the date arithmetic itself is done by the resolver, not the model).
    """

    def __init__(self, llm: AnswerDriver, compiler: PlanCompiler, clock: Clock) -> None:
        self._llm = llm
        self._compiler = compiler
        self._clock = clock

    def plan(
        self,
        question: str,
        catalogs: Sequence[TypeCatalog],
        known_values: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    ) -> QueryPlan:
        messages: list[dict] = [
            {
                "role": "user",
                "content": self._prompt(question, catalogs, known_values or {}),
            }
        ]
        type_names = {c.doc_type.type for c in catalogs}
        keys = [k for c in catalogs for k in c.keys]
        error, reply = "", ""
        for attempt in range(2):
            reply = self._llm.run_tool_calling_turn(messages).content or ""
            try:
                plan = parse_plan(
                    extract_json(reply, reject_duplicate_keys=True), type_names
                )
                if plan.operation is not Operation.UNSUPPORTED:
                    self._compiler.compile(plan, keys)  # nothing to compile otherwise
                return plan
            except (PlanError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
            if attempt == 0:
                messages += [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": (
                            f"That plan was rejected: {error}\n"
                            "Return a corrected plan as ONLY a JSON object."
                        ),
                    },
                ]
        raise PlanningFailed(error, reply)

    def _prompt(
        self,
        question: str,
        catalogs: Sequence[TypeCatalog],
        known_values: Mapping[str, Mapping[str, Sequence[str]]],
    ) -> str:
        blocks = []
        for catalog in catalogs:
            t = catalog.doc_type
            lines = [f'- "{t.type}": {t.name}. {t.description}']
            if catalog.keys:
                lines.append("  Keys (usable only in a plan about this type):")
            else:
                lines.append("  (no keys: it can only be counted, listed or read)")
            for k in catalog.keys:
                line = f"    - {k.key} ({k.value_type.value}): {k.description}"
                if k.allowed_values:
                    line += f" Allowed values: {', '.join(k.allowed_values)}."
                stored = known_values.get(t.type, {}).get(k.key)
                if stored:
                    line += (
                        " Stored values (copy one exactly; a combined name is ONE value): "
                        + "; ".join(stored)
                        + "."
                    )
                lines.append(line)
            blocks.append("\n".join(lines))
        only = catalogs[0].doc_type.type if len(catalogs) == 1 else None
        type_rule = (
            f'Only ONE document type exists ("{only}"): use it whenever the question '
            'counts, lists, sums or summarises documents, even when it just says "documents".'
            if only
            else "If the question counts or lists documents without saying which kind and "
            'no single listed kind clearly fits, use null and "lookup".'
        )
        return _TEMPLATE.format(
            today=self._clock.today().isoformat(),
            types="\n".join(blocks),
            type_rule=type_rule,
            question=question,
        )


_TEMPLATE = """You turn a question about a collection of documents into a JSON query plan.
The plan is executed exactly over structured per-document metadata. You do NOT write SQL and you do NOT do date arithmetic.

Today is {today}.

OPERATIONS
- "count": how many documents match (optionally "group_by": a key with a limited set of values, to break the count down).
- "list": which documents match.
- "sum": the total of a number key ("sum_key") over the matching documents.
- "overview": the matching documents with their summaries (for "what kinds of ... are there / summarise").
- "unsupported": ONLY when the question asks for documents SIMILAR or RELATED to a specific named document or case ("list five cases similar to case X", "find cases like this one", "hasonló ügyeket"). The system cannot do that yet. Give a short "reason" in the question's language saying what was asked. Do NOT use it for ordinary questions about the content of a document, for comparing named documents, or for questions about the practice in general: those are "lookup".
- "lookup": the question asks about the CONTENT of specific documents (who, why, what did the court decide, what does clause X say), or names a case/document identifier, or cannot be answered from the keys below. The filters then only narrow down which documents to read; use [] when the question names no key-based restriction. NEVER put a case or document identifier into the filters of a lookup: the retrieval finds a named case by itself, and an exact match on its written form is brittle (a trailing full stop or a different suffix finds nothing, and then nothing is read). An identifier filter is right only to count or list documents by their number.

DOCUMENT TYPES (the kinds of document that exist, and the keys of each)
{types}

Choose ONE document type that the question is about and put its exact name in "document_type". The filters, "group_by" and "sum_key" may use only the keys of that type. "count", "list", "sum" and "overview" ALWAYS need a document type: an exact answer is only possible for one kind of document. {type_rule} Use null ONLY for a question about the CONTENT of documents (what a ruling says, why, who ...) that names no listed type; then the operation must be "lookup" and "filters" must be [].

FILTERS: a list of {{"key", "op", "value"}} that must ALL hold.
- ops: eq, ne, in (value is a list; for a date key, a list of date specs), contains (text substring), gt, gte, lt, lte, between (value is [low, high]).
- Text values are copied in the language of the documents (the question's own wording, e.g. a court's name). Keys with allowed values take exactly one of those English tokens.
- Number values are plain numbers (no separators, no currency).
- Date keys take a date spec instead of a value, in this closed grammar:
  {{"kind": "calendar", "year": 2022, "month": 3}}          a month (omit "month" for a whole year; "quarter": 1-4; "day": 1-31)
  {{"kind": "calendar", "year_offset": -1, "month": 10}}    October of last year (year_offset is relative to today's year)
  {{"kind": "relative", "unit": "day|week|month|quarter|year", "offset": 0}}   this (0), last (-1), next (1) calendar unit
  {{"kind": "rolling", "unit": "day|week|month|year", "count": 30, "direction": "past|future"}}   the last/next N units including today
  {{"kind": "absolute", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
  {{"kind": "between", "from": <spec>, "to": <spec>}}
  A named period ("last year", "March 2023", "last quarter", "next week") is ALWAYS a closed range: the filter is {{"key": <date key>, "op": "between", "value": <ONE spec from the list above>}}, e.g. last year = {{"key": "decision_date", "op": "between", "value": {{"kind": "relative", "unit": "year", "offset": -1}}}}. Every spec, including the value of "between", has a "kind". The "between" KIND is only for spanning two different specs ("from March to May 2023"). "The last N days/weeks/months" is a rolling spec. Use "gte"/"lt" etc. with a spec only for open-ended questions ("after March 2023", "before 2020"). SEVERAL SEPARATE periods of the same date key ("in 2021 and in 2023", "in March and in June") are ONE filter with op "in" and a list of specs, e.g. {{"key": <date key>, "op": "in", "value": [{{"kind": "calendar", "year": 2021}}, {{"kind": "calendar", "year": 2023}}]}}. NEVER write two filters on the same date key for them: all filters must hold at once, so two different periods would match no document. A span from one period to another ("from 2020 to 2022", "between 2020 and 2022") is not a set: it is one "between" range.

RULES
- Use only the keys of the chosen type. If part of the question has no matching key, put that part in "residual" (a short string in the question's language) and keep the rest as filters.
- Never invent allowed values or keys.
- "limit": at most how many documents to return (default 50).
- Output ONLY one JSON object: {{"document_type": "<a type above>" or null, "operation": ..., "filters": [...], "group_by": null, "sum_key": null, "limit": 50, "residual": null, "reason": null}}

QUESTION
{question}
"""
