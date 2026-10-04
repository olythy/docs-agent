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
from typing import Protocol

from drivers.llm import AnswerDriver
from llm_json import extract_json
from metadata.clock import Clock
from metadata.compiler import PlanCompiler
from metadata.plan import PlanError, QueryPlan, parse_plan
from models import MetaKey, ValueType

#: A text key with more distinct values than this is too free-form to list in a prompt.
MAX_LISTED_VALUES = 60


class ValueSource(Protocol):
    """The slice of :class:`document_store.DocumentStore` that lists stored values."""

    def distinct_text_values(self, key: str, limit: int) -> list[str] | None: ...


def collect_known_values(
    source: ValueSource, keys: Sequence[MetaKey]
) -> dict[str, list[str]]:
    """The exact stored values of every low-cardinality free-text key.

    A planner that has not seen the stored spellings guesses them (and splits a
    combined name into two). Keys with an allowed-value list already say what they
    take, and keys with many distinct values are left out.

    Args:
        source: Where the stored values are read.
        keys: The approved keys the plan may use.
    """
    known: dict[str, list[str]] = {}
    for key in keys:
        if key.value_type is ValueType.TEXT and not key.allowed_values:
            values = source.distinct_text_values(key.key, MAX_LISTED_VALUES)
            if values:
                known[key.key] = values
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
        doc_type: str,
        keys: list[MetaKey],
        known_values: Mapping[str, Sequence[str]] | None = None,
    ) -> QueryPlan:
        """Return a plan for ``question``.

        Args:
            question: The user's question, in any language.
            doc_type: The catalog the keys belong to.
            keys: The *approved* keys the plan may use.
            known_values: Exact stored values of low-cardinality text keys
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
        doc_type: str,
        keys: list[MetaKey],
        known_values: Mapping[str, Sequence[str]] | None = None,
    ) -> QueryPlan:
        messages: list[dict] = [
            {
                "role": "user",
                "content": self._prompt(question, keys, known_values or {}),
            }
        ]
        error, reply = "", ""
        for attempt in range(2):
            reply = self._llm.run_tool_calling_turn(messages).content or ""
            try:
                plan = parse_plan(extract_json(reply), doc_type)
                self._compiler.compile(plan, keys)
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
        keys: list[MetaKey],
        known_values: Mapping[str, Sequence[str]],
    ) -> str:
        described = []
        for k in keys:
            line = f"- {k.key} ({k.value_type.value}): {k.description}"
            if k.allowed_values:
                line += f" Allowed values: {', '.join(k.allowed_values)}."
            if k.key in known_values:
                line += (
                    " Stored values (copy one exactly; a combined name is ONE value): "
                    + "; ".join(known_values[k.key])
                    + "."
                )
            described.append(line)
        return _TEMPLATE.format(
            today=self._clock.today().isoformat(),
            keys="\n".join(described),
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
- "lookup": the question asks about the CONTENT of specific documents (who, why, what did the court decide, what does clause X say), or names a case/document identifier, or cannot be answered from the keys below. The filters then only narrow down which documents to read; use [] when the question names no key-based restriction.

KEYS (the only keys you may filter on; use these exact names)
{keys}

FILTERS: a list of {{"key", "op", "value"}} that must ALL hold.
- ops: eq, ne, in (value is a list), contains (text substring), gt, gte, lt, lte, between (value is [low, high]).
- Text values are copied in the language of the documents (the question's own wording, e.g. a court's name). Keys with allowed values take exactly one of those English tokens.
- Number values are plain numbers (no separators, no currency).
- Date keys take a date spec instead of a value, in this closed grammar:
  {{"kind": "calendar", "year": 2022, "month": 3}}          a month (omit "month" for a whole year; "quarter": 1-4; "day": 1-31)
  {{"kind": "calendar", "year_offset": -1, "month": 10}}    October of last year (year_offset is relative to today's year)
  {{"kind": "relative", "unit": "day|week|month|quarter|year", "offset": 0}}   this (0), last (-1), next (1) calendar unit
  {{"kind": "rolling", "unit": "day|week|month|year", "count": 30, "direction": "past|future"}}   the last/next N units including today
  {{"kind": "absolute", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
  {{"kind": "between", "from": <spec>, "to": <spec>}}
  A named period ("last year", "March 2023", "last quarter", "next week") is ALWAYS a closed range: the filter is {{"key": <date key>, "op": "between", "value": <ONE spec from the list above>}}, e.g. last year = {{"key": "decision_date", "op": "between", "value": {{"kind": "relative", "unit": "year", "offset": -1}}}}. Every spec, including the value of "between", has a "kind". The "between" KIND is only for spanning two different specs ("from March to May 2023"). "The last N days/weeks/months" is a rolling spec. Use "gte"/"lt" etc. with a spec only for open-ended questions ("after March 2023", "before 2020").

RULES
- Use only the keys above. If part of the question has no matching key, put that part in "residual" (a short string in the question's language) and keep the rest as filters.
- Never invent allowed values or keys.
- "limit": at most how many documents to return (default 50).
- Output ONLY one JSON object: {{"operation": ..., "filters": [...], "group_by": null, "sum_key": null, "limit": 50, "residual": null}}

QUESTION
{question}
"""
