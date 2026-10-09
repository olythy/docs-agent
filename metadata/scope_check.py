"""Checking that an exact plan's topic filters are not wider than the question.

A plan that counts or lists documents is exact only if its filters select exactly the
documents the question is about. A filter on a key with a closed list of values can fail
that quietly: the value that fits best may be a whole *family* of cases (a "company law"
value for a question about one kind of company decision), so the exact figure or list is a
superset of what was asked, presented as the answer. The planner makes this choice inside a
prompt that already carries many rules, and a rule added there did not change its plans
(``docs/decisions.md``, 2026-10-10); the survey resolver, a separate and checked call, does
not have the problem. This is the same remedy for the plans the planner makes itself.

One narrow model call looks at the question and at the values a plan selected on each such
key, and names the keys whose values are wider than the question's topic, with the narrower
topic in the question's language. Code accepts only what it can check (a key the plan really
filters on, a non-empty phrase) and puts that topic into the plan's ``residual``, which the
decider already reads as "a part of the question no key covers": the plan is then *read*
instead of counted, the filters still narrowing the documents. If the call gives no usable
answer the plan is read as well: a wrong exact figure is worse than a read answer.

Key exports:
    PlanChecker       -- The contract.
    LLMPlanChecker    -- The model-backed implementation.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from drivers.llm import AnswerDriver
from llm_json import extract_json
from metadata.plan import FilterOp, Operation, QueryPlan
from metadata.planner import TypeCatalog
from models import MetaKey, ValueType

logger = logging.getLogger(__name__)

#: The exact operations whose filters are checked (a lookup is read anyway).
CHECKED_OPERATIONS = frozenset(
    {Operation.COUNT, Operation.LIST, Operation.SUM, Operation.OVERVIEW}
)

_MAX_TOPIC = 200

#: Said in the plan's residual when the call gave no usable answer.
UNCHECKED = "the plan's topic filter could not be checked against the question"


class PlanChecker(Protocol):
    """Looks at an exact plan and returns it, or a read version of it."""

    def check(
        self, question: str, plan: QueryPlan, catalogs: Sequence[TypeCatalog]
    ) -> QueryPlan: ...


class LLMPlanChecker:
    """Asks one focused model call whether a plan's closed-value filters are too wide.

    Args:
        llm: The driver used for the call.
        temperature: Sampling temperature (0: a judgement, not a draft).
    """

    def __init__(self, llm: AnswerDriver, temperature: float = 0.0) -> None:
        self._llm = llm
        self._temperature = temperature

    def check(
        self, question: str, plan: QueryPlan, catalogs: Sequence[TypeCatalog]
    ) -> QueryPlan:
        """Return ``plan`` if its topic filters fit the question, otherwise a plan to read.

        Args:
            question: The user's question.
            plan: The planner's plan.
            catalogs: The approved types with their approved keys.
        """
        if plan.operation not in CHECKED_OPERATIONS or plan.residual:
            return plan
        catalog = next((c for c in catalogs if c.doc_type.type == plan.doc_type), None)
        if catalog is None:
            return plan
        selected = self._selected_values(plan, catalog)
        if not selected:
            return plan  # no closed-value topic filter: nothing here can be too wide
        reply = self._ask(question, selected)
        if reply is None:
            logger.warning("[check] No usable answer; the plan will be read.")
            return replace(plan, residual=UNCHECKED)
        topics = self._wider_topics(reply, selected)
        if not topics:
            return plan
        logger.info(
            "[check] A filter is wider than the question (%s): read instead of counted.",
            ", ".join(selected.keys() & {k for k, _ in topics}),
        )
        return replace(plan, residual="; ".join(topic for _, topic in topics))

    @staticmethod
    def _selected_values(
        plan: QueryPlan, catalog: TypeCatalog
    ) -> dict[str, tuple[MetaKey, list[str]]]:
        """The values the plan selects on each closed-vocabulary text key."""
        keys = {
            k.key: k
            for k in catalog.keys
            if k.value_type is ValueType.TEXT and k.allowed_values
        }
        selected: dict[str, tuple[MetaKey, list[str]]] = {}
        for f in plan.filters:
            if f.key not in keys or f.op not in (FilterOp.EQ, FilterOp.IN):
                continue
            values = f.value if isinstance(f.value, list) else [f.value]
            names = [v for v in values if isinstance(v, str)]
            if names:
                selected[f.key] = (keys[f.key], names)
        return selected

    def _ask(
        self, question: str, selected: dict[str, tuple[MetaKey, list[str]]]
    ) -> dict | None:
        """The model's answer as a dict, or ``None`` if there is no usable JSON object."""
        turn = self._llm.run_tool_calling_turn(
            [{"role": "user", "content": self._prompt(question, selected)}],
            temperature=self._temperature,
        )
        try:
            data = extract_json(turn.content or "", reject_duplicate_keys=True)
        except (ValueError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _wider_topics(
        reply: dict, selected: dict[str, tuple[MetaKey, list[str]]]
    ) -> list[tuple[str, str]]:
        """``(key, narrower topic)`` for each key the reply names that the plan filters on."""
        wider = reply.get("wider")
        if not isinstance(wider, list):
            return []
        topics: list[tuple[str, str]] = []
        for entry in wider:
            if not isinstance(entry, dict):
                continue
            key, topic = entry.get("key"), entry.get("topic")
            if key in selected and isinstance(topic, str) and topic.strip():
                topics.append((key, topic.strip()[:_MAX_TOPIC]))
        return topics

    @staticmethod
    def _prompt(question: str, selected: dict[str, tuple[MetaKey, list[str]]]) -> str:
        """The checker prompt: the question and, per key, what it means and what was selected."""
        blocks = []
        for name, (key, values) in selected.items():
            blocks.append(
                f"- {name}: {key.description}\n  Selected values: {', '.join(values)}"
            )
        return (
            "A question about a collection of documents was turned into a plan that selects "
            "documents by the key values below and then counts or lists them EXACTLY. For "
            "each key decide whether the selected values name exactly the kind of case the "
            "question asks about, or whether they also take in other kinds of cases (a whole "
            "family of cases, a broader category), so that the exact result would be a "
            "superset of what was asked.\n\n"
            "SELECTED\n" + "\n".join(blocks) + "\n\nRULES\n"
            '- List a key under "wider" only if the question names a narrower kind of case '
            "than the selected values describe, and the key cannot express that narrower "
            'kind. For each, give in "topic" a short phrase in the question\'s language '
            "naming the narrower kind of case.\n"
            "- Do not list a key whose selected values match the question's topic, even "
            "when they are worded differently. Do not list a key because the question also "
            "asks for more than the key holds (a count, an outcome, a year).\n"
            'Output ONLY one JSON object: {"wider": [{"key": "<key>", "topic": "<text>"}]} '
            "with an empty list when no key is too wide.\n\n"
            f"QUESTION\n{question}\n"
        )
