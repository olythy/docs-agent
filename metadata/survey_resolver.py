"""Turning a "survey" question into an exact count, when the keys really cover it.

A question about the practice or the outcomes across many documents ("how did the court
decide the cases about X in 2020-2024") is a survey: the planner cannot answer it from the
metadata and the system cannot read that many documents. But when the catalog has a key
whose stored values name the topic (a classification of the cases) and another key that
holds what the question asks about (how they ended), the question *is* a count: the
documents of the topic, broken down by the second key. That is exact, complete (every
matching document is counted, with the ones that cannot be decided reported) and cheap.

The judgement "does a stored value name this topic?" is a judgement about the question's
content against the catalog, and a wrong one would produce a confident, exact-looking and
wrong figure. So it is not left to the planner's prompt (an attempt to do that cost
five of thirty survey routings, see ``docs/decisions.md``); it is one separate, narrow
model call, and its answer is checked by code before anything is built from it:

* the keys must be approved, text, closed-vocabulary and single-valued;
* every value must be one the key defines, and never the catch-all ``other``;
* the topic key and the key to group by must differ;
* the topic filter replaces the planner's own filter on that key (the planner sees the
  catalog too and often adds one); the planner's other restrictions are kept.

Anything that fails ends in ``None``, which the decider turns into the plain "not supported
yet", never into a number.

Key exports:
    SurveyResolver     -- The contract.
    LLMSurveyResolver  -- The model-backed implementation.
    CATCH_ALL_VALUE    -- The stored value that never names a topic.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from drivers.llm import AnswerDriver
from llm_json import extract_json
from metadata.plan import DEFAULT_LIMIT, Filter, FilterOp, Operation, QueryPlan
from metadata.planner import TypeCatalog
from models import MetaKey, ValueType

logger = logging.getLogger(__name__)

#: A closed-vocabulary key's catch-all ("none of the above"). It never names a topic: a
#: count of the "other" documents is not a count of anything the question asked about.
CATCH_ALL_VALUE = "other"

#: What every resolved answer says about its figures, whatever the model said.
EXTRACTED_BY_MODEL = (
    "the documents are counted by values a model extracted from each document and "
    "checked only against a quotation from it, not by hand"
)

_MAX_UNANSWERED = 300


class SurveyResolver(Protocol):
    """Turns a survey plan into an exact one, or says it cannot."""

    def resolve(
        self, question: str, plan: QueryPlan, catalogs: Sequence[TypeCatalog]
    ) -> QueryPlan | None: ...


def usable_keys(catalog: TypeCatalog) -> dict[str, MetaKey]:
    """The keys a survey can be resolved with: closed vocabulary, text, one value each."""
    return {
        k.key: k
        for k in catalog.keys
        if k.value_type is ValueType.TEXT and k.allowed_values and not k.multi_valued
    }


class LLMSurveyResolver:
    """Asks one focused model call which keys and values a survey question is about.

    Args:
        llm: The driver used for the call.
        temperature: Sampling temperature (0: a judgement, not a draft).
    """

    def __init__(self, llm: AnswerDriver, temperature: float = 0.0) -> None:
        self._llm = llm
        self._temperature = temperature

    def resolve(
        self, question: str, plan: QueryPlan, catalogs: Sequence[TypeCatalog]
    ) -> QueryPlan | None:
        """Return an exact ``count`` plan for ``question``, or ``None``.

        Args:
            question: The user's question.
            plan: The planner's ``survey`` plan: the type, and the filters (court, period)
                the question states.
            catalogs: The approved types with their approved keys.
        """
        if plan.residual:
            # Part of the question is covered by no key: a count would answer an easier
            # question than the one asked and present it as the answer.
            return None
        catalog = next((c for c in catalogs if c.doc_type.type == plan.doc_type), None)
        if catalog is None:
            return None
        keys = usable_keys(catalog)
        if not keys:
            return None
        reply = self._ask(question, plan, keys)
        if reply is None:
            return None
        return self._build(plan, keys, reply)

    def _ask(
        self, question: str, plan: QueryPlan, keys: dict[str, MetaKey]
    ) -> dict | None:
        """The model's answer as a dict, or ``None`` if there is no usable JSON object."""
        turn = self._llm.run_tool_calling_turn(
            [{"role": "user", "content": self._prompt(question, plan, keys)}],
            temperature=self._temperature,
        )
        try:
            data = extract_json(turn.content or "", reject_duplicate_keys=True)
        except (ValueError, json.JSONDecodeError):
            logger.info("[survey] The resolver's reply was not JSON.")
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _build(
        plan: QueryPlan, keys: dict[str, MetaKey], reply: dict
    ) -> QueryPlan | None:
        """The checked count plan the reply describes, or ``None``."""
        topic_key = reply.get("topic_key")
        verdicts = reply.get("topic_values")
        group_by = reply.get("group_by")
        if not isinstance(topic_key, str) or topic_key not in keys:
            return None
        # One true/false verdict per stored value, so no value is skipped without a word.
        if not isinstance(verdicts, dict):
            return None
        allowed = set(keys[topic_key].allowed_values or ())
        if not all(v in allowed and isinstance(j, bool) for v, j in verdicts.items()):
            return None
        values = [v for v, named in verdicts.items() if named]
        if not values or CATCH_ALL_VALUE in values:
            return None
        if group_by is not None and (
            not isinstance(group_by, str)
            or group_by not in keys
            or group_by == topic_key
        ):
            return None
        unanswered = reply.get("unanswered")
        caveat = EXTRACTED_BY_MODEL
        if isinstance(unanswered, str) and unanswered.strip():
            caveat += (
                f"; not answered by the figures: {unanswered.strip()[:_MAX_UNANSWERED]}"
            )
        return replace(
            plan,
            operation=Operation.COUNT,
            filters=(
                *(f for f in plan.filters if f.key != topic_key),
                Filter(topic_key, FilterOp.IN, list(dict.fromkeys(values))),
            ),
            group_by=group_by,
            limit=DEFAULT_LIMIT,
            residual=None,
            reason=None,
            caveat=caveat,
        )

    @staticmethod
    def _prompt(question: str, plan: QueryPlan, keys: dict[str, MetaKey]) -> str:
        """The resolver prompt: the question and the closed-vocabulary keys."""
        described = []
        for k in keys.values():
            values = ", ".join(k.allowed_values or ())
            described.append(f"- {k.key}: {k.description}\n  Stored values: {values}")
        restricted = ", ".join(f.key for f in plan.filters) or "none"
        return (
            "A question was asked about a collection of documents. Decide whether it can be "
            "answered EXACTLY by counting documents: the documents about the question's "
            "topic, optionally broken down by one key. You may only use the keys below, "
            "and only their stored values.\n\n"
            "KEYS (closed lists of stored values)\n"
            + "\n".join(described)
            + "\n\nRULES\n"
            "- topic_key / topic_values: the key whose stored values NAME what the cases in "
            "the question are about, and for EVERY stored value of that key a true or "
            "false verdict: true if the cases it describes are about any topic the question "
            "asks about (a value that covers several topics at once is true when the "
            "question asks about any of them), false otherwise. The catch-all value "
            "(other) and a value that is merely related or broader are false. If ANY topic "
            'the question asks about has no stored value at all, answer with "topic_key": '
            "null: a count would silently leave that topic out.\n"
            "- group_by: the other key whose values the question asks to break the cases "
            "down by. A question about how the court decided, the practice, the results or "
            "the outcomes of the cases asks for the key that holds how the cases ended: use "
            "it if there is one. null only when the question just asks how many.\n"
            "- unanswered: in the question's language, a short phrase naming only what the "
            'count cannot answer, such as "the reasons", "the content of the rulings" or '
            '"the change from year to year"; never the question itself, never what the '
            "count does answer; null when the count answers all of it.\n"
            "- If the question cannot be answered by a count (it needs the content of the "
            'rulings, or no key fits), answer with "topic_key": null.\n'
            f"- The question's court, period and similar restrictions are applied "
            f"separately (already restricted: {restricted}); do not repeat them.\n"
            'Output ONLY one JSON object: {"topic_key": "<key>" or null, '
            '"topic_values": {"<stored value>": true or false, ...}, "group_by": "<key>" or null, '
            '"unanswered": "<text>" or null}\n\n'
            f"QUESTION\n{question}\n"
        )
