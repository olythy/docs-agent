"""Tests for metadata.planner.LLMQueryPlanner (a scripted fake model, no network)."""

from datetime import date
from types import SimpleNamespace

import pytest

from metadata.clock import FixedClock
from metadata.compiler import PlanCompiler
from metadata.date_ranges import DateRangeResolver
from metadata.plan import Operation
from metadata.planner import LLMQueryPlanner, PlanningFailed
from models import KeyStatus, MetaKey, ValueType

DT = "court_decision"
CLOCK = FixedClock(date(2026, 10, 4))
KEYS = [
    MetaKey(
        DT,
        "issuing_body",
        ValueType.TEXT,
        "The court.",
        status=KeyStatus.APPROVED,
    ),
    MetaKey(
        DT,
        "decision_date",
        ValueType.DATE,
        "Date of the decision.",
        status=KeyStatus.APPROVED,
    ),
    MetaKey(
        DT,
        "document_kind",
        ValueType.TEXT,
        "Kind.",
        allowed_values=("judgment", "order"),
        status=KeyStatus.APPROVED,
    ),
]

GOOD = (
    '{"operation": "count", "filters": [{"key": "document_kind", "op": "eq", '
    '"value": "judgment"}]}'
)


class ScriptedLLM:
    """Returns the scripted replies in order and records every conversation."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    def run_tool_calling_turn(self, messages, *args, **kwargs):
        self.calls.append([dict(m) for m in messages])
        return SimpleNamespace(content=self._replies.pop(0))


def _planner(llm):
    return LLMQueryPlanner(llm, PlanCompiler(DateRangeResolver(CLOCK)), CLOCK)


def test_a_valid_reply_becomes_a_plan_in_one_call():
    llm = ScriptedLLM(GOOD)

    plan = _planner(llm).plan("how many judgments?", DT, KEYS)

    assert plan.operation is Operation.COUNT and plan.filters[0].key == "document_kind"
    assert len(llm.calls) == 1


def test_the_prompt_carries_today_the_keys_their_allowed_values_and_the_question():
    llm = ScriptedLLM(GOOD)

    _planner(llm).plan("hány ítélet volt?", DT, KEYS)

    prompt = llm.calls[0][0]["content"]
    assert "Today is 2026-10-04." in prompt
    assert "- document_kind (text): Kind. Allowed values: judgment, order." in prompt
    assert "hány ítélet volt?" in prompt


def test_a_reply_wrapped_in_a_code_fence_is_accepted():
    plan = _planner(ScriptedLLM(f"```json\n{GOOD}\n```")).plan("q", DT, KEYS)

    assert plan.operation is Operation.COUNT


def test_a_plan_the_catalog_rejects_is_retried_once_with_the_exact_error():
    bad = '{"operation": "count", "filters": [{"key": "no_such", "op": "eq", "value": "x"}]}'
    llm = ScriptedLLM(bad, GOOD)

    plan = _planner(llm).plan("q", DT, KEYS)

    assert plan.filters[0].key == "document_kind"
    retry = llm.calls[1]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert retry[1]["content"] == bad
    assert "unknown or unapproved key" in retry[2]["content"]


def test_a_reply_that_is_not_json_is_retried_too():
    llm = ScriptedLLM("I think the answer is 42", GOOD)

    assert _planner(llm).plan("q", DT, KEYS).operation is Operation.COUNT
    assert len(llm.calls) == 2


def test_two_bad_plans_fail_loudly_with_the_last_error_and_reply():
    bad = '{"operation": "average"}'
    llm = ScriptedLLM(bad, bad)

    with pytest.raises(PlanningFailed) as failure:
        _planner(llm).plan("q", DT, KEYS)

    assert "unknown operation" in failure.value.reason
    assert failure.value.reply == bad
    assert len(llm.calls) == 2  # exactly one retry, no more


def test_the_doc_type_comes_from_the_caller_not_from_the_model():
    plan = _planner(ScriptedLLM(GOOD)).plan("q", DT, KEYS)

    assert plan.doc_type == DT
