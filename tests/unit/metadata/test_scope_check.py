"""The plan checker (no database, a scripted model): when an exact plan is turned into a read.

The model's answer is only a proposal; what matters here is the code around it. A plan is
kept exact only when the call says no filter is too wide, and it is read whenever the call
names a filter that is too wide or gives no usable answer.
"""

import json
from types import SimpleNamespace

import pytest

from metadata.plan import Filter, FilterOp, Operation, QueryPlan
from metadata.planner import TypeCatalog
from metadata.scope_check import UNCHECKED, LLMPlanChecker
from models import DocumentType, KeyStatus, MetaKey, TypeStatus, ValueType

DT = "court_decision"


def _key(name, allowed=None, vtype=ValueType.TEXT):
    return MetaKey(
        DT,
        name,
        vtype,
        f"the {name}",
        allowed_values=allowed,
        status=KeyStatus.APPROVED,
    )


CATALOG = TypeCatalog(
    DocumentType(DT, "Court decision", "A ruling.", TypeStatus.APPROVED),
    [
        _key("court"),  # free text
        _key("subject", ("company_law", "inheritance", "other")),
        _key("kind", ("judgment", "order")),
        _key("decided", None, ValueType.DATE),
    ],
)
COURT = Filter("court", FilterOp.EQ, "Eger")
SUBJECT = Filter("subject", FilterOp.IN, ["company_law"])


class Llm:
    def __init__(self, reply):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.calls = []

    def run_tool_calling_turn(self, messages, tools=None, temperature=None):
        self.calls.append((messages[0]["content"], temperature))
        return SimpleNamespace(content=self.reply)


def check(reply, plan, catalogs=(CATALOG,)):
    llm = Llm(reply)
    return LLMPlanChecker(llm).check("a question", plan, list(catalogs)), llm  # type: ignore[arg-type]


def plan(*filters, operation=Operation.COUNT, **kw):
    return QueryPlan(DT, operation, tuple(filters), **kw)


NONE_WIDER = {"wider": []}


def test_a_plan_whose_filters_fit_is_kept_as_it_is():
    original = plan(COURT, SUBJECT)

    kept, _ = check(NONE_WIDER, original)

    assert kept is original


def test_a_filter_named_as_too_wide_turns_the_plan_into_a_read_with_the_narrower_topic():
    original = plan(COURT, SUBJECT)

    result, _ = check(
        {"wider": [{"key": "subject", "topic": "company assembly resolutions"}]},
        original,
    )

    assert result.residual == "company assembly resolutions"
    assert result.filters == original.filters  # the filters still narrow the documents


@pytest.mark.parametrize(
    "operation", [Operation.COUNT, Operation.LIST, Operation.SUM, Operation.OVERVIEW]
)
def test_every_exact_operation_is_checked(operation):
    result, llm = check(
        {"wider": [{"key": "subject", "topic": "x"}]},
        plan(SUBJECT, operation=operation),
    )

    assert result.residual == "x" and len(llm.calls) == 1


@pytest.mark.parametrize(
    "operation", [Operation.LOOKUP, Operation.SURVEY, Operation.UNSUPPORTED]
)
def test_a_plan_that_is_not_exact_is_not_checked(operation):
    original = plan(SUBJECT, operation=operation)

    result, llm = check({"wider": [{"key": "subject", "topic": "x"}]}, original)

    assert result is original and llm.calls == []


def test_a_plan_that_already_has_a_residual_is_not_checked_again():
    original = plan(SUBJECT, residual="something no key covers")

    result, llm = check({"wider": [{"key": "subject", "topic": "x"}]}, original)

    assert result is original and llm.calls == []


@pytest.mark.parametrize(
    "filters",
    [
        (COURT,),  # free text: exact by construction
        (Filter("decided", FilterOp.BETWEEN, {"kind": "calendar", "year": 2022}),),
        (
            Filter("subject", FilterOp.NE, "other"),
        ),  # "everything but" has no breadth to check
        (),
    ],
)
def test_a_plan_with_no_closed_value_topic_filter_costs_no_call(filters):
    original = plan(*filters)

    result, llm = check({"wider": [{"key": "subject", "topic": "x"}]}, original)

    assert result is original and llm.calls == []


def test_an_answer_naming_a_key_the_plan_does_not_filter_on_is_ignored():
    original = plan(SUBJECT)

    result, _ = check(
        {"wider": [{"key": "kind", "topic": "x"}, {"key": "nope", "topic": "y"}]},
        original,
    )

    assert result is original


@pytest.mark.parametrize(
    "entry",
    [
        {"key": "subject", "topic": ""},
        {"key": "subject", "topic": "  "},
        {"key": "subject"},
        {"key": "subject", "topic": 3},
        "subject",
        None,
    ],
)
def test_an_entry_without_a_usable_topic_is_ignored(entry):
    original = plan(SUBJECT)

    result, _ = check({"wider": [entry]}, original)

    assert result is original


def test_several_wider_keys_are_all_named_and_a_long_topic_is_cut():
    original = plan(Filter("kind", FilterOp.EQ, "order"), SUBJECT)

    result, _ = check(
        {
            "wider": [
                {"key": "subject", "topic": "a" * 500},
                {"key": "kind", "topic": "orders about costs"},
            ]
        },
        original,
    )

    assert result.residual is not None
    first, second = result.residual.split("; ")
    assert first == "a" * 200 and second == "orders about costs"


@pytest.mark.parametrize("text", ["not json", "[1]", '{"wider": ', "", '{"other": 1}'])
def test_a_reply_that_gives_no_verdict_reads_the_plan_when_it_is_not_json_and_keeps_it_when_it_is_empty(
    text,
):
    original = plan(SUBJECT)

    result, _ = check(text, original)

    if text == '{"other": 1}':
        assert (
            result is original
        )  # valid JSON that names nothing: nothing was found too wide
    else:
        assert (
            result.residual == UNCHECKED
        )  # no usable answer: a wrong exact figure is worse


def test_the_prompt_carries_the_question_the_descriptions_and_the_selected_values_and_runs_at_zero():
    _, llm = check(NONE_WIDER, plan(COURT, SUBJECT))

    prompt, temperature = llm.calls[0]
    assert temperature == 0.0
    assert (
        "the subject" in prompt and "company_law" in prompt and "a question" in prompt
    )
    assert "- court:" not in prompt  # the free-text key is not offered


def test_a_plan_over_an_unknown_type_is_kept():
    original = QueryPlan("invoice", Operation.COUNT, (SUBJECT,))

    result, llm = check(NONE_WIDER, original)

    assert result is original and llm.calls == []
