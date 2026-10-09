"""The survey resolver (no database, a scripted model): what it builds and what it refuses.

The model's answer is only a proposal; every case below is about the code that checks it.
A wrong answer must end in ``None`` (the plain "not supported yet"), never in a count.
"""

import json
from types import SimpleNamespace

import pytest

from metadata.plan import Filter, FilterOp, Operation, QueryPlan
from metadata.planner import TypeCatalog
from metadata.survey_resolver import (
    CATCH_ALL_VALUE,
    EXTRACTED_BY_MODEL,
    LLMSurveyResolver,
    usable_keys,
)
from models import DocumentType, KeyStatus, MetaKey, TypeStatus, ValueType

DT = "court_decision"


def _key(name, allowed=None, vtype=ValueType.TEXT, multi=False):
    return MetaKey(
        DT,
        name,
        vtype,
        f"the {name}",
        allowed_values=allowed,
        multi_valued=multi,
        status=KeyStatus.APPROVED,
    )


SUBJECT = _key("subject", ("sale", "lease", "sale_and_lease", CATCH_ALL_VALUE))
OUTCOME = _key("outcome", ("granted", "dismissed", CATCH_ALL_VALUE))
CATALOG = TypeCatalog(
    DocumentType(DT, "Court decision", "A ruling.", TypeStatus.APPROVED),
    [
        _key("court", None),  # free text: not a closed vocabulary
        _key("fee", None, ValueType.NUMBER),
        _key("tags", ("a", "b"), multi=True),  # several values per document
        SUBJECT,
        OUTCOME,
    ],
)
COURT = Filter("court", FilterOp.EQ, "Eger")
SURVEY = QueryPlan(DT, Operation.SURVEY, (COURT,))


class Llm:
    """Answers with a scripted reply and remembers what it was asked."""

    def __init__(self, reply):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.calls = []

    def run_tool_calling_turn(self, messages, tools=None, temperature=None):
        self.calls.append((messages[0]["content"], temperature))
        return SimpleNamespace(content=self.reply)


def verdicts(*named):
    """A verdict for every stored value of the subject key: true for ``named``."""
    return {v: v in named for v in SUBJECT.allowed_values or ()}


def resolver(llm):
    """The resolver over a scripted fake (which is not an AnswerDriver)."""
    return LLMSurveyResolver(llm)  # type: ignore[arg-type]


def good(**kw):
    return {
        "topic_key": "subject",
        "topic_values": verdicts("sale", "sale_and_lease"),
        "group_by": "outcome",
        "unanswered": None,
    } | kw


def resolve(reply, plan=SURVEY):
    llm = Llm(reply)
    return resolver(llm).resolve("a question", plan, [CATALOG]), llm


def test_a_checked_answer_becomes_a_count_by_the_outcome_over_the_topic_values():
    plan, _ = resolve(good())

    assert plan is not None
    assert plan.operation is Operation.COUNT
    assert plan.group_by == "outcome"
    assert plan.filters == (
        COURT,  # the planner's own restriction is kept
        Filter("subject", FilterOp.IN, ["sale", "sale_and_lease"]),
    )
    assert plan.residual is None and plan.doc_type == DT


def test_the_figures_always_carry_the_note_that_a_model_extracted_the_values():
    plan, _ = resolve(good())

    assert plan is not None and plan.caveat == EXTRACTED_BY_MODEL


def test_what_the_count_does_not_answer_is_added_and_kept_short():
    plan, _ = resolve(good(unanswered="the reasons " + "x" * 1000))

    assert plan is not None and plan.caveat is not None
    assert plan.caveat.startswith(
        EXTRACTED_BY_MODEL + "; not answered by the figures: the reasons"
    )
    assert len(plan.caveat) < len(EXTRACTED_BY_MODEL) + 400


def test_only_the_closed_vocabulary_single_valued_text_keys_are_offered():
    assert set(usable_keys(CATALOG)) == {"subject", "outcome"}


def test_the_model_is_asked_at_temperature_zero_with_the_keys_and_their_values():
    _, llm = resolve(good())

    prompt, temperature = llm.calls[0]
    assert temperature == 0.0
    assert "sale_and_lease" in prompt and "granted" in prompt
    assert "EVERY stored value" in prompt  # the verdict per value is asked for
    assert "- court:" not in prompt and "- tags:" not in prompt  # not usable keys


def test_a_count_with_no_breakdown_is_allowed():
    plan, _ = resolve(good(group_by=None))

    assert plan is not None and plan.group_by is None


@pytest.mark.parametrize(
    "reply",
    [
        good(topic_key=None),  # the model says no key covers it
        good(topic_key="court"),  # a free-text key is not a topic list
        good(topic_key="nope"),
        good(topic_key="tags"),  # several values per document
        good(topic_values={}),
        good(topic_values=verdicts()),  # every value false: nothing names the topic
        good(topic_values=["sale"]),  # not a verdict per value
        good(
            topic_values={"sale": True, "mortgage": True}
        ),  # a value the key does not define
        good(topic_values={"sale": "yes"}),  # not a boolean
        good(
            topic_values=verdicts("sale", CATCH_ALL_VALUE)
        ),  # "other" is never a topic
        good(topic_values=verdicts(CATCH_ALL_VALUE)),
        good(group_by="subject"),  # the same key twice
        good(group_by="nope"),
        good(group_by="court"),
        good(group_by=3),
    ],
)
def test_an_answer_that_fails_a_check_gives_no_plan(reply):
    plan, _ = resolve(reply)

    assert plan is None


@pytest.mark.parametrize("text", ["not json at all", "[1, 2]", '{"topic_key": 1', ""])
def test_a_reply_that_is_not_a_json_object_gives_no_plan(text):
    plan, _ = resolve(text)

    assert plan is None


def test_the_topic_filter_replaces_the_planners_own_filter_on_that_key_and_keeps_the_rest():
    planner_plan = QueryPlan(
        DT,
        Operation.SURVEY,
        (
            COURT,
            Filter("subject", FilterOp.EQ, "lease"),
            Filter("outcome", FilterOp.EQ, "granted"),
        ),
    )

    plan, _ = resolve(good(), planner_plan)

    assert plan is not None
    assert plan.filters == (
        COURT,
        Filter(
            "outcome", FilterOp.EQ, "granted"
        ),  # another restriction of the question stays
        Filter("subject", FilterOp.IN, ["sale", "sale_and_lease"]),  # not "lease"
    )


def test_a_survey_with_a_residual_is_never_made_exact_and_the_model_is_not_asked():
    llm = Llm(good())

    plan = resolver(llm).resolve(
        "q",
        QueryPlan(DT, Operation.SURVEY, (COURT,), residual="the reasons"),
        [CATALOG],
    )

    assert plan is None and llm.calls == []


def test_without_a_type_or_with_no_usable_key_the_model_is_not_asked():
    llm = Llm(good())
    no_keys = TypeCatalog(CATALOG.doc_type, [_key("court", None)])

    assert (
        resolver(llm).resolve("q", QueryPlan(None, Operation.SURVEY), [CATALOG]) is None
    )
    assert resolver(llm).resolve("q", SURVEY, [no_keys]) is None
    assert (
        resolver(llm).resolve("q", QueryPlan("invoice", Operation.SURVEY), [CATALOG])
        is None
    )
    assert llm.calls == []


def test_a_verdict_list_may_name_only_the_values_it_judged_true_or_false():
    plan, _ = resolve(good(topic_values={"sale": True}))

    assert plan is not None
    assert plan.filters[-1] == Filter(
        "subject", FilterOp.IN, ["sale"]
    )  # a partial list is fine
