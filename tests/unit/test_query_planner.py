"""Tests for metadata.planner.LLMQueryPlanner (a scripted fake model, no network)."""

from datetime import date
from types import SimpleNamespace

import pytest

from metadata.clock import FixedClock
from metadata.compiler import PlanCompiler
from metadata.date_ranges import DateRangeResolver
from metadata.plan import Operation
from metadata.planner import (
    LLMQueryPlanner,
    PlanningFailed,
    TypeCatalog,
    collect_known_values,
    load_catalogs,
)
from models import DocumentType, KeyStatus, MetaKey, TypeStatus, ValueType

DT = "court_decision"
CLOCK = FixedClock(date(2026, 10, 4))
COURT_KEYS = [
    MetaKey(
        DT, "issuing_body", ValueType.TEXT, "The court.", status=KeyStatus.APPROVED
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
INVOICE_KEYS = [
    MetaKey(
        "invoice",
        "total_amount",
        ValueType.NUMBER,
        "Gross total.",
        status=KeyStatus.APPROVED,
    ),
    MetaKey(
        "invoice",
        "issuing_body",
        ValueType.TEXT,
        "The seller.",
        status=KeyStatus.APPROVED,
    ),
]
COURT = DocumentType(DT, "Court decision", "A ruling by a court.", TypeStatus.APPROVED)
INVOICE = DocumentType("invoice", "Invoice", "A bill for goods.", TypeStatus.APPROVED)
CATALOGS = [TypeCatalog(COURT, COURT_KEYS), TypeCatalog(INVOICE, INVOICE_KEYS)]

GOOD = (
    '{"document_type": "court_decision", "operation": "count", "filters": '
    '[{"key": "document_kind", "op": "eq", "value": "judgment"}]}'
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

    plan = _planner(llm).plan("how many judgments?", CATALOGS)

    assert plan.operation is Operation.COUNT and plan.filters[0].key == "document_kind"
    assert len(llm.calls) == 1


def test_the_model_chooses_the_document_type_and_the_plan_carries_it():
    reply = (
        '{"document_type": "invoice", "operation": "sum", "sum_key": "total_amount",'
        ' "filters": []}'
    )

    plan = _planner(ScriptedLLM(reply)).plan(
        "what is the total of the bills?", CATALOGS
    )

    assert plan.doc_type == "invoice" and plan.sum_key == "total_amount"


def test_the_prompt_lists_every_type_with_its_description_and_only_its_own_keys():
    llm = ScriptedLLM(GOOD)

    _planner(llm).plan("hány ítélet volt?", CATALOGS)

    prompt = llm.calls[0][0]["content"]
    assert "Today is 2026-10-04." in prompt and "hány ítélet volt?" in prompt
    assert '- "court_decision": Court decision. A ruling by a court.' in prompt
    assert '- "invoice": Invoice. A bill for goods.' in prompt
    assert "- document_kind (text): Kind. Allowed values: judgment, order." in prompt
    assert "- total_amount (number): Gross total." in prompt
    # the keys sit under their own type (court keys before the invoice block)
    assert prompt.index("document_kind") < prompt.index('- "invoice"')
    assert prompt.index('- "invoice"') < prompt.index("total_amount")


def test_a_type_without_keys_is_still_offered_it_can_be_counted_and_read():
    llm = ScriptedLLM(GOOD)
    bare = TypeCatalog(DocumentType("memo", "Memo", "A note.", TypeStatus.APPROVED), [])

    _planner(llm).plan("q", [*CATALOGS, bare])

    assert '- "memo": Memo. A note.' in llm.calls[0][0]["content"]
    assert (
        "no keys: it can only be counted, listed or read" in llm.calls[0][0]["content"]
    )


def test_a_reply_wrapped_in_a_code_fence_is_accepted():
    plan = _planner(ScriptedLLM(f"```json\n{GOOD}\n```")).plan("q", CATALOGS)

    assert plan.operation is Operation.COUNT


def test_a_type_the_model_invented_is_retried_once_with_the_exact_error():
    bad = '{"document_type": "spaceship", "operation": "count", "filters": []}'
    llm = ScriptedLLM(bad, GOOD)

    plan = _planner(llm).plan("q", CATALOGS)

    assert plan.doc_type == "court_decision"
    retry = llm.calls[1]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert (
        "unknown document_type 'spaceship'; use one of court_decision, invoice"
        in retry[2]["content"]
    )


def test_a_key_of_another_type_is_rejected_and_retried():
    wrong = (
        '{"document_type": "invoice", "operation": "count", "filters":'
        ' [{"key": "document_kind", "op": "eq", "value": "judgment"}]}'
    )
    llm = ScriptedLLM(wrong, GOOD)

    plan = _planner(llm).plan("q", CATALOGS)

    assert plan.doc_type == "court_decision"
    assert "unknown or unapproved key" in llm.calls[1][2]["content"]


def test_a_plan_without_a_document_type_is_retried():
    llm = ScriptedLLM('{"operation": "count"}', GOOD)

    _planner(llm).plan("q", CATALOGS)

    assert "needs a 'document_type'" in llm.calls[1][2]["content"]


def test_a_count_that_names_no_type_is_refused_not_guessed():
    none = '{"document_type": null, "operation": "count", "filters": []}'
    llm = ScriptedLLM(none, none)

    with pytest.raises(PlanningFailed, match="needs a document_type"):
        _planner(llm).plan("how many spaceships?", CATALOGS)


def test_a_lookup_may_name_no_type():
    reply = '{"document_type": null, "operation": "lookup", "filters": []}'

    plan = _planner(ScriptedLLM(reply)).plan(
        "why did the court dismiss the claim?", CATALOGS
    )

    assert plan.doc_type is None and plan.operation is Operation.LOOKUP


def test_a_reply_that_is_not_json_is_retried_too():
    llm = ScriptedLLM("I think the answer is 42", GOOD)

    assert _planner(llm).plan("q", CATALOGS).operation is Operation.COUNT
    assert len(llm.calls) == 2


def test_two_bad_plans_fail_loudly_with_the_last_error_and_reply():
    bad = '{"document_type": "court_decision", "operation": "average"}'
    llm = ScriptedLLM(bad, bad)

    with pytest.raises(PlanningFailed) as failure:
        _planner(llm).plan("q", CATALOGS)

    assert "unknown operation" in failure.value.reason
    assert failure.value.reply == bad
    assert len(llm.calls) == 2  # exactly one retry, no more


def test_stored_values_are_shown_under_the_type_they_belong_to():
    llm = ScriptedLLM(GOOD)
    known = {
        DT: {"issuing_body": ["Kúria", "Budapesti XVIII. és XIX. Kerületi Bíróság"]},
        "invoice": {"issuing_body": ["Példa Kft."]},
    }

    _planner(llm).plan("q", CATALOGS, known)

    prompt = llm.calls[0][0]["content"]
    assert "a combined name is ONE value): Kúria; Budapesti XVIII. és XIX." in prompt
    assert "a combined name is ONE value): Példa Kft." in prompt
    # same key name in two types: each type lists only its own values
    assert (
        prompt.index("Kúria") < prompt.index('- "invoice"') < prompt.index("Példa Kft.")
    )


class _Source:
    """A store double for the catalog and the stored values."""

    def __init__(self, types, keys, values=None):
        self._types, self._keys, self._values = types, keys, values or {}
        self.value_calls: list[tuple] = []

    def list_types(self, status=None):
        return [t for t in self._types if status is None or t.status is status]

    def list_keys(self, doc_type, status=None):
        return [
            k
            for k in self._keys
            if k.doc_type == doc_type and (status is None or k.status is status)
        ]

    def distinct_text_values(self, key, limit, doc_type=None):
        self.value_calls.append((key, doc_type))
        return self._values.get((doc_type, key))  # None = too many


def test_only_low_cardinality_free_text_keys_get_their_values_listed_per_type():
    source = _Source([COURT, INVOICE], [], {(DT, "issuing_body"): ["Kúria"]})

    known = collect_known_values(source, CATALOGS)

    assert known == {DT: {"issuing_body": ["Kúria"]}}
    # decision_date is a date and document_kind has allowed values: neither is asked
    # for; the invoice's issuing_body is asked for under the *invoice* type
    assert ("issuing_body", "invoice") in source.value_calls
    assert ("document_kind", DT) not in source.value_calls


def test_load_catalogs_reads_only_approved_types_with_their_approved_keys():
    draft = DocumentType("draft", "Draft", "d", TypeStatus.PROPOSED)
    retired_key = MetaKey(DT, "old", ValueType.TEXT, "x", status=KeyStatus.RETIRED)
    source = _Source([COURT, INVOICE, draft], [*COURT_KEYS, *INVOICE_KEYS, retired_key])

    catalogs = load_catalogs(source)

    assert [c.doc_type.type for c in catalogs] == [
        DT,
        "invoice",
    ]  # not the proposed one
    assert "old" not in {k.key for k in catalogs[0].keys}


def test_load_catalogs_fails_loudly_when_no_type_is_approved():
    with pytest.raises(RuntimeError, match="no approved document types"):
        load_catalogs(
            _Source([DocumentType("draft", "Draft", "d", TypeStatus.PROPOSED)], [])
        )


FUSED_FILTERS = (
    '{"document_type": "court_decision", "operation": "count", "filters": ['
    '{"key": "issuing_body", "op": "eq", "value": "X", '
    '"key": "document_kind", "op": "eq", "value": "judgment"}]}'
)


def test_two_filters_fused_into_one_object_are_rejected_not_silently_reduced_to_one():
    """A model once merged two filters; the parser kept only the second, the court
    filter vanished and a wrong count was presented as exact."""
    llm = ScriptedLLM(FUSED_FILTERS, GOOD)

    plan = _planner(llm).plan("q", CATALOGS)

    assert plan.filters[0].key == "document_kind"  # the corrected reply
    assert "the key 'key' twice" in llm.calls[1][2]["content"]


def test_with_exactly_one_type_the_prompt_tells_the_model_to_use_it_by_name():
    """A generic "how many documents" must not become a null-type lookup when the
    only possible kind of document is known."""
    llm = ScriptedLLM(GOOD)

    _planner(llm).plan("hány dokumentum kelt jövő hónapban?", [CATALOGS[0]])

    prompt = llm.calls[0][0]["content"]
    assert 'Only ONE document type exists ("court_decision")' in prompt
    assert 'even when it just says "documents"' in prompt


def test_with_several_types_a_generic_question_is_not_forced_onto_one_of_them():
    llm = ScriptedLLM(GOOD)

    _planner(llm).plan("how many documents?", CATALOGS)

    prompt = llm.calls[0][0]["content"]
    assert "Only ONE document type exists" not in prompt
    assert "no single listed kind clearly fits" in prompt
