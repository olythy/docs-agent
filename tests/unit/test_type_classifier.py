"""Tests for metadata.classifier.LLMTypeClassifier (a scripted fake model, no network)."""

import json
from types import SimpleNamespace

import pytest

from metadata.classifier import Classification, LLMTypeClassifier, TypeProposal
from models import DocumentType, TypeStatus

TYPES = [
    DocumentType(
        "court_decision", "Court decision", "A ruling by a court.", TypeStatus.APPROVED
    ),
    DocumentType("invoice", "Invoice", "A bill for goods.", TypeStatus.PROPOSED),
]


class ScriptedLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def run_tool_calling_turn(self, messages):
        self.prompts.append(messages[0]["content"])
        return SimpleNamespace(content=self.reply)


def _classify(reply, text="TEXT BODY"):
    llm = ScriptedLLM(reply if isinstance(reply, str) else json.dumps(reply))
    return LLMTypeClassifier(llm).classify(text, TYPES), llm


def test_it_picks_an_existing_type_and_keeps_the_quote():
    result, _ = _classify(
        {"type": "court_decision", "proposal": None, "evidence": "ÍTÉLETET"}
    )

    assert result == Classification(type="court_decision", evidence="ÍTÉLETET")


def test_it_can_propose_a_new_type():
    result, _ = _classify(
        {
            "type": None,
            "proposal": {
                "type": "contract",
                "name": "Contract",
                "description": "An agreement.",
            },
            "evidence": "SZERZŐDÉS",
        }
    )

    assert result == Classification(
        proposal=TypeProposal("contract", "Contract", "An agreement."),
        evidence="SZERZŐDÉS",
    )


def test_the_prompt_lists_every_type_with_its_status_and_description_and_the_text():
    _, llm = _classify(
        {"type": "invoice", "proposal": None, "evidence": "x"}, text="THE OPENING"
    )

    prompt = llm.prompts[0]
    assert "- court_decision (approved): Court decision. A ruling by a court." in prompt
    assert "- invoice (proposed): Invoice. A bill for goods." in prompt
    assert "THE OPENING" in prompt and "VERBATIM quote" in prompt


def test_the_prompt_says_none_yet_when_no_type_is_known():
    llm = ScriptedLLM(json.dumps({"type": None, "proposal": None, "evidence": ""}))

    LLMTypeClassifier(llm).classify("t", [])

    assert "(none yet)" in llm.prompts[0]


@pytest.mark.parametrize(
    "reply",
    [
        "I think it is an invoice",  # not JSON
        {"type": None, "proposal": None, "evidence": "x"},  # decided nothing
        {  # decided both
            "type": "invoice",
            "proposal": {"type": "a", "name": "b", "description": "c"},
            "evidence": "x",
        },
        {"type": 7, "proposal": None, "evidence": "x"},  # a type that is not text
        {
            "type": None,
            "proposal": {"type": "a", "name": "", "description": "c"},
            "evidence": "x",
        },
        {"type": None, "proposal": "contract", "evidence": "x"},
        {"type": None, "proposal": {"type": "a", "name": "b"}, "evidence": "x"},
    ],
)
def test_an_unusable_answer_is_a_failure_not_a_guess(reply):
    result, _ = _classify(reply)

    assert result == Classification(failed=True)


def test_evidence_that_is_not_text_is_dropped_so_the_runner_will_reject_it():
    result, _ = _classify({"type": "invoice", "proposal": None, "evidence": ["x"]})

    assert result.evidence is None
