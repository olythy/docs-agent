"""The runtime claim judge (a scripted model): one call for all pairs, and no verdict unless it is clean."""

import json
from types import SimpleNamespace

import pytest

from query.claim_judge import LLMClaimJudge


class Llm:
    def __init__(self, reply):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.calls = []

    def run_tool_calling_turn(self, messages, tools=None, temperature=None):
        self.calls.append((messages[0]["content"], temperature))
        return SimpleNamespace(content=self.reply)


def judge(reply):
    llm = Llm(reply)
    return LLMClaimJudge(llm), llm  # type: ignore[arg-type]


PAIRS = [
    ("The claim is made.", "a passage"),
    ("Another claim is made.", "another passage"),
]


def test_all_pairs_go_in_one_call_at_temperature_zero():
    j, llm = judge({"supported": [True, False]})

    verdicts = j.supports(PAIRS)

    assert verdicts == [True, False]
    assert len(llm.calls) == 1 and llm.calls[0][1] == 0.0
    assert (
        "The claim is made." in llm.calls[0][0] and "another passage" in llm.calls[0][0]
    )


def test_no_pairs_means_no_call():
    j, llm = judge({"supported": []})

    assert j.supports([]) == [] and llm.calls == []


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        "[true, false]",
        '{"supported": [true]}',
        '{"supported": [true, "yes"]}',
        '{"other": 1}',
        "",
    ],
)
def test_a_reply_that_is_not_one_clean_verdict_per_pair_gives_none(reply):
    j, _ = judge(reply)

    assert j.supports(PAIRS) is None
