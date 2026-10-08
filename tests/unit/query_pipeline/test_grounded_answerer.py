"""Writing an answer from the chunks read: the prompt is the original's, the policy is explicit."""

import pytest

import drivers.llm as llm_module
from drivers.llm import REFUSAL_SENTENCE, AnswerDriver, _build_prompt
from models import ChunkMetadata, RetrievedChunk
from query.answering import AnswerPolicy, GroundedAnswerer


def chunk(text, source="a.pdf", page=1, date="2024-01-02"):
    return RetrievedChunk(
        id=1,
        content=text,
        metadata=ChunkMetadata(
            source_file=source, page_number=page, chunk_index=0, document_date=date
        ),
        score=0.5,
    )


class FakeDriver(AnswerDriver):
    def __init__(self, reply="An answer."):
        super().__init__("fake")
        self.reply, self.calls = reply, []

    def generate(self, system_prompt, user_message, max_tokens=1024):
        self.calls.append((system_prompt, user_message, max_tokens))
        return self.reply

    def run_tool_calling_turn(self, messages, tools=None):
        raise NotImplementedError


CHUNKS = (chunk("first"), chunk("second", source="b.pdf", page=3))
POLICIES = [AnswerPolicy(p, d) for p in (False, True) for d in (False, True)]


@pytest.mark.parametrize("policy", POLICIES)
def test_the_prompt_is_the_original_prompt_for_the_same_policy(
    policy, monkeypatch, settings_override
):
    monkeypatch.setattr(
        llm_module,
        "settings",
        settings_override(
            ANSWER_PARTIAL_COVERAGE=policy.partial_coverage,
            EXPOSE_DOCUMENT_DATE=policy.expose_document_date,
        ),
    )
    driver = FakeDriver()

    GroundedAnswerer(driver, policy, max_tokens=77).answer("Q?", CHUNKS)

    expected = _build_prompt("Q?", list(CHUNKS))
    assert driver.calls == [(*expected, 77)]


def test_the_policy_given_wins_over_the_settings(monkeypatch, settings_override):
    monkeypatch.setattr(
        llm_module,
        "settings",
        settings_override(ANSWER_PARTIAL_COVERAGE=False, EXPOSE_DOCUMENT_DATE=False),
    )
    driver = FakeDriver()

    GroundedAnswerer(driver, AnswerPolicy(True, True)).answer("Q?", CHUNKS)

    system, user, _ = driver.calls[0]
    assert "A partial answer is always better" in system
    assert "small sample" in system  # the sample note follows the policy too
    assert "date: 2024-01-02" in user


def test_the_default_answer_method_builds_the_same_prompt_and_calls_generate():
    driver = FakeDriver()

    driver.answer("Q?", list(CHUNKS), max_tokens=5)

    assert driver.calls == [(*_build_prompt("Q?", list(CHUNKS)), 5)]


@pytest.mark.parametrize(
    "reply",
    [REFUSAL_SENTENCE, f"  '{REFUSAL_SENTENCE}'\n", REFUSAL_SENTENCE.upper()],
)
def test_the_models_own_refusal_is_recognised(reply):
    answer = GroundedAnswerer(FakeDriver(reply), POLICIES[0]).answer("Q?", CHUNKS)

    assert answer.refused and answer.text == reply


@pytest.mark.parametrize(
    "reply",
    [
        "The court decided X.",
        f"X was decided. {REFUSAL_SENTENCE}",  # an answer that mentions it is an answer
        "",
    ],
)
def test_anything_else_is_an_answer(reply):
    answer = GroundedAnswerer(FakeDriver(reply), POLICIES[0]).answer("Q?", CHUNKS)

    assert not answer.refused and answer.text == reply


def test_both_refusal_clauses_still_contain_the_sentence_that_is_recognised():
    for partial in (False, True):
        system, _ = _build_prompt("q", [], partial_coverage=partial)
        assert f"'{REFUSAL_SENTENCE}'" in system
