"""Writing an answer from the chunks read: the policy is explicit, the model's refusal is recognised."""

import pytest

import drivers.llm as llm_module
from drivers.llm import REFUSAL_SENTENCE, AnswerDriver, _build_prompt
from models import ChunkMetadata, Citation, RetrievedChunk
from query.answering import AnswerPolicy, GroundedAnswerer, cited_labels, render_sources


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
POLICIES = [AnswerPolicy(expose_document_date=d) for d in (False, True)]


@pytest.mark.parametrize("policy", POLICIES)
def test_the_prompt_is_the_drivers_prompt_for_the_same_policy(policy):
    driver = FakeDriver()

    GroundedAnswerer(driver, policy, max_tokens=77).answer("Q?", CHUNKS)

    expected = _build_prompt(
        "Q?", list(CHUNKS), expose_document_date=policy.expose_document_date
    )
    assert driver.calls == [(*expected, 77)]


def test_the_policy_given_wins_over_the_settings(monkeypatch, settings_override):
    monkeypatch.setattr(
        llm_module, "settings", settings_override(EXPOSE_DOCUMENT_DATE=False)
    )
    driver = FakeDriver()

    GroundedAnswerer(driver, AnswerPolicy(expose_document_date=True)).answer(
        "Q?", CHUNKS
    )

    assert "date: 2024-01-02" in driver.calls[0][1]


def test_without_the_policy_the_dates_are_not_shown():
    driver = FakeDriver()

    GroundedAnswerer(driver, AnswerPolicy(expose_document_date=False)).answer(
        "Q?", CHUNKS
    )

    assert "date:" not in driver.calls[0][1]


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


def test_the_refusal_clause_contains_the_sentence_that_is_recognised():
    system, _ = _build_prompt("q", [])

    assert f"'{REFUSAL_SENTENCE}'" in system


class RecordingObserver:
    def __init__(self):
        self.calls = []

    def on_answer(self, question, chunks, answer, seconds):
        self.calls.append((question, chunks, answer, seconds))


class SlowDriver(FakeDriver):
    """A model that takes ``seconds`` on a clock the test controls."""

    def __init__(self, clock_state, seconds, reply="An answer."):
        super().__init__(reply)
        self.clock_state, self.seconds = clock_state, seconds

    def generate(self, system_prompt, user_message, max_tokens=1024):
        self.clock_state["now"] += self.seconds
        return super().generate(system_prompt, user_message, max_tokens)


class TestObservers:
    def test_every_observer_is_told_in_order_with_the_time_the_model_took(self):
        state = {"now": 100.0}
        order = []
        first, second, recording = (
            RecordingObserver(),
            RecordingObserver(),
            RecordingObserver(),
        )
        first.on_answer = lambda *a: order.append("first")  # type: ignore[method-assign]
        second.on_answer = lambda *a: order.append("second")  # type: ignore[method-assign]

        answerer = GroundedAnswerer(
            SlowDriver(state, 2.5),
            POLICIES[0],
            observers=(first, second, recording),
            clock=lambda: state["now"],
        )
        answer = answerer.answer("Q?", CHUNKS)

        assert order == ["first", "second"]
        assert recording.calls == [("Q?", CHUNKS, answer, 2.5)]

    def test_the_observer_sees_whether_the_model_refused(self):
        state = {"now": 0.0}
        recording = RecordingObserver()

        GroundedAnswerer(
            SlowDriver(state, 0.0, REFUSAL_SENTENCE),
            POLICIES[0],
            observers=(recording,),
            clock=lambda: state["now"],
        ).answer("Q?", CHUNKS)

        assert recording.calls[0][2].refused is True

    def test_without_observers_nothing_is_needed(self):
        assert GroundedAnswerer(FakeDriver(), POLICIES[0]).answer("Q?", CHUNKS).text


# ------------------------------------------------------------------ citations


def numbered(*ids_and_files):
    """Chunks with distinct ids, files and chunk indexes, best first."""
    return tuple(
        RetrievedChunk(
            id=i,
            content=f"text {i}",
            metadata=ChunkMetadata(source_file=name, page_number=None, chunk_index=idx),
            score=0.5,
        )
        for i, name, idx in ids_and_files
    )


# best first: id 10 is the best chunk; the prompt labels worst-to-best, so label 1 is id 30
RANKED = numbered((10, "a.docx", 4), (20, "b.docx", 7), (30, "c.docx", 0))


def answer_with(text, chunks=RANKED):
    return GroundedAnswerer(
        FakeDriver(text), AnswerPolicy(expose_document_date=False)
    ).answer("Q?", chunks)


def test_a_label_in_the_answer_is_resolved_to_the_chunk_shown_under_it():
    answer = answer_with("The court dismissed the claim [3].")

    # label 3 is the last one shown = the best chunk (id 10), not the third by rank
    assert answer.citations == (Citation(3, 10, "a.docx", 4),)
    assert answer.unknown_labels == ()


def test_several_labels_are_resolved_each_once_in_the_order_first_cited():
    answer = answer_with("First [2]. Then [1, 3], and again [2][3].")

    assert [c.label for c in answer.citations] == [2, 1, 3]
    assert [c.chunk_id for c in answer.citations] == [20, 30, 10]


def test_a_label_no_excerpt_carried_is_a_wrong_citation_not_a_resolved_one():
    answer = answer_with("It says so [2] and so [7] and [0].")

    assert [c.label for c in answer.citations] == [2]
    assert answer.unknown_labels == (7, 0)


def test_the_models_refusal_cites_nothing():
    answer = answer_with(REFUSAL_SENTENCE + " [1]")

    assert answer.refused and answer.citations == () and answer.unknown_labels == ()


def test_an_answer_without_labels_has_no_citations():
    answer = answer_with("The court dismissed the claim (a.docx, page 2).")

    assert answer.citations == () and answer.unknown_labels == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("see [1]", [1]),
        ("see [1, 2] and [3;4]", [1, 2, 3, 4]),
        ("see [ 12 ]", [12]),
        ("decided in [2023] by the court", []),  # a year is not a label
        ("see [a] and [1a] and []", []),
        ("see [1][2][1]", [1, 2]),
        ("no brackets 1 2 3", []),
    ],
)
def test_which_bracketed_numbers_count_as_labels(text, expected):
    assert cited_labels(text) == expected


def test_the_sources_block_names_the_cited_chunks_and_is_empty_without_citations():
    assert render_sources(()) == ""
    block = render_sources((Citation(2, 20, "b.docx", 7), Citation(1, 30, "c.docx", 0)))

    assert block == "Sources:\n[2] b.docx, section 7\n[1] c.docx, section 0"
