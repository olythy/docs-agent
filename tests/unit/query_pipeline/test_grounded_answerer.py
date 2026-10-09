"""Writing an answer from the chunks read: the policy is explicit, the model's refusal is recognised."""

import pytest

import drivers.llm as llm_module
from drivers.llm import REFUSAL_SENTENCE, AnswerDriver, _build_prompt
from models import ChunkMetadata, Citation, RetrievedChunk
from query.answering import (
    AnswerPolicy,
    GroundedAnswerer,
    cited_labels,
    render_sources,
    split_quotes,
    strip_labels,
)


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
            content=f"The court said text {i} about the claim.",
            metadata=ChunkMetadata(source_file=name, page_number=None, chunk_index=idx),
            score=0.5,
        )
        for i, name, idx in ids_and_files
    )


# best first: id 10 is the best chunk; the prompt labels worst-to-best, so label 1 is id 30
RANKED = numbered((10, "a.docx", 4), (20, "b.docx", 7), (30, "c.docx", 0))
#: What each label's chunk really says: label 3 = id 10, label 2 = id 20, label 1 = id 30.
QUOTE = {1: "said text 30 about", 2: "said text 20 about", 3: "said text 10 about"}


def reply(text, *labels):
    """A model reply: the text, then a QUOTES block with a true quote for each label."""
    quotes = "".join(f'[{n}] "{QUOTE[n]}"\n' for n in labels)
    return f"{text}\n\nQUOTES:\n{quotes}" if labels else text


def answer_with(text, chunks=RANKED):
    return GroundedAnswerer(
        FakeDriver(text), AnswerPolicy(expose_document_date=False)
    ).answer("Q?", chunks)


def test_a_label_with_a_quote_that_is_in_its_chunk_is_a_citation():
    answer = answer_with(reply("The court dismissed the claim [3].", 3))

    # label 3 is the last one shown = the best chunk (id 10), not the third by rank
    assert answer.citations == (Citation(3, 10, "a.docx", 4, "said text 10 about"),)
    assert answer.unknown_labels == () and answer.unverified_labels == ()


def test_the_quotes_block_is_not_part_of_the_visible_answer_but_the_reply_is_kept():
    raw = reply("The court dismissed the claim [3].", 3)

    answer = answer_with(raw)

    assert answer.visible == "The court dismissed the claim [3]."
    assert answer.text == raw  # for the audit log, unchanged


def test_several_labels_are_resolved_each_once_in_the_order_first_cited():
    answer = answer_with(reply("First [2]. Then [1, 3], and again [2][3].", 2, 1, 3))

    assert [c.label for c in answer.citations] == [2, 1, 3]
    assert [c.chunk_id for c in answer.citations] == [20, 30, 10]


def test_a_label_without_a_quote_is_not_a_citation_and_leaves_the_visible_answer():
    answer = answer_with(reply("Claim one [3]. Claim two [2].", 3))  # no quote for 2

    assert [c.label for c in answer.citations] == [3]
    assert answer.unverified_labels == (2,)
    assert answer.visible == "Claim one [3]. Claim two."


def test_a_quote_that_is_not_in_the_chunk_under_that_label_is_not_a_citation():
    # label 2's quote is true, but of label 3's chunk: the label names another chunk
    text = 'The court ruled [2].\n\nQUOTES:\n[2] "said text 10 about"\n'

    answer = answer_with(text)

    assert answer.citations == () and answer.unverified_labels == (2,)
    assert answer.visible == "The court ruled."


def test_a_quote_may_leave_text_out_with_an_ellipsis():
    text = (
        'The court ruled [3].\n\nQUOTES:\n[3] "The court said ... about the claim."\n'
    )

    assert [c.label for c in answer_with(text).citations] == [3]


def test_a_label_no_excerpt_carried_is_a_wrong_citation_and_leaves_the_visible_answer():
    answer = answer_with(reply("It says so [2] and so [7] and [0].", 2))

    assert [c.label for c in answer.citations] == [2]
    assert answer.unknown_labels == (7, 0)
    assert answer.visible == "It says so [2] and so and."


def test_the_models_refusal_cites_nothing():
    answer = answer_with(REFUSAL_SENTENCE + " [1]")

    assert answer.refused and answer.citations == () and answer.unknown_labels == ()
    assert answer.visible == REFUSAL_SENTENCE + " [1]"


def test_an_answer_without_labels_has_no_citations():
    answer = answer_with("The court dismissed the claim (a.docx, page 2).")

    assert answer.citations == () and answer.unverified_labels == ()
    assert answer.visible == "The court dismissed the claim (a.docx, page 2)."


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


@pytest.mark.parametrize(
    ("text", "body", "quotes"),
    [
        ('Answer [1].\n\nQUOTES:\n[1] "a passage"', "Answer [1].", {1: "a passage"}),
        (
            "Answer [1].\nQUOTES:\n[1] a passage without marks",
            "Answer [1].",
            {1: "a passage without marks"},
        ),
        (
            'Answer [1].\n\nQUOTES:\n[1] "first"\n[1] "second"\n[2] “curly”',
            "Answer [1].",
            {1: "first", 2: "curly"},
        ),
        ("No block here [1].", "No block here [1].", {}),
        ("Answer.\n\nQUOTES:\n", "Answer.", {}),
        ('Answer.\n\nQUOTES:\nnot a quote line\n[3] ""', "Answer.", {}),
    ],
)
def test_the_quotes_block_is_split_off_and_read(text, body, quotes):
    assert split_quotes(text) == (body, quotes)


@pytest.mark.parametrize(
    ("text", "labels", "expected"),
    [
        ("A fact [1]. B fact [2].", {1}, "A fact. B fact [2]."),
        ("A fact [1][2].", {1, 2}, "A fact."),
        ("A fact [1, 3].", {1}, "A fact [3]."),
        ("A fact [1; 3].", {3}, "A fact [1]."),
        ("In [2023] it was [4].", {4}, "In [2023] it was."),
        ("No labels here.", {1}, "No labels here."),
        ("A fact [1].", set(), "A fact [1]."),
    ],
)
def test_labels_are_taken_out_of_the_visible_text(text, labels, expected):
    assert strip_labels(text, labels) == expected


def test_the_sources_block_names_the_cited_chunks_with_their_quotes_and_is_empty_without_citations():
    assert render_sources(()) == ""
    block = render_sources(
        (Citation(2, 20, "b.docx", 7, "a passage"), Citation(1, 30, "c.docx", 0))
    )

    assert (
        block == 'Sources:\n[2] b.docx, section 7: "a passage"\n[1] c.docx, section 0'
    )


def test_a_long_quote_is_cut_in_the_sources_block():
    block = render_sources((Citation(1, 1, "a.docx", 0, "x" * 400),))

    assert len(block.splitlines()[1]) < 220 and block.endswith('…"')


def test_a_line_starting_with_quotes_inside_the_answer_does_not_hide_the_real_block():
    text = 'Intro.\nQUOTES: is also a word in this answer [3].\n\nQUOTES:\n[3] "said text 10 about"\n'

    body, quotes = split_quotes(text)

    assert quotes == {3: "said text 10 about"}
    assert body.endswith("in this answer [3].")  # the LAST mark ends the answer


def test_the_system_prompt_asks_for_a_word_for_word_quote_per_label_and_none_when_refusing():
    from drivers.llm import _build_prompt

    system_prompt, _ = _build_prompt("Q?", [])

    assert "QUOTES:" in system_prompt
    assert "copied word for word" in system_prompt
    assert "do not write QUOTES: when you refuse" in system_prompt
