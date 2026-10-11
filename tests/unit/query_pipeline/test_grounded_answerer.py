"""Writing an answer from the chunks read: the policy is explicit, the model's refusal is recognised."""

import pytest

import drivers.llm as llm_module
from drivers.llm import REFUSAL_SENTENCE, AnswerDriver, _build_prompt
from models import ChunkMetadata, Citation, RetrievedChunk
from query.answering import (
    AnswerPolicy,
    GroundedAnswerer,
    render_sources,
    split_quotes,
)
from query.claims import cited_labels, strip_labels


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
    "reply", ["STATUS: none", "status: NONE\n", "**STATUS: none**"]
)
def test_status_none_is_a_refusal_the_system_writes(reply):
    answer = GroundedAnswerer(FakeDriver(reply), POLICIES[0]).answer("Q?", CHUNKS)

    assert answer.refused and answer.text == REFUSAL_SENTENCE
    assert answer.status == "none" and answer.visible == REFUSAL_SENTENCE


@pytest.mark.parametrize(
    "reply",
    [
        "The court decided X.",
        "STATUS: answered\nThe court decided X.",  # claims, nothing it can point at
        f"X was decided. {REFUSAL_SENTENCE}",
        "",
    ],
)
def test_a_reply_without_one_citation_that_holds_is_refused_by_the_system(reply):
    answer = GroundedAnswerer(FakeDriver(reply), POLICIES[0]).answer("Q?", CHUNKS)

    assert answer.refused and answer.text == REFUSAL_SENTENCE and answer.citations == ()


def test_a_reply_with_a_citation_that_holds_is_an_answer_and_keeps_its_status():
    reply = 'STATUS: partial\nThe court ruled [2]. Costs are not covered.\n\nQUOTES:\n[2] "first"'
    answer = GroundedAnswerer(FakeDriver(reply), POLICIES[0]).answer("Q?", CHUNKS)

    assert not answer.refused and answer.status == "partial"
    assert [c.label for c in answer.citations] == [2]
    assert answer.visible.startswith("The court ruled [2]. Costs are not covered.")


def test_the_prompt_asks_for_a_status_line_and_not_for_the_exact_refusal_sentence():
    system, _ = _build_prompt("q", [])

    for status in ("STATUS: answered", "STATUS: partial", "STATUS: none"):
        assert status in system
    assert REFUSAL_SENTENCE not in system


@pytest.mark.parametrize(
    ("text", "status", "rest"),
    [
        ("STATUS: answered\nBody.", "answered", "Body."),
        ("  status : Partial  (some)\n\nBody.", "partial", "Body."),
        ("**STATUS: none**", "none", ""),
        (
            "Body. STATUS: none",
            None,
            "Body. STATUS: none",
        ),  # only an opening line counts
        ("STATUS: maybe\nBody.", None, "STATUS: maybe\nBody."),
    ],
)
def test_the_status_line_is_read_only_when_it_opens_the_reply(text, status, rest):
    from query.answering import split_status

    assert split_status(text) == (status, rest)


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
    # the sentence whose only citation failed states what nothing backs: it is left out
    assert answer.visible == "Claim one [3]." and answer.omitted == 1


def test_a_quote_that_is_not_in_the_chunk_under_that_label_is_not_a_citation():
    # label 2's quote is true, but of label 3's chunk: the label names another chunk
    text = 'The court ruled [2].\n\nQUOTES:\n[2] "said text 10 about"\n'

    answer = answer_with(text)

    # no citation holds, so the system refuses instead of showing an uncited claim
    assert answer.citations == () and answer.unverified_labels == (2,)
    assert answer.refused and answer.visible == REFUSAL_SENTENCE


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
    assert answer.refused and answer.visible == REFUSAL_SENTENCE


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
        ('Answer [1].\n\nQUOTES:\n[1] "a passage"', "Answer [1].", [(1, "a passage")]),
        (
            "Answer [1].\nQUOTES:\n[1] a passage without marks",
            "Answer [1].",
            [(1, "a passage without marks")],
        ),
        (
            'Answer [1].\n\nQUOTES:\n[1] "first"\n[1] "second"\n[2] “curly”',
            "Answer [1].",
            [
                (1, "first"),
                (1, "second"),
                (2, "curly"),
            ],  # one line per citation, in order
        ),
        ("No block here [1].", "No block here [1].", []),
        ("Answer.\n\nQUOTES:\n", "Answer.", []),
        ('Answer.\n\nQUOTES:\nnot a quote line\n[3] ""', "Answer.", []),
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

    assert quotes == [(3, "said text 10 about")]
    assert body.endswith("in this answer [3].")  # the LAST mark ends the answer


def test_the_system_prompt_asks_for_a_word_for_word_quote_per_label_and_none_for_status_none():
    from drivers.llm import _build_prompt

    system_prompt, _ = _build_prompt("Q?", [])

    assert "QUOTES:" in system_prompt
    assert "copied word for word" in system_prompt
    assert "do not write QUOTES: when STATUS is none" in system_prompt


# ---------------------------------------------------- one quote per citation, judged


class Judge:
    """A claim judge with scripted verdicts; remembers what it was asked."""

    def __init__(self, verdicts=None):
        self.verdicts, self.asked = verdicts, []

    def supports(self, pairs):
        self.asked.append(list(pairs))
        if callable(self.verdicts):
            return self.verdicts(pairs)
        return self.verdicts


def judged(text, judge, chunks=RANKED):
    return GroundedAnswerer(
        FakeDriver(text), AnswerPolicy(expose_document_date=False), judge=judge
    ).answer("Q?", chunks)


TWO = (
    "The court dismissed the claim [3]. It ordered the costs [3][2].\n\nQUOTES:\n"
    '[3] "said text 10 about"\n[3] "The court said text 10"\n[2] "said text 20 about"\n'
)


def test_the_judge_is_asked_about_each_sentence_and_every_verified_quote_of_its_label():
    judge = Judge(lambda pairs: [True] * len(pairs))

    judged(TWO, judge)

    assert judge.asked == [
        [
            ("The court dismissed the claim.", "said text 10 about"),
            ("The court dismissed the claim.", "The court said text 10"),
            ("It ordered the costs.", "said text 10 about"),
            ("It ordered the costs.", "The court said text 10"),
            ("It ordered the costs.", "said text 20 about"),
        ]
    ]


def test_a_citation_the_judge_rejects_leaves_only_that_sentence_and_is_recorded():
    # no quote of label 2 states the second sentence
    judge = Judge([True, False, True, False, False])

    answer = judged(TWO, judge)

    assert (
        answer.visible == "The court dismissed the claim [3]. It ordered the costs [3]."
    )
    assert [c.label for c in answer.citations] == [3]
    assert answer.rejected == ((2, "It ordered the costs."),)
    assert answer.unverified_labels == (2,)  # label 2 never held anywhere


def test_a_label_rejected_in_one_sentence_but_kept_in_another_is_still_a_citation():
    judge = Judge([False, False, True, True, True])

    answer = judged(TWO, judge)

    # the first sentence's only citation failed, so it is left out; the second holds
    assert answer.visible == "It ordered the costs [3][2]." and answer.omitted == 1
    assert [c.label for c in answer.citations] == [3, 2]
    assert answer.unverified_labels == ()  # label 3 holds in the second sentence


def test_a_quote_written_for_another_sentence_does_not_reject_a_sound_citation():
    """The model wrote an extra quote for the label (one for the case number), so pairing
    quotes with sentences by position gave the second sentence the wrong passage and the
    judge rejected a correct citation. Any verified quote of the label may state it."""
    text = (
        "First claim is made here [3]. Second claim is made here [3].\n\nQUOTES:\n"
        '[3] "The court said text 10"\n[3] "said text 10 about"\n'
    )
    states = {
        "First claim is made here.": "The court said text 10",
        "Second claim is made here.": "said text 10 about",
    }
    judge = Judge(lambda pairs: [states[claim] == quote for claim, quote in pairs])

    answer = judged(text, judge)

    assert (
        answer.visible == "First claim is made here [3]. Second claim is made here [3]."
    )
    assert answer.rejected == () and [c.label for c in answer.citations] == [3]
    # the quote shown for the label is the first one that stated a sentence
    assert answer.citations[0].quote == "The court said text 10"


def test_a_single_quote_for_a_label_cited_twice_is_reused_for_the_second_sentence():
    text = (
        "First claim is made here [3]. Second claim is made here [3].\n\n"
        'QUOTES:\n[3] "said text 10 about"\n'
    )
    judge = Judge(lambda pairs: [True] * len(pairs))

    judged(text, judge)

    assert [quote for _, quote in judge.asked[0]] == ["said text 10 about"] * 2


def test_a_judge_that_gives_no_verdicts_keeps_the_quotes_that_are_in_their_chunks():
    answer = judged(TWO, Judge(None))

    assert [c.label for c in answer.citations] == [3, 2]
    assert answer.rejected == () and answer.unverified_labels == ()


def test_without_a_judge_a_quote_in_the_chunk_is_enough():
    answer = answer_with(TWO)

    assert [c.label for c in answer.citations] == [3, 2]


def test_the_judge_is_not_asked_when_nothing_is_cited_or_no_quote_is_in_a_chunk():
    judge = Judge(lambda pairs: [True] * len(pairs))

    judged("No labels in this answer at all.", judge)
    judged('A claim is made here [3].\n\nQUOTES:\n[3] "not in any chunk text"', judge)

    assert judge.asked == [[], []]


def test_the_citation_keeps_the_first_quote_that_held():
    answer = judged(TWO, Judge([False, True, True]))

    assert answer.citations[0] == Citation(3, 10, "a.docx", 4, "The court said text 10")


def test_an_unknown_label_cited_twice_is_one_wrong_citation_not_two():
    answer = answer_with("First claim here [7]. Second claim here [7].")

    assert answer.unknown_labels == (7,)


def test_the_prompt_asks_for_one_quote_line_per_citation_in_the_order_of_the_citations():
    from drivers.llm import _build_prompt

    system_prompt, _ = _build_prompt("Q?", [])

    assert (
        "one line each time you cite a label, in the order of your citations"
        in system_prompt
    )
    assert "the fact of the sentence you cited it for" in system_prompt


# ------------------------------------------------ the summary that opens every chunk


def with_summary(chunk_id, summary, body, index=0):
    return RetrievedChunk(
        id=chunk_id,
        content=f"{summary}\n\n{body}",
        metadata=ChunkMetadata(
            source_file="a.docx",
            page_number=None,
            chunk_index=index,
            document_summary=summary,
        ),
        score=0.5,
    )


SUMMARY = "The court dismissed the plaintiff's claim about the gift."
SUMMED = (
    with_summary(1, SUMMARY, "The amount is 733910 forint.", 0),
    with_summary(2, SUMMARY, "The case concerned a gift of land.", 1),
)


def test_a_quote_from_the_document_summary_is_not_a_citation_of_any_chunk():
    # the summary is in every chunk's content, but it is not the chunk's own text
    text = 'It was dismissed [1].\n\nQUOTES:\n[1] "dismissed the plaintiff\'s claim"\n'

    answer = answer_with(text, SUMMED)

    assert answer.citations == () and answer.unverified_labels == (1,)


def test_a_quote_from_the_excerpt_itself_is_a_citation():
    text = (
        'The amount is stated here [2].\n\nQUOTES:\n[2] "The amount is 733910 forint"\n'
    )

    answer = answer_with(text, SUMMED)

    assert [c.label for c in answer.citations] == [2]


def test_the_prompt_shows_the_summary_apart_from_the_excerpt():
    from drivers.llm import _build_prompt

    _, user_message = _build_prompt("Q?", list(SUMMED))

    assert f"Document summary (not to be quoted): {SUMMARY}" in user_message
    assert "Excerpt: The amount is 733910 forint." in user_message
    assert user_message.count(SUMMARY) == 2  # once per excerpt, labelled as context


def packaged(
    body,
    summary="A summary of the case.",
    identifiers=("4.P.20.1/2021/5",),
    header="I. Reasoning",
    chunk_id=1,
    index=3,
):
    """A chunk exactly as the ingest stores it: summary, identifiers, header, then the text."""
    from ingestion.chunker import _package_chunk

    metadata = ChunkMetadata(
        source_file="a.docx",
        page_number=None,
        chunk_index=index,
        document_summary=summary,
        document_identifiers=identifiers,
        header_path=header,
    )
    return RetrievedChunk(
        id=chunk_id,
        content=_package_chunk(body, metadata).content,
        metadata=metadata,
        score=0.5,
    )


def test_a_stored_chunk_has_three_layers_before_its_own_text_and_the_body_is_only_the_text():
    chunk = packaged("The court granted the claim in full.")

    assert chunk.content.startswith("A summary of the case.")
    assert "4.P.20.1/2021/5" in chunk.content and "I. Reasoning" in chunk.content
    assert chunk.body == "The court granted the claim in full."


def test_a_quote_from_the_identifiers_or_the_heading_is_not_a_citation_either():
    chunk = packaged("The court granted the claim in full.")
    for quote in ("4.P.20.1/2021/5", "A summary of the case."):
        text = f'It is so [1].\n\nQUOTES:\n[1] "{quote}"\n'

        answer = answer_with(text, (chunk,))

        assert answer.citations == () and answer.unverified_labels == (1,)


def test_a_quote_from_the_chunks_own_text_is_a_citation_of_a_packaged_chunk():
    chunk = packaged("The court granted the claim in full.")
    text = 'It is so [1].\n\nQUOTES:\n[1] "granted the claim in full"\n'

    assert [c.label for c in answer_with(text, (chunk,)).citations] == [1]


def test_the_prompt_gives_the_summary_and_the_heading_as_context_and_the_text_as_the_excerpt():
    from drivers.llm import _build_prompt

    _, user_message = _build_prompt("Q?", [packaged("The court granted the claim.")])

    assert "Document summary (not to be quoted): A summary of the case." in user_message
    assert "Section heading (not to be quoted): I. Reasoning" in user_message
    assert "Excerpt: The court granted the claim." in user_message
    assert (
        "4.P.20.1/2021/5" not in user_message
    )  # the identifiers prefix is not shown twice


def test_a_chunk_without_any_context_is_shown_as_it_is():
    from drivers.llm import _build_prompt

    bare = RetrievedChunk(
        id=9,
        content="Just text.",
        metadata=ChunkMetadata(source_file="a.docx", page_number=None, chunk_index=0),
        score=0.5,
    )

    _, user_message = _build_prompt("Q?", [bare])

    assert "[1] Source: a.docx, section 0\nJust text." in user_message
    assert "Document summary" not in user_message and "Excerpt:" not in user_message


def test_a_chunk_with_only_a_leading_summary_still_has_its_text_as_the_body():
    assert SUMMED[0].body == "The amount is 733910 forint."
    assert SUMMED[0].content.startswith(SUMMARY)


def test_a_quote_the_model_repeats_for_a_label_is_asked_about_once():
    text = (
        "First claim is made here [3].\n\nQUOTES:\n"
        '[3] "said text 10 about"\n[3] "said text 10 about"\n'
    )
    judge = Judge(lambda pairs: [True] * len(pairs))

    judged(text, judge)

    assert judge.asked == [[("First claim is made here.", "said text 10 about")]]


def test_with_omission_off_a_sentence_whose_citation_failed_stays_without_its_label():
    reply_text = reply("Claim one [3]. Claim two [2].", 3)  # no quote for label 2

    answer = GroundedAnswerer(
        FakeDriver(reply_text), AnswerPolicy(False, omit_unbacked=False)
    ).answer("Q?", RANKED)

    assert answer.visible == "Claim one [3]. Claim two." and answer.omitted == 0
    assert [c.label for c in answer.citations] == [3]
