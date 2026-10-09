"""Giving golden citations a verified quote (a scripted model; no database)."""

from corpus.golden_quotes import MAX_QUOTE_CHARS, add_quotes, locate

CHUNKS = [
    (
        0,
        "A bíróság a felperes keresetét elutasítja. A per tárgya tulajdonjog megállapítása.",
    ),
    (1, "Az indoklás szerint a felperes nem bizonyította az elbirtoklás feltételeit."),
]


def fetch(source_file):
    return CHUNKS


def question(qid="q1", citations=None):
    return {
        "id": qid,
        "persona_id": "fact_finder",
        "question": "Mi lett az eredmény?",
        "expected_answer": "A keresetet elutasították.",
        "citations": citations
        if citations is not None
        else [{"court": "C", "case_number": "1/2020", "source_file": "a.docx"}],
    }


class Ask:
    """Replies with scripted dicts, one per call, and remembers the feedback it was given."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, q, expected, citation, chunks, feedback):
        self.calls.append(feedback)
        return self.replies.pop(0)


def test_a_quote_that_is_in_a_chunk_is_stored_on_the_citation():
    questions = [question()]

    report = add_quotes(
        questions, fetch, Ask({"quote": "a felperes keresetét elutasítja"})
    )

    assert questions[0]["citations"][0]["quote"] == "a felperes keresetét elutasítja"
    assert (
        report.added == [("q1", "a.docx", "a felperes keresetét elutasítja")]
        and not report.failed
    )


def test_a_quote_that_is_not_in_any_chunk_is_refused_and_asked_for_again_with_the_reason():
    questions = [question()]
    ask = Ask(
        {"quote": "a bíróság helyt ad"},
        {"quote": "A per tárgya tulajdonjog megállapítása"},
    )

    report = add_quotes(questions, fetch, ask)

    assert (
        questions[0]["citations"][0]["quote"]
        == "A per tárgya tulajdonjog megállapítása"
    )
    assert ask.calls == ["", "That quote is not word for word in any excerpt."]
    assert len(report.added) == 1 and not report.failed


def test_a_citation_the_model_cannot_quote_is_reported_and_left_without_one():
    questions = [question()]
    ask = Ask({"quote": "invented"}, {"quote": "still invented"}, {"quote": "nope"})

    report = add_quotes(questions, fetch, ask)

    assert "quote" not in questions[0]["citations"][0]
    assert report.failed == [
        ("q1", "a.docx", "That quote is not word for word in any excerpt.")
    ]


def test_a_missing_quote_is_asked_for_again_with_the_reason():
    questions = [question()]
    ask = Ask(None, {"quote": "A per tárgya tulajdonjog megállapítása"})

    report = add_quotes(questions, fetch, ask)

    assert ask.calls == ["", "Your reply had no quote."]
    assert len(report.added) == 1 and not report.failed


def test_a_verified_quote_that_is_too_long_is_cut_at_a_word_boundary():
    long_text = " ".join(f"szó{i}" for i in range(200))
    assert len(long_text) > MAX_QUOTE_CHARS
    questions = [question()]

    report = add_quotes(
        questions, lambda source_file: [(0, long_text)], Ask({"quote": long_text})
    )

    stored = questions[0]["citations"][0]["quote"]
    assert 0 < len(stored) <= MAX_QUOTE_CHARS
    assert long_text.startswith(stored) and long_text[len(stored)] == " "
    assert report.added and not report.failed


def test_a_citation_that_already_has_a_quote_is_not_asked_about_again():
    questions = [question(citations=[{"source_file": "a.docx", "quote": "already"}])]
    ask = Ask()

    report = add_quotes(questions, fetch, ask)

    assert (
        ask.calls == []
        and report.skipped == 1
        and questions[0]["citations"][0]["quote"] == "already"
    )


def test_only_the_selected_questions_are_touched_and_a_question_without_citations_is_fine():
    q1, q2, q3 = question("q1"), question("q2"), question("q3", citations=[])
    ask = Ask({"quote": "A per tárgya tulajdonjog megállapítása"})

    add_quotes([q1, q2, q3], fetch, ask, only=lambda q: q["id"] == "q2")

    assert (
        "quote" not in q1["citations"][0]
        and "quote" in q2["citations"][0]
        and q3["citations"] == []
    )


def test_each_citation_of_a_multi_document_question_gets_its_own_quote():
    q = question(citations=[{"source_file": "a.docx"}, {"source_file": "b.docx"}])
    ask = Ask(
        {"quote": "a felperes keresetét elutasítja"},
        {"quote": "elbirtoklás feltételeit"},
    )

    add_quotes([q], fetch, ask)

    assert [c["quote"] for c in q["citations"]] == [
        "a felperes keresetét elutasítja",
        "elbirtoklás feltételeit",
    ]


def test_locate_names_every_chunk_that_contains_the_quote_ignoring_case_and_accents():
    assert locate("A FELPERES KERESETÉT ELUTASÍTJA", CHUNKS) == [0]
    assert locate("elbirtoklas feltetelei", CHUNKS) == [1]
    assert (
        locate("a felperes ... elbirtoklas", CHUNKS) == []
    )  # fragments of one chunk only
    assert locate("a felperes", CHUNKS) == [0, 1]
    assert locate("", CHUNKS) == []


def test_section_hits_needs_every_passage_in_a_chunk_of_its_document():
    from corpus.golden_quotes import section_hits

    citations = [
        {"source_file": "a.pdf", "quote": "the contract is void"},
        {"source_file": "b.pdf", "quote": "costs are shared"},
    ]
    both = [("a.pdf", "so the contract is void here"), ("b.pdf", "costs are shared")]
    assert section_hits(citations, both) is True
    assert section_hits(citations, both[:1]) is False
    # right text, wrong document
    assert (
        section_hits(citations, [("b.pdf", "the contract is void"), both[1]]) is False
    )


def test_section_hits_is_none_without_quotes():
    from corpus.golden_quotes import section_hits

    assert section_hits([], [("a.pdf", "x")]) is None
    assert section_hits([{"source_file": "a.pdf"}], [("a.pdf", "x")]) is None
