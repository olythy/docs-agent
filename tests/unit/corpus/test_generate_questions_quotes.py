"""A verified question's citations get a verified quote, or the question needs review."""

from corpus.commands.generate_questions import quote_citations

TEXT = "A bíróság a felperes keresetét elutasítja."


def entry():
    return {
        "id": "q1",
        "question": "Mi lett az eredmény?",
        "expected_answer": "Elutasították.",
        "citations": [{"source_file": "a.docx"}],
        "verification_status": "verified",
        "verification_notes": "ok",
    }


def test_a_quotable_citation_keeps_the_question_verified():
    e = entry()

    quote_citations(e, fetch=lambda f: [(0, TEXT)], ask=lambda *a: {"quote": TEXT})

    assert e["citations"][0]["quote"] == TEXT and e["verification_status"] == "verified"


def test_an_unquotable_citation_sends_the_question_to_review():
    e = entry()

    quote_citations(e, fetch=lambda f: [(0, TEXT)], ask=lambda *a: {"quote": "made up"})

    assert "quote" not in e["citations"][0]
    assert (
        e["verification_status"] == "needs_review"
        and "a.docx" in e["verification_notes"]
    )


def trap(**over):
    base = {
        "question": "Mit döntött a bíróság a 9.P.20.123/2021/5. számú ügyben?",
        "trap_case_number": "9.P.20.123/2021/5.",
        "citations": [],
    }
    return {**base, **over}


SAMPLE = {"content": "text", "case_number": "1.P.1/2020/1"}


def test_an_invented_case_number_that_exists_in_no_document_is_a_valid_trap():
    from corpus.commands.generate_questions import verify_trap

    verdict, _ = verify_trap(
        trap(), "invented_case_number", SAMPLE, exists=lambda n: False
    )

    assert verdict == "SUPPORTED"


def test_an_invented_case_number_that_exists_is_not_a_trap():
    from corpus.commands.generate_questions import verify_trap

    verdict, reason = verify_trap(
        trap(), "invented_case_number", SAMPLE, exists=lambda n: True
    )

    assert verdict == "NOT_SUPPORTED" and "exists" in reason


def test_the_invented_number_must_be_in_the_question_and_nothing_may_be_cited():
    from corpus.commands.generate_questions import verify_trap

    never = lambda n: False
    absent = trap(question="Mit döntött a bíróság?")
    cited = trap(citations=[{"source_file": "a.docx"}])

    assert (
        verify_trap(absent, "invented_case_number", SAMPLE, never)[0] == "NOT_SUPPORTED"
    )
    assert (
        verify_trap(cited, "invented_case_number", SAMPLE, never)[0] == "NOT_SUPPORTED"
    )


def test_a_missing_fact_trap_needs_the_judge_to_find_the_detail_absent():
    from corpus.commands.generate_questions import verify_trap

    def judge_says(verdict):
        return lambda drafted, content: (verdict, "because")

    assert (
        verify_trap(trap(), "missing_fact", SAMPLE, judge=judge_says("ABSENT"))[0]
        == "SUPPORTED"
    )
    assert (
        verify_trap(trap(), "missing_fact", SAMPLE, judge=judge_says("PRESENT"))[0]
        == "NOT_SUPPORTED"
    )
    assert (
        verify_trap(trap(), "missing_fact", SAMPLE, judge=judge_says("UNCLEAR"))[0]
        == "NOT_SUPPORTED"
    )


def test_a_question_must_name_court_and_year_and_never_the_case_number():
    from corpus.commands.generate_questions import identifies_by_content

    sample = {"court": "Fővárosi Ítélőtábla", "year": 2022, "subject": "ingatlan"}
    cites = [{"case_number": "9.Pf.20.270/2023/4"}]

    assert (
        identifies_by_content(
            "A fővárosi ítélőtábla 2022-ben mit döntött?", sample, cites
        )
        is None
    )
    for question, word in (
        ("Mit döntött a bíróság 2022-ben?", "court"),
        ("Mit döntött a Fővárosi Ítélőtábla?", "year"),
        (
            "A Fővárosi Ítélőtábla 2022-ben a 9.Pf.20.270/2023/4 ügyben mit döntött?",
            "case number",
        ),
    ):
        assert word in (identifies_by_content(question, sample, cites) or "")


def test_appending_reads_the_file_again_so_parallel_runs_do_not_overwrite(tmp_path):
    from corpus.commands.generate_questions import append_question, save_questions

    path = tmp_path / "q.json"
    first, second = {"question": "a"}, {"question": "b"}
    append_question(first, path)
    # another run added a question after this one had loaded the file
    save_questions(
        [{"id": "q0001", "question": "a"}, {"id": "q0002", "question": "x"}], path
    )
    questions = append_question(second, path)

    assert [q["id"] for q in questions] == ["q0001", "q0002", "q0003"]


def test_a_case_number_of_an_uncited_document_in_the_question_is_flagged():
    from corpus.commands.generate_questions import names_another_document

    owners = {"9.G.40.230/2020/32": "first_instance.docx"}
    find = lambda number, _source: owners.get(number)
    cited = [{"source_file": "appeal.docx"}]

    problem = names_another_document(
        "Az 9.G.40.230/2020/32 ügyben mi lett?", cited, find
    )

    assert problem and "first_instance.docx" in problem
    # the same number is fine when that document is the cited one
    assert (
        names_another_document(
            "Az 9.G.40.230/2020/32 ügyben mi lett?",
            [{"source_file": "first_instance.docx"}],
            find,
        )
        is None
    )
    assert names_another_document("Mi lett az ügyben?", cited, find) is None


def test_a_description_that_fits_other_documents_is_ambiguous():
    from corpus.commands.generate_questions import ambiguous_description

    sample = {"source_file": "a.docx"}
    cands = lambda s: [{"source_file": "a.docx"}, {"source_file": "b.docx"}]

    only_own = ambiguous_description("q", sample, cands, lambda q, c: ["a.docx"])
    shared = ambiguous_description(
        "q", sample, cands, lambda q, c: ["a.docx", "b.docx"]
    )
    lost = ambiguous_description("q", sample, cands, lambda q, c: ["b.docx"])

    assert only_own is None
    assert shared and "b.docx" in shared
    assert lost and "own document" in lost
