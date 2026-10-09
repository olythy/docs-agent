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
