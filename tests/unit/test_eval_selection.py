"""Tests for corpus.commands.eval's question selection and repeat summary."""

import pytest
import typer

from corpus.commands.eval import print_repeat_summary, select_questions

QUESTIONS = [
    {"id": "q0001"},
    {"id": "q0010"},
    {"id": "q0012"},
]


def test_select_questions_returns_everything_without_ids():
    assert select_questions(QUESTIONS, None) == QUESTIONS
    assert select_questions(QUESTIONS, []) == QUESTIONS


def test_select_questions_keeps_only_the_requested_ids_in_the_order_given():
    result = select_questions(QUESTIONS, ["q0012", "q0001"])

    assert [q["id"] for q in result] == ["q0012", "q0001"]


def test_select_questions_ignores_a_repeated_id():
    assert [q["id"] for q in select_questions(QUESTIONS, ["q0010", "q0010"])] == [
        "q0010"
    ]


def test_select_questions_rejects_an_unknown_id_and_lists_the_valid_ones():
    with pytest.raises(typer.BadParameter, match="q9999"):
        select_questions(QUESTIONS, ["q0010", "q9999"])


def _run(qid, answer_ok, retrieval_ok, answer="valasz"):
    return {
        "question_id": qid,
        "persona_id": "synthesizer",
        "answer": answer,
        "grades": {
            "independent_fact": {
                "answer_correct": answer_ok,
                "retrieval_hit": retrieval_ok,
            }
        },
    }


def test_repeat_summary_reports_pass_counts_and_flags_a_question_below_target(capsys):
    results = [
        _run("q0010", True, True),
        _run("q0010", False, True, answer="I could not find this information"),
        _run("q0012", True, True),
        _run("q0012", True, True),
    ]

    print_repeat_summary(results, target=0.95)

    out = capsys.readouterr().out
    q0010 = out.split("q0010")[1].split("q0012")[0]
    q0012 = out.split("q0012")[1]
    assert "answer correct 1/2" in q0010
    assert "refused 1/2" in q0010
    assert "below target" in q0010
    assert "answer correct 2/2" in q0012
    assert "below target" not in q0012


class _FakeStore:
    """Stands in for store.VectorStore: only the ingested file names matter here."""

    def get_all_source_files(self):
        return {"A_P_1_2020_1.docx", "B_P_2_2021_2.docx"}

    def search_by_identifier(self, tokens, top_k, per_token=False):
        raise AssertionError("must not fall back to identifier search when files are named")


def test_cited_documents_are_the_file_names_the_answer_names(monkeypatch):
    """Regression: a fraction ('15/100-ad') and a statute fragment ('1)-(2') in
    the answer were matched literally against document content, 'citing'
    unrelated courts' documents the answer never mentioned."""
    import store
    from corpus.commands.eval import _resolve_cited_source_files

    monkeypatch.setattr(store, "VectorStore", _FakeStore)
    answer = (
        "A felperes 15/100-ad tulajdoni illetősége (Ptk. 5:84. § (1)-(2)) "
        "(A_P_1_2020_1.docx, 3. oldal) és (B_P_2_2021_2.docx, 2. oldal); "
        "lásd még Nincs_Ilyen.docx."
    )

    assert _resolve_cited_source_files(answer) == ["A_P_1_2020_1.docx", "B_P_2_2021_2.docx"]
