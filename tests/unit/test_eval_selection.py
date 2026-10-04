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
