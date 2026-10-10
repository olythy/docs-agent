"""Grading adversarial answers by what they say, and reading the results per question."""

from types import SimpleNamespace

import pytest

from corpus.decline_grader import grade_decline
from corpus.stats import wilson_lower


def judge_says(declined, asserted):
    return lambda question, answer: {"declined": declined, "asserted": asserted}


def graded(declined, asserted):
    grade = grade_decline("q", "a", judge_says(declined, asserted))
    assert grade is not None
    return grade


def test_a_decline_that_asserts_nothing_is_correct():
    assert graded(True, False).correct is True


def test_a_decline_that_still_states_the_fact_is_wrong():
    assert graded(True, True).correct is False


def test_an_answer_that_states_the_fact_without_declining_is_wrong():
    assert graded(False, True).correct is False


def test_an_unusable_judge_reply_gives_no_grade():
    assert grade_decline("q", "a", lambda q, a: None) is None
    assert grade_decline("q", "a", lambda q, a: {"declined": "yes"}) is None


def test_wilson_lower_bound():
    assert wilson_lower(0, 0) is None
    assert wilson_lower(30, 30) == pytest.approx(0.886, abs=0.005)
    assert wilson_lower(60, 60) == pytest.approx(0.940, abs=0.005)
    assert wilson_lower(0, 10) == 0.0


def test_a_question_passes_only_if_every_run_was_right():
    from corpus.commands.eval import _question_passed

    right = {"grades": {"exact_match": {"answer_correct": True}}}
    wrong = {"grades": {"exact_match": {"answer_correct": False}}}

    assert _question_passed([right, right, right]) is True
    assert _question_passed([right, wrong, right]) is False
    assert _question_passed([{"is_decline": True}, {"is_decline": False}]) is False
    assert _question_passed([{}]) is None


def test_the_adversarial_result_uses_the_models_reading_and_keeps_the_keywords(
    monkeypatch,
):
    import corpus.commands.eval as ev
    import corpus.decline_grader as dg

    full = SimpleNamespace(
        text="nem említi, de 5 millió Ft", explain=SimpleNamespace(chunks=[])
    )
    monkeypatch.setattr(ev, "_answer", lambda question, profile: full)
    monkeypatch.setattr(
        dg, "llm_decline_judge", lambda q, a: {"declined": True, "asserted": True}
    )
    question = {
        "id": "q1",
        "persona_id": "adversarial",
        "question": "Mennyi?",
        "citations": [],
    }

    result = ev.evaluate_one(question, None, {})

    assert "decline_keywords" in result
    assert result["is_decline"] is False
    assert result["decline_grade"] == {"declined": True, "asserted": True}


def test_without_a_usable_judge_the_keyword_check_decides(monkeypatch):
    import corpus.commands.eval as ev
    import corpus.decline_grader as dg

    full = SimpleNamespace(
        text="I could not find this information in the provided documents.",
        explain=SimpleNamespace(chunks=[]),
    )
    monkeypatch.setattr(ev, "_answer", lambda question, profile: full)
    monkeypatch.setattr(dg, "llm_decline_judge", lambda q, a: None)
    question = {
        "id": "q1",
        "persona_id": "adversarial",
        "question": "?",
        "citations": [],
    }

    result = ev.evaluate_one(question, None, {})

    assert result["is_decline"] is True and result["decline_grade"] is None
