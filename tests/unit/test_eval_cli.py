"""Unit tests for scripts.eval_cli (pure logic, fixtures discovery, decline detection)."""

import json

import scripts.eval_cli as eval_cli_module
from scripts.eval_cli import (
    _looks_like_a_decline,
    discover_eval_fixtures,
)


def test_discover_eval_fixtures_derives_from_question_source_files(
    tmp_path, monkeypatch
):
    """discover_eval_fixtures() must read exactly the files
    sample_questions.json's questions reference, not scan a directory --
    scanning would silently couple this eval set to whatever else lives in
    tests/data/ (real pytest DB fixtures included), added for unrelated
    reasons."""
    questions_path = tmp_path / "sample_questions.json"
    questions_path.write_text(
        json.dumps(
            [
                {"expected_source_file": "sample.pdf"},
                {"expected_source_file": "sample.md"},
                {"expected_source_file": "sample.pdf"},  # duplicate, deduped
                {"question": "no expected_source_file at all"},
            ]
        )
    )
    monkeypatch.setattr(eval_cli_module, "EVAL_QUESTIONS_PATH", questions_path)
    monkeypatch.setattr(eval_cli_module, "EVAL_DATA_DIR", tmp_path)

    fixtures = discover_eval_fixtures()
    names = [f.name for f in fixtures]

    assert names == ["sample.md", "sample.pdf"]


def test_looks_like_a_decline_english():
    assert (
        _looks_like_a_decline(
            "I could not find information about that in the document."
        )
        is True
    )
    assert (
        _looks_like_a_decline("The provided context does not mention the fee.") is True
    )
    assert _looks_like_a_decline("Unable to find the requested details.") is True


def test_looks_like_a_decline_hungarian():
    assert (
        _looks_like_a_decline("A megadott szövegben nem találtam információt erről.")
        is True
    )
    assert _looks_like_a_decline("A dokumentumban nem szerepel az adószám.") is True
    assert _looks_like_a_decline("Nincs információ a nyitvatartásról.") is True


def test_looks_like_a_decline_false_for_valid_answers():
    assert (
        _looks_like_a_decline("The company was founded in 2018 in Budapest.") is False
    )
    assert _looks_like_a_decline("A projekt költségvetése 5 millió forint.") is False


def test_the_routers_own_plain_refusal_counts_as_a_decline():
    from query.decline_detection import looks_like_a_decline

    assert looks_like_a_decline(
        "No documents match the filter (court_decision documents: document_identifier "
        "contains 'PK-987654')."
    )
    assert not looks_like_a_decline("A bíróság a keresetet elutasította.")


def test_the_routers_other_plain_refusals_are_declines_too():
    """Every refusal the system itself words must read as one to the eval."""
    from query.decline_detection import looks_like_a_decline

    assert looks_like_a_decline(
        "This kind of question is not supported yet, so I will not guess at an answer. "
        "(five cases similar to the 27.P.20.339/2021/37 case)"
    )
    assert looks_like_a_decline(
        "I could not interpret this question well enough to answer it from the "
        "structured data, and I did not want to guess."
    )
    assert not looks_like_a_decline(
        "The supported claim was granted."
    )  # not just any "support"
