"""Unit tests for scripts.eval_cli (pure logic, fixtures discovery, decline detection)."""

from pathlib import Path

from scripts.eval_cli import (
    _looks_like_a_decline,
    discover_eval_fixtures,
)


def test_discover_eval_fixtures_filters_json_and_hidden(tmp_path: Path):
    (tmp_path / "sample.pdf").write_text("dummy")
    (tmp_path / "notes.md").write_text("dummy")
    (tmp_path / "guide.txt").write_text("dummy")
    (tmp_path / "questions.json").write_text("{}")
    (tmp_path / ".DS_Store").write_text("")
    (tmp_path / "photo.png").write_text("")

    fixtures = discover_eval_fixtures(data_dir=tmp_path)
    names = [f.name for f in fixtures]

    assert names == ["guide.txt", "notes.md", "sample.pdf"]


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
