"""Tests for scripts.meta_cli's pure helpers (no live DB required)."""

from collections.abc import Sequence
from pathlib import Path

from metadata.identifiers import identifier_matches
from scripts.meta_cli import _documents_named_in


class StubStore:
    """Resolves identifiers against a small table; exact matches only."""

    def __init__(self, documents: dict[int, list[str]]):
        self.documents = documents

    def documents_with_identifiers(self, wanted: Sequence[str]):
        return [
            (position, doc_id)
            for position, w in enumerate(wanted, start=1)
            for doc_id, values in sorted(self.documents.items())
            if any(identifier_matches(v, w) for v in values)
        ]

    def documents_matching_identifier_patterns(
        self, patterns: Sequence[str], ignoring_separators: bool = False
    ):
        return []


STORE = StubStore({1: ["4.P.20.111/2021/5"], 2: ["7.K.700.200/2022/2"]})


def test_the_listed_identifiers_become_document_ids(tmp_path: Path, capsys):
    path = tmp_path / "ids.txt"
    path.write_text("7.K.700.200/2022/2\n\n  4.P.20.111/2021/5  \n")

    assert _documents_named_in(STORE, path) == (1, 2)  # type: ignore[arg-type]
    assert "2 document(s) selected" in capsys.readouterr().out


def test_an_identifier_no_document_carries_is_reported_but_does_not_stop_the_rest(
    tmp_path: Path, capsys
):
    path = tmp_path / "ids.txt"
    path.write_text("4.P.20.111/2021/5\n9.P.99.999/2099/1\n")

    assert _documents_named_in(STORE, path) == (1,)  # type: ignore[arg-type]
    assert "9.P.99.999/2099/1" in capsys.readouterr().out


def test_a_file_that_names_no_document_extracts_nothing(tmp_path: Path, capsys):
    path = tmp_path / "ids.txt"
    path.write_text("9.P.99.999/2099/1\n")

    assert _documents_named_in(STORE, path) is None  # type: ignore[arg-type]
    assert "nothing to extract" in capsys.readouterr().out


def test_a_missing_file_extracts_nothing(tmp_path: Path, capsys):
    assert _documents_named_in(STORE, tmp_path / "nope.txt") is None  # type: ignore[arg-type]
    assert "No such file" in capsys.readouterr().out
