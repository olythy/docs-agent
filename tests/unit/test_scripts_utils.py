"""Tests for scripts.utils helpers (truncate, wrap, format_paragraphs, run_cmd, resolve_doc_path)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.utils import (
    PROJECT_ROOT,
    format_paragraphs,
    resolve_doc_path,
    run_cmd,
    truncate,
    wrap,
)


def test_project_root_is_valid():
    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_truncate_returns_text_unchanged_when_it_already_fits():
    assert truncate("short text", max_len=50) == "short text"


def test_truncate_shortens_on_a_word_boundary():
    text = "one two three four five six seven eight nine ten"
    result = truncate(text, max_len=20)

    assert len(result) <= 20
    assert result.endswith("...")
    words = result[: -len("...")].split()
    assert all(w in text.split() for w in words)


def test_truncate_collapses_internal_whitespace():
    assert truncate("a   b\nc", max_len=50) == "a b c"


def test_wrap_returns_single_line_unchanged_when_it_already_fits():
    assert wrap("short text", width=50) == "short text"


def test_wrap_splits_a_long_paragraph_across_multiple_lines_at_word_boundaries():
    text = "one two three four five six seven eight nine ten"
    result = wrap(text, width=20)

    lines = result.split("\n")
    assert len(lines) > 1
    assert all(len(line) <= 20 for line in lines)
    assert all(w in text.split() for line in lines for w in line.split())


def test_wrap_loses_no_words():
    text = "one two three four five six seven eight nine ten"
    result = wrap(text, width=20)
    assert result.replace("\n", " ").split() == text.split()


def test_format_paragraphs_indents_and_preserves_blank_lines():
    text = "First paragraph here.\n\nSecond paragraph has much longer text that needs to wrap properly."
    formatted = format_paragraphs(text, indent="  ", width=30)
    lines = formatted.split("\n")
    # First line starts with indent
    assert lines[0].startswith("  First")
    # Empty line preserved between paragraphs
    assert "" in lines
    # All non-empty lines start with indent
    for line in lines:
        if line:
            assert line.startswith("  ")
            assert len(line) <= 30


def test_run_cmd_success_and_failure():
    assert run_cmd(["true"]) == 0
    assert run_cmd(["false"]) == 1
    assert run_cmd(["non_existent_command_xyz_123"]) == 127


def test_run_cmd_silent_suppresses_output():
    # Silent flag should not crash or change exit code
    assert run_cmd(["true"], silent=True) == 0
    assert run_cmd(["false"], silent=True) == 1
    assert run_cmd(["non_existent_command_xyz_123"], silent=True) == 127


def test_resolve_doc_path_with_explicit_valid_file(tmp_path: Path):
    sample = tmp_path / "doc.txt"
    sample.write_text("hello")
    resolved = resolve_doc_path(str(sample))
    assert resolved == sample


def test_resolve_doc_path_with_env_setting(tmp_path: Path):
    sample = tmp_path / "env_doc.txt"
    sample.write_text("hello from env")
    mock_settings = patch("config.settings", create=True)
    with mock_settings as m:
        m.TEST_DOC_PATH = str(sample)
        resolved = resolve_doc_path(None)
        assert resolved == sample


def test_resolve_doc_path_missing_arg_and_setting():
    mock_settings = patch("config.settings", create=True)
    with mock_settings as m, pytest.raises(SystemExit):
        m.TEST_DOC_PATH = None
        resolve_doc_path(None)


def test_resolve_doc_path_nonexistent_file():
    with pytest.raises(SystemExit):
        resolve_doc_path("non_existent_file_abc_123.md")
