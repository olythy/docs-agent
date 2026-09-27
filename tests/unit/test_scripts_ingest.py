"""Tests for scripts.ingest CLI entry point."""

from pathlib import Path
from unittest.mock import MagicMock

from scripts.ingest import main, parse_args


def test_parse_args_defaults():
    args = parse_args(["file1.pdf", "dir/"])
    assert args.paths == ["file1.pdf", "dir/"]
    assert args.force is False
    assert args.no_recursive is False
    assert args.extensions is None


def test_parse_args_with_flags():
    args = parse_args(["dir/", "--force", "--no-recursive", "--ext", ".md,.pdf"])
    assert args.paths == ["dir/"]
    assert args.force is True
    assert args.no_recursive is True
    assert args.extensions == ".md,.pdf"


def test_main_calls_add_document_for_file(tmp_path, monkeypatch):
    f = tmp_path / "doc.md"
    f.write_text("# Hello")

    fake_add_document = MagicMock()
    monkeypatch.setattr("scripts.ingest.add_document", fake_add_document)

    exit_code = main([str(f), "--force"])

    assert exit_code == 0
    fake_add_document.assert_called_once_with(Path(f), force=True)


def test_main_calls_add_directory_for_directory(tmp_path, monkeypatch):
    d = tmp_path / "folder"
    d.mkdir()

    fake_add_directory = MagicMock(return_value={"failed": []})
    monkeypatch.setattr("scripts.ingest.add_directory", fake_add_directory)

    exit_code = main([str(d), "--ext", ".md,.markdown", "--no-recursive"])

    assert exit_code == 0
    fake_add_directory.assert_called_once_with(
        Path(d),
        recursive=False,
        force=False,
        allowed_extensions=[".md", ".markdown"],
    )


def test_main_returns_error_code_on_missing_path():
    exit_code = main(["/nonexistent/path/for/sure/12345.pdf"])
    assert exit_code == 1


def test_main_returns_error_code_when_directory_has_failures(tmp_path, monkeypatch):
    d = tmp_path / "folder"
    d.mkdir()

    fake_add_directory = MagicMock(
        return_value={"failed": [{"file": "bad.pdf", "error": "corrupt"}]}
    )
    monkeypatch.setattr("scripts.ingest.add_directory", fake_add_directory)

    exit_code = main([str(d)])
    assert exit_code == 1
