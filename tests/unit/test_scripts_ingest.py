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


def test_parse_args_with_delete():
    args = parse_args(["file.md", "--delete"])
    assert args.delete is True


def test_main_delete_by_hash(monkeypatch):
    fake_store = MagicMock()
    fake_store.delete_chunks_by_hash.return_value = 3
    monkeypatch.setattr("store.VectorStore", lambda: fake_store)

    target_hash = "a" * 64
    exit_code = main(["--delete", target_hash])

    assert exit_code == 0
    fake_store.delete_chunks_by_hash.assert_called_once_with(target_hash)


def test_main_delete_by_file(tmp_path, monkeypatch):
    f = tmp_path / "doc.md"
    f.write_text("# Test doc")

    fake_store = MagicMock()
    fake_store.delete_chunks_by_hash.return_value = 2
    monkeypatch.setattr("store.VectorStore", lambda: fake_store)

    exit_code = main(["--delete", str(f)])

    assert exit_code == 0
    assert fake_store.delete_chunks_by_hash.call_count == 1


def test_main_delete_by_missing_source_path(monkeypatch):
    fake_store = MagicMock()
    fake_store.delete_chunks_from_source.return_value = 1
    monkeypatch.setattr("store.VectorStore", lambda: fake_store)

    exit_code = main(["--delete", "nonexistent/doc.md"])

    assert exit_code == 0
    fake_store.delete_chunks_from_source.assert_called_once_with("nonexistent/doc.md")


def test_resolve_input_paths_reconstructs_spaces(tmp_path):
    from scripts.ingest import resolve_input_paths

    doc_dir = tmp_path / "My Folder"
    doc_dir.mkdir()
    doc_file = doc_dir / "Special File.md"
    doc_file.write_text("# Hello")

    # Simulate Make splitting the path on spaces
    split_tokens = [str(tmp_path / "My"), "Folder/Special", "File.md"]
    resolved = resolve_input_paths(split_tokens)

    assert len(resolved) == 1
    assert resolved[0] == str(doc_file)
