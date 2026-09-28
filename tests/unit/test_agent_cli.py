"""Tests for scripts.agent_cli (ingest commands, path resolution, MCP configuration patching)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.agent_cli import (
    cmd_ingest,
    patch_args,
    resolve_input_paths,
)
from scripts.agent_cli import (
    parse_ingest_args as parse_args,
)

# --- MCP Registration & Config Patching Tests ---


def _config_with_entry(**overrides) -> dict:
    entry = {
        "command": "/Users/x/.local/bin/uv",
        "args": [
            "run",
            "--frozen",
            "--with",
            "mcp[cli]==2.2.0",
            "mcp",
            "run",
            "/path/mcp_server.py",
        ],
        "env": {"DATABASE_URL": "postgresql://..."},
    }
    entry.update(overrides)
    return {"mcpServers": {"docs-agent": entry}}


def test_patch_args_rewrites_to_use_project_flag():
    config = _config_with_entry()

    patch_args(config, project_root=Path("/Users/x/Sites/docs-agent"))

    assert config["mcpServers"]["docs-agent"]["args"] == [
        "run",
        "--project",
        "/Users/x/Sites/docs-agent",
        "/Users/x/Sites/docs-agent/mcp_server.py",
    ]


def test_patch_args_leaves_env_and_command_untouched():
    config = _config_with_entry()
    original_env = config["mcpServers"]["docs-agent"]["env"]
    original_command = config["mcpServers"]["docs-agent"]["command"]

    patch_args(config, project_root=Path("/Users/x/Sites/docs-agent"))

    assert config["mcpServers"]["docs-agent"]["env"] == original_env
    assert config["mcpServers"]["docs-agent"]["command"] == original_command


def test_patch_args_raises_when_entry_missing():
    config = {"mcpServers": {}}

    with pytest.raises(KeyError):
        patch_args(config, project_root=Path("/Users/x/Sites/docs-agent"))


def test_patch_args_uses_given_server_name():
    config = {"mcpServers": {"other-name": {"args": []}}}

    patch_args(config, project_root=Path("/x"), server_name="other-name")

    assert config["mcpServers"]["other-name"]["args"][:2] == ["run", "--project"]


# --- Document Ingestion & Deletion Tests ---


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
    monkeypatch.setattr("scripts.agent_cli.add_document", fake_add_document)

    exit_code = cmd_ingest([str(f), "--force"])

    assert exit_code == 0
    fake_add_document.assert_called_once_with(Path(f), force=True)


def test_main_calls_add_directory_for_directory(tmp_path, monkeypatch):
    d = tmp_path / "folder"
    d.mkdir()

    fake_add_directory = MagicMock(return_value={"failed": []})
    monkeypatch.setattr("scripts.agent_cli.add_directory", fake_add_directory)

    exit_code = cmd_ingest([str(d), "--ext", ".md,.markdown", "--no-recursive"])

    assert exit_code == 0
    fake_add_directory.assert_called_once_with(
        Path(d),
        recursive=False,
        force=False,
        allowed_extensions=[".md", ".markdown"],
    )


def test_main_returns_error_code_on_missing_path():
    exit_code = cmd_ingest(["/nonexistent/path/for/sure/12345.pdf"])
    assert exit_code == 1


def test_main_returns_error_code_when_directory_has_failures(tmp_path, monkeypatch):
    d = tmp_path / "folder"
    d.mkdir()

    fake_add_directory = MagicMock(
        return_value={"failed": [{"file": "bad.pdf", "error": "corrupt"}]}
    )
    monkeypatch.setattr("scripts.agent_cli.add_directory", fake_add_directory)

    exit_code = cmd_ingest([str(d)])
    assert exit_code == 1


def test_parse_args_with_delete():
    args = parse_args(["file.md", "--delete"])
    assert args.delete is True


def test_main_delete_by_hash(monkeypatch):
    fake_store = MagicMock()
    fake_store.delete_chunks_by_hash.return_value = 3
    monkeypatch.setattr("store.VectorStore", lambda: fake_store)

    target_hash = "a" * 64
    exit_code = cmd_ingest(["--delete", target_hash])

    assert exit_code == 0
    fake_store.delete_chunks_by_hash.assert_called_once_with(target_hash)


def test_main_delete_by_file(tmp_path, monkeypatch):
    f = tmp_path / "doc.md"
    f.write_text("# Test doc")

    fake_store = MagicMock()
    fake_store.delete_chunks_by_hash.return_value = 2
    monkeypatch.setattr("store.VectorStore", lambda: fake_store)

    exit_code = cmd_ingest(["--delete", str(f)])

    assert exit_code == 0
    assert fake_store.delete_chunks_by_hash.call_count == 1


def test_main_delete_by_missing_source_path(monkeypatch):
    fake_store = MagicMock()
    fake_store.delete_chunks_from_source.return_value = 1
    monkeypatch.setattr("store.VectorStore", lambda: fake_store)

    exit_code = cmd_ingest(["--delete", "nonexistent/doc.md"])

    assert exit_code == 0
    fake_store.delete_chunks_from_source.assert_called_once_with("nonexistent/doc.md")


def test_resolve_input_paths_reconstructs_spaces(tmp_path):
    doc_dir = tmp_path / "My Folder"
    doc_dir.mkdir()
    doc_file = doc_dir / "Special File.md"
    doc_file.write_text("# Hello")

    # Simulate Make splitting the path on spaces
    split_tokens = [str(tmp_path / "My"), "Folder/Special", "File.md"]
    resolved = resolve_input_paths(split_tokens)

    assert len(resolved) == 1
    assert resolved[0] == str(doc_file)
