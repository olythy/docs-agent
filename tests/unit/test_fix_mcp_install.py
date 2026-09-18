"""Tests for scripts.fix_mcp_install.patch_args (pure logic, no real config file)."""

from pathlib import Path

import pytest

from scripts.fix_mcp_install import patch_args


def _config_with_entry(**overrides) -> dict:
    entry = {
        "command": "/Users/x/.local/bin/uv",
        "args": ["run", "--frozen", "--with", "mcp[cli]==2.2.0", "mcp", "run", "/path/mcp_server.py"],
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
