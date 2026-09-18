"""Patches `uv run mcp install`'s generated Claude Desktop config entry.

Purpose:
    ``mcp install`` (the MCP Python SDK's own CLI) writes a launch command
    designed for a standalone, single-file server with no dependencies of
    its own beyond ``mcp[cli]`` — ``uv run --with "mcp[cli]==X.Y.Z" mcp run
    <path>``, deliberately using no project/venv context, "so it works from
    any directory." docs-agent's ``mcp_server.py`` isn't standalone: it
    imports the whole project (``psycopg2``, ``openai``,
    ``sentence-transformers``, ...), none of which that command's ephemeral
    environment has. Confirmed empirically: Claude Desktop's real spawn (a
    foreign working directory, no inherited venv) fails with
    ``ModuleNotFoundError: No module named 'psycopg2'`` using the
    SDK-generated command as-is.

    The fix — ``uv run --project <docs-agent dir> mcp_server.py`` instead —
    resolves against this project's own environment regardless of the
    caller's cwd, and is itself confirmed working the same way (a real MCP
    client connecting to the server spawned exactly as Claude Desktop
    would: unrelated cwd, no inherited virtualenv). ``mcp install`` has no
    flag for this (only ``mcp dev`` has ``--with-editable``), so this
    script runs right after it and rewrites just the ``args`` field —
    ``env`` (loaded from ``.env`` by ``mcp install -f .env``) is left as-is.

Usage:
    uv run mcp install mcp_server.py --name "docs-agent" -f .env
    uv run python scripts/fix_mcp_install.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_NAME = "docs-agent"

# macOS only — Claude Desktop's config also lives at %APPDATA%\Claude\
# claude_desktop_config.json on Windows, but this project's only ever run
# on macOS so far; add that path here if that ever changes.
CONFIG_PATH = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"


def patch_args(config: dict, project_root: Path, server_name: str = SERVER_NAME) -> dict:
    """Rewrite ``server_name``'s ``args`` in-place to use ``--project``.

    Pure aside from mutating ``config`` (the same shape ``json.loads`` on
    the real config file produces) — split out from I/O so this is
    directly unit-testable without touching the real filesystem.

    Args:
        config: The parsed Claude Desktop config (``mcpServers`` at top
            level).
        project_root: Absolute path to the docs-agent project directory.
        server_name: The ``mcpServers`` entry to patch.

    Returns:
        ``config``, mutated.

    Raises:
        KeyError: If ``server_name`` isn't present under ``mcpServers``.
    """
    entry = config["mcpServers"][server_name]
    entry["args"] = [
        "run",
        "--project",
        str(project_root),
        str(project_root / "mcp_server.py"),
    ]
    return config


def main() -> None:
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found — run `make mcp-install` first.")
        sys.exit(1)

    config = json.loads(CONFIG_PATH.read_text())
    try:
        patch_args(config, PROJECT_ROOT)
    except KeyError:
        print(f"ERROR: no '{SERVER_NAME}' entry in {CONFIG_PATH} — run `make mcp-install` first.")
        sys.exit(1)

    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"Patched '{SERVER_NAME}' entry in {CONFIG_PATH} to use --project.")
    print("Fully quit and reopen Claude Desktop for this to take effect.")


if __name__ == "__main__":
    main()
