"""Agent and runtime CLI for docs-agent.

Consolidates user-facing agent interactions, document ingestion,
and Model Context Protocol (MCP) server lifecycle:
- Question answering (one-shot RAG query)
- Interactive conversational terminal chat
- Document and directory ingestion or deletion
- MCP server inspection and automated Claude Desktop registration

Usage:
    uv run python scripts/agent_cli.py [command] [args]

Commands:
    query <question>       Ask a question against the knowledge base (full RAG pipeline).
    chat                   Start the interactive conversational REPL terminal.
    ingest <paths...>      Ingest file(s) or directories into document_chunks.
                           Options:
                             --force, -f         Replace existing document chunks.
                             --ext .md,.pdf      Filter extensions when ingesting directories.
                             --no-recursive      Do not recurse into subdirectories.
                             --delete, -d        Delete chunks matching given path(s) or hashes.
    mcp-dev                Launch mcp_server.py under the MCP Inspector for local testing.
    mcp-install            Register mcp_server.py with Claude Desktop and auto-patch launch config.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.ingest import add_directory, add_document
from scripts.utils import run_cmd

SERVER_NAME = "docs-agent"
CLAUDE_CONFIG_PATH = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Claude"
    / "claude_desktop_config.json"
)


# --- MCP Server Management ---


def patch_args(
    config: dict, project_root: Path, server_name: str = SERVER_NAME
) -> dict:
    """Rewrite ``server_name``'s ``args`` in-place to use ``--project``.

    Pure function split out from I/O for direct unit testing.
    """
    entry = config["mcpServers"][server_name]
    entry["args"] = [
        "run",
        "--project",
        str(project_root),
        str(project_root / "mcp_server.py"),
    ]
    return config


def patch_claude_desktop_config(
    config_path: Path = CLAUDE_CONFIG_PATH, project_root: Path = PROJECT_ROOT
) -> int:
    """Patch Claude Desktop's config file to use project-scoped execution."""
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        return 1

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        patch_args(config, project_root)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"Patched '{SERVER_NAME}' entry in {config_path} to use --project.")
        print("Fully quit and reopen Claude Desktop for changes to take effect.")
        return 0
    except KeyError:
        print(f"ERROR: No '{SERVER_NAME}' entry in {config_path}.")
        return 1
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR updating config: {e}")
        return 1


def cmd_mcp_dev() -> int:
    """Launch the MCP server in developer inspector mode."""
    print("Launching MCP inspector for mcp_server.py...")
    return run_cmd(["uv", "run", "mcp", "dev", "mcp_server.py"])


def cmd_mcp_install() -> int:
    """Register mcp_server.py with Claude Desktop and patch its launch arguments."""
    print("Registering mcp_server.py with Claude Desktop...")
    code = run_cmd(
        [
            "uv",
            "run",
            "mcp",
            "install",
            "mcp_server.py",
            "--name",
            SERVER_NAME,
            "-f",
            ".env",
        ]
    )
    if code != 0:
        print("ERROR: Failed to register MCP server.")
        return code

    print("Patching configuration to resolve project dependencies...")
    return patch_claude_desktop_config()


# --- Document Ingestion & Deletion ---


def parse_ingest_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ingestion."""
    parser = argparse.ArgumentParser(
        prog="agent_cli.py ingest",
        description="Ingest documents or directories into the docs-agent knowledge base.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more document file paths or directory paths to ingest or delete.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Replace existing chunks if already present in the knowledge base.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive searching when ingesting directories.",
    )
    parser.add_argument(
        "--ext",
        "--extensions",
        dest="extensions",
        type=str,
        default=None,
        help="Comma-separated file extensions to filter directories (e.g. '.md' or '.pdf,.md').",
    )
    parser.add_argument(
        "--delete",
        "-d",
        action="store_true",
        help="Delete document chunks matching the given paths or content hashes instead of ingesting.",
    )
    return parser.parse_args(args)


def resolve_input_paths(raw_paths: list[str]) -> list[str]:
    """Resolve and reconstruct input paths, handling shell expansion and space-split tokens."""
    resolved: list[str] = []
    i = 0
    while i < len(raw_paths):
        current = raw_paths[i]
        expanded = Path(current).expanduser()

        if len(current) == 64 and all(c in "0123456789abcdefABCDEF" for c in current):
            resolved.append(current)
            i += 1
            continue

        if expanded.exists():
            resolved.append(str(expanded))
            i += 1
            continue

        accumulated = current
        found = False
        j = i + 1
        while j < len(raw_paths):
            accumulated += " " + raw_paths[j]
            test_path = Path(accumulated).expanduser()
            if test_path.exists():
                resolved.append(str(test_path))
                i = j + 1
                found = True
                break
            j += 1

        if not found:
            resolved.append(str(expanded))
            i += 1

    return resolved


def cmd_ingest(argv: list[str]) -> int:
    """Ingest or delete documents from CLI arguments."""
    args = parse_ingest_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    resolved_paths = resolve_input_paths(args.paths)
    allowed_exts = (
        [ext.strip() for ext in args.extensions.split(",") if ext.strip()]
        if args.extensions
        else None
    )

    if args.delete:
        from ingestion.hash import compute_file_hash
        from store import VectorStore

        store = VectorStore()
        for target in resolved_paths:
            target_path = Path(target)
            if len(target) == 64 and all(c in "0123456789abcdefABCDEF" for c in target):
                deleted = store.delete_chunks_by_hash(target.lower())
                print(f"[ingest] Deleted {deleted} chunk(s) for hash {target[:8]}.")
            elif target_path.exists() and target_path.is_file():
                content_hash = compute_file_hash(target_path)
                deleted = store.delete_chunks_by_hash(content_hash)
                print(
                    f"[ingest] Deleted {deleted} chunk(s) for file '{target}' (hash {content_hash[:8]})."
                )
            else:
                deleted = store.delete_chunks_from_source(target)
                print(f"[ingest] Deleted {deleted} chunk(s) for source '{target}'.")
        return 0

    has_errors = False
    for raw_path in resolved_paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"[ingest] Error: Path not found: {raw_path}")
            has_errors = True
            continue

        if path.is_dir():
            try:
                summary = add_directory(
                    path,
                    recursive=not args.no_recursive,
                    force=args.force,
                    allowed_extensions=allowed_exts,
                )
                if summary["failed"]:
                    has_errors = True
            except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
                print(f"[ingest] Error processing directory '{path}': {exc}")
                has_errors = True
        elif path.is_file():
            try:
                add_document(path, force=args.force)
            except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
                print(f"[ingest] Error ingesting file '{path}': {exc}")
                has_errors = True
        else:
            print(f"[ingest] Error: '{path}' is neither a file nor a directory.")
            has_errors = True

    return 1 if has_errors else 0


parse_args = parse_ingest_args


# --- Query & Chat Interaction ---


def cmd_query(question: str) -> int:
    """Run a single question through the RAG pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from query.retrieval import query_knowledge_base

    answer = query_knowledge_base(question)
    print("\n--- Answer ---")
    print(answer)
    print("--------------\n")
    return 0


def cmd_chat() -> int:
    """Start the interactive agent console session."""
    from agent import run_interactive

    run_interactive()
    return 0


# --- Dispatcher ---


def print_help() -> None:
    """Print command usage and descriptions."""
    print(__doc__.strip())


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for agent and runtime operations."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"--help", "-h", "help"}:
        print_help()
        return 0

    command = argv[0]
    sub_args = argv[1:]

    if command == "query":
        if not sub_args:
            print("Usage: uv run python scripts/agent_cli.py query <question>")
            return 1
        return cmd_query(" ".join(sub_args))

    if command == "chat":
        return cmd_chat()

    if command == "ingest":
        if not sub_args:
            print(
                "Usage: uv run python scripts/agent_cli.py ingest <paths...> [options]"
            )
            return 1
        return cmd_ingest(sub_args)

    if command == "mcp-dev":
        return cmd_mcp_dev()

    if command == "mcp-install":
        return cmd_mcp_install()

    print(f"Unknown command: '{command}'")
    print("Available commands: query, chat, ingest, mcp-dev, mcp-install")
    return 1


if __name__ == "__main__":
    sys.exit(main())
