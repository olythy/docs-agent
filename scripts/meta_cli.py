"""Structured-metadata CLI for docs-agent.

Maintains the typed, per-document metadata layer that counting and listing
questions rely on (see docs/structured-metadata-design.md):
- Registering every ingested document in the ``documents`` table

Usage:
    uv run python scripts/meta_cli.py [command]

Commands:
    sync-documents    Make the documents table match the ingested chunks: register
                      (or refresh) one row per distinct content hash and remove rows
                      whose chunks are gone. Idempotent; safe to re-run after any
                      ingest, replace or flush.

Runs against whatever DATABASE_URL points at (the local Docker Postgres in dev),
so check it first, as with ``make db-migrate``.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from document_store import DocumentStore


def cmd_sync_documents() -> int:
    """Sync ``documents`` with the ingested chunks and report what changed."""
    store = DocumentStore()
    result = store.sync_from_chunks()
    print(
        f"documents synced: {result.upserted} registered/refreshed, "
        f"{result.removed} removed (no chunks left). "
        f"Total now: {store.count_documents()}."
    )
    return 0


COMMANDS = {"sync-documents": cmd_sync_documents}


def main() -> int:
    """CLI entry point."""
    if len(sys.argv) < 2 or sys.argv[1] in {"--help", "-h", "help"}:
        print((__doc__ or "").strip())
        return 0
    command = COMMANDS.get(sys.argv[1])
    if command is None:
        print(f"Unknown command: {sys.argv[1]!r}. Available: {', '.join(COMMANDS)}")
        return 2
    return command()


if __name__ == "__main__":
    sys.exit(main())
