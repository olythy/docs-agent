"""Structured-metadata CLI for docs-agent.

Maintains the typed, per-document metadata layer that counting and listing
questions rely on (see docs/structured-metadata-design.md):
- Registering every ingested document in the ``documents`` table
- Loading a corpus's key catalog (a JSON file of keys, types, descriptions) into
  ``meta_keys``

Usage:
    uv run python scripts/meta_cli.py [command]

Commands:
    sync-documents    Make the documents table match the ingested chunks: register
                      (or refresh) one row per distinct content hash and remove rows
                      whose chunks are gone. Idempotent; safe to re-run after any
                      ingest, replace or flush.

    load-catalog <file>
                      Validate a catalog JSON file and write it to meta_keys. A new key
                      starts at version 1; a key whose definition changed gets its
                      version bumped (so values extracted under the old definition are
                      recognisably stale); a key that already matches is left alone.
                      Example: corpus/data/meta_catalog.json.

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
from metadata.catalog import KeyCatalog, load_catalog_seed


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


def cmd_load_catalog(args: list[str]) -> int:
    """Validate a catalog file and import it into ``meta_keys``."""
    if len(args) != 1:
        print("Usage: meta_cli.py load-catalog <file>")
        return 2
    try:
        keys = load_catalog_seed(Path(args[0]))
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    result = KeyCatalog(DocumentStore()).import_seed(keys)
    print(
        f"catalog loaded ({keys[0].doc_type if keys else 'empty'}): "
        f"{result.added} added, {result.revised} revised, {result.unchanged} unchanged."
    )
    return 0


COMMANDS = {
    "sync-documents": lambda args: cmd_sync_documents(),
    "load-catalog": cmd_load_catalog,
}


def main() -> int:
    """CLI entry point."""
    if len(sys.argv) < 2 or sys.argv[1] in {"--help", "-h", "help"}:
        print((__doc__ or "").strip())
        return 0
    command = COMMANDS.get(sys.argv[1])
    if command is None:
        print(f"Unknown command: {sys.argv[1]!r}. Available: {', '.join(COMMANDS)}")
        return 2
    return command(sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
