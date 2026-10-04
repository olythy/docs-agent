"""Structured-metadata CLI for docs-agent.

Maintains the typed, per-document metadata layer that counting and listing
questions rely on (see docs/structured-metadata-design.md):
- Registering every ingested document in the ``documents`` table
- Loading a corpus's key catalog (a JSON file of keys, types, descriptions) into
  ``meta_keys``
- Extracting the catalog's keys from every document (with a verbatim quote per
  LLM-extracted value) and reporting how complete the result is

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

    extract-meta [--doc-type T] [--limit N]
                      Extract every pending approved key of up to N documents (default:
                      all that still need it). Idempotent and resumable: it only does
                      (document, key) pairs with no status at the key's current version,
                      so an interrupted run continues where it stopped and a changed key
                      definition refreshes only that key. Costs one LLM call per
                      document for the LLM-sourced keys. Start with --limit 50.
    coverage [--doc-type T]
                      For every approved key: how many documents have a verified value,
                      are confirmed absent, are unverified or were never attempted. A
                      count over a key must say "+K unknown"; this shows K, in red when
                      it is not zero.

Runs against whatever DATABASE_URL points at (the local Docker Postgres in dev),
so check it first, as with ``make db-migrate``.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from document_store import DocumentStore
from drivers.llm import get_answer_driver
from drivers.reranker import get_reranker_driver
from metadata.catalog import KeyCatalog, load_catalog_seed
from metadata.evidence import EvidenceSelector
from metadata.runner import MetaExtractionRunner
from metadata.sources import ChunkMetadataSource, LLMMetaSource
from models import KeyStatus
from store import VectorStore


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


#: Catalog keys that this project's ingestion already extracts deterministically
#: into the chunk metadata (chunker.extract_document_date). Configuration of the
#: *court-decision corpus*, not of the generic core: another corpus passes none.
_DETERMINISTIC_FIELDS = {"decision_date": "document_date"}

_RED = "\033[31;1m"
_GREEN = "\033[32m"
_RESET = "\033[0m"


def _parse(args: list[str], description: str, with_limit: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="meta_cli.py", description=description)
    parser.add_argument("--doc-type", default="court_decision", help="catalog to use")
    if with_limit:
        parser.add_argument(
            "--limit", type=int, default=None, help="process at most N documents"
        )
    return parser.parse_args(args)


def cmd_extract_meta(args: list[str]) -> int:
    """Run the extraction for the pending keys of the documents that need it."""
    options = _parse(
        args, "Extract the catalog's keys from the documents.", with_limit=True
    )
    store = DocumentStore()
    if not store.list_keys(options.doc_type):
        print(f"No catalog for doc_type {options.doc_type!r}. Run load-catalog first.")
        return 1

    def progress(done: int, total: int) -> None:
        print(f"  ... {done}/{total} documents", flush=True)

    runner = MetaExtractionRunner(
        documents=store,
        chunks=VectorStore(),
        selector=EvidenceSelector(get_reranker_driver()),
        sources=[
            ChunkMetadataSource(_DETERMINISTIC_FIELDS),
            LLMMetaSource(get_answer_driver()),
        ],
        on_progress=progress,
    )
    report = runner.run(options.doc_type, limit=options.limit)
    print(
        f"\nextracted for {report.documents} document(s): {report.present} values present, "
        f"{report.confirmed_absent} confirmed absent, {report.unverified} unverified, "
        f"{report.failed} failed (will be retried), {report.proposed_keys} key(s) proposed."
    )
    return cmd_coverage(["--doc-type", options.doc_type])


def cmd_coverage(args: list[str]) -> int:
    """Print, per approved key, how complete the extracted metadata is."""
    options = _parse(args, "Report metadata coverage per key.", with_limit=False)
    store = DocumentStore()
    keys = store.list_keys(options.doc_type, KeyStatus.APPROVED)
    if not keys:
        print(f"No approved keys for doc_type {options.doc_type!r}.")
        return 1
    print(
        f"\nmetadata coverage for {options.doc_type} ({store.count_documents()} documents)"
    )
    print(
        f"{'key':<22}{'present':>9}{'absent':>9}{'unverified':>12}{'not tried':>11}{'unknown':>9}"
    )
    for cov in store.coverage(keys):
        colour = _GREEN if cov.unknown == 0 else _RED
        print(
            f"{cov.key:<22}{cov.present:>9}{cov.confirmed_absent:>9}{cov.unverified:>12}"
            f"{cov.not_attempted:>11}{colour}{cov.unknown:>9}{_RESET}"
        )
    print(
        "\nunknown = unverified + not tried: a count over that key must say '+K unknown'. "
        "'absent' means the document was examined and does not state it."
    )
    return 0


COMMANDS = {
    "sync-documents": lambda args: cmd_sync_documents(),
    "load-catalog": cmd_load_catalog,
    "extract-meta": cmd_extract_meta,
    "coverage": cmd_coverage,
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
