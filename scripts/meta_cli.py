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
    sync-documents    Remove documents that have no chunks. The ingest registers a
                      document before it saves its chunks, so an ingest that failed in
                      between leaves a bare row; its values and statuses go with it.
                      Idempotent; safe to re-run at any time.

    load-catalog <file>
                      Validate a catalog JSON file and write it to meta_keys. A new key
                      starts at version 1; a key whose definition changed gets its
                      version bumped (so values extracted under the old definition are
                      recognisably stale); a key that already matches is left alone.
                      Example: corpus/data/meta_catalog.json.

    extract-meta [--doc-type T] [--limit N [--seed S]]
                      Extract every pending approved key of up to N documents of each
                      document type (or of type T only); a type's keys are asked only of
                      its own documents, and documents with no type are skipped (default:
                      all that still need it). Idempotent and resumable: it only does
                      (document, key) pairs with no status at the key's current version,
                      so an interrupted run continues where it stopped and a changed key
                      definition refreshes only that key. Costs one LLM call per
                      document for the LLM-sourced keys. Start with --limit 50. Without --seed the first
                      N by file name are taken (files are named by court, so that is one
                      court's sample); with --seed S, a random, reproducible sample.
    keys [--doc-type T]
                      List the catalog's keys with their status. A key the extractor
                      proposed stays 'proposed' (unusable in queries) until approved.
    set-key-status <doc_type> <key> <approved|retired|proposed>
                      Approve a proposed key, or retire one (e.g. a duplicate of an
                      existing key). Takes effect on the next extract-meta run.
    classify-documents [--limit N [--seed S]]
                      Give each document that has no type one: an LLM reads the document's
                      summary and opening and picks a known type (approved or already
                      proposed) or proposes a new one, with a verbatim quote as evidence
                      that the code verifies. A proposed type is stored as 'proposed' and
                      cannot be used in queries until you approve it. A document whose
                      answer is unusable or whose quote is not in the text stays
                      unclassified (and is retried next run). One LLM call per document;
                      start with --limit 20 --seed 1.
    types
                      List the document types with their status and how many documents
                      each has, and how many documents have no type yet.
    set-type-status <type> <approved|retired|proposed>
                      Approve a proposed type, or retire one.
    assign-type <type> [--yes]
                      Manual shortcut: give EVERY document that has no type this (approved)
                      type, without an LLM call. Meant for a corpus that has one kind of
                      document. Prints how many documents it would change; changes them
                      only with --yes.
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
from metadata.classification_runner import ClassificationRunner
from metadata.classifier import LLMTypeClassifier
from metadata.evidence import EvidenceSelector
from metadata.runner import MetaExtractionRunner
from metadata.sources import ChunkMetadataSource, LLMMetaSource
from models import KeyStatus, TypeStatus
from store import VectorStore


def cmd_sync_documents() -> int:
    """Remove documents that have no chunks (what a failed ingest leaves behind)."""
    store = DocumentStore()
    removed = store.remove_documents_without_chunks()
    print(
        f"{removed} document(s) removed (no chunks). "
        f"Total now: {store.count_documents()}."
    )
    return 0


def cmd_load_catalog(args: list[str]) -> int:
    """Validate a catalog file and import it into ``meta_keys``."""
    if len(args) != 1:
        print("Usage: meta_cli.py load-catalog <file>")
        return 2
    try:
        catalog = load_catalog_seed(Path(args[0]))
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    result = KeyCatalog(DocumentStore()).import_catalog(catalog)
    print(
        f"catalog loaded ({', '.join(t.type for t in catalog.types)}): "
        f"types {result.types_added} added, {result.types_updated} updated; "
        f"keys {result.added} added, {result.revised} revised, "
        f"{result.unchanged} unchanged."
    )
    return 0


#: Catalog keys that this project's ingestion already extracts deterministically
#: into the chunk metadata (chunker.extract_document_date). Configuration of the
#: *court-decision corpus*, not of the generic core: another corpus passes none.
_DETERMINISTIC_FIELDS = {
    "decision_date": "document_date",
    "document_identifier": "document_identifiers",
}

_RED = "\033[31;1m"
_GREEN = "\033[32m"
_RESET = "\033[0m"


def _parse(args: list[str], description: str, with_limit: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="meta_cli.py", description=description)
    parser.add_argument(
        "--doc-type",
        default=None,
        help="only this document type (default: every approved type that has keys)",
    )
    if with_limit:
        parser.add_argument(
            "--limit", type=int, default=None, help="process at most N documents"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="with --limit: a random sample fixed by this seed (default: the first N by file name)",
        )
    return parser.parse_args(args)


def _types_with_keys(store: DocumentStore, only: str | None) -> list[str]:
    """The document types a command applies to: ``only``, or every approved type with keys."""
    if only is not None:
        return [only]
    return [
        t.type
        for t in store.list_types(TypeStatus.APPROVED)
        if store.list_keys(t.type, KeyStatus.APPROVED)
    ]


def cmd_extract_meta(args: list[str]) -> int:
    """Run the extraction for the pending keys of the documents that need it.

    Each document type's keys are extracted from the documents *of that type*;
    documents that have no type yet are not touched (classify them first).
    """
    options = _parse(
        args, "Extract the catalog's keys from the documents.", with_limit=True
    )
    store = DocumentStore()
    doc_types = _types_with_keys(store, options.doc_type)
    if not doc_types:
        print("No document type has approved keys. Run load-catalog first.")
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
    for doc_type in doc_types:
        if not store.list_keys(doc_type):
            print(f"No catalog for document type {doc_type!r}. Run load-catalog first.")
            return 1
        print(f"\n== {doc_type}")
        report = runner.run(doc_type, limit=options.limit, seed=options.seed)
        print(
            f"extracted for {report.documents} document(s): {report.present} values present, "
            f"{report.confirmed_absent} confirmed absent, {report.unverified} unverified, "
            f"{report.failed} failed (will be retried), {report.proposed_keys} key(s) proposed."
        )
    untyped = store.count_by_type().get(None, 0)
    if untyped:
        print(
            f"\n{_RED}{untyped} document(s) have no type and were not extracted;{_RESET} "
            "run classify-documents (or assign-type) first."
        )
    return cmd_coverage(["--doc-type", options.doc_type] if options.doc_type else [])


def cmd_coverage(args: list[str]) -> int:
    """Print, per approved key, how complete the extracted metadata is (per document type)."""
    options = _parse(args, "Report metadata coverage per key.", with_limit=False)
    store = DocumentStore()
    doc_types = _types_with_keys(store, options.doc_type)
    if not doc_types:
        print("No document type has approved keys.")
        return 1
    for doc_type in doc_types:
        keys = store.list_keys(doc_type, KeyStatus.APPROVED)
        if not keys:
            print(f"No approved keys for document type {doc_type!r}.")
            return 1
        print(
            f"\nmetadata coverage for {doc_type} ({store.count_documents(doc_type)} documents)"
        )
        print(
            f"{'key':<22}{'present':>9}{'absent':>9}{'unverified':>12}{'not tried':>11}{'unknown':>9}"
        )
        for cov in store.coverage(keys, doc_type):
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


def cmd_keys(args: list[str]) -> int:
    """List the catalog's keys with status and version (every type, or one)."""
    options = _parse(args, "List the catalog's keys.", with_limit=False)
    store = DocumentStore()
    doc_types = (
        [options.doc_type] if options.doc_type else [t.type for t in store.list_types()]
    )
    rows = [(t, key) for t in doc_types for key in store.list_keys(t)]
    if not rows:
        print("No keys.")
        return 1
    print(f"{'type':<18}{'key':<24}{'value':<8}{'status':<10}{'v':>3}  description")
    for doc_type, key in rows:
        print(
            f"{doc_type:<18}{key.key:<24}{key.value_type.value:<8}{key.status.value:<10}"
            f"{key.version:>3}  {key.description[:60]}"
        )
    return 0


def cmd_set_key_status(args: list[str]) -> int:
    """Approve, retire or re-propose one catalog key."""
    if len(args) != 3:
        print(
            "Usage: meta_cli.py set-key-status <doc_type> <key> <approved|retired|proposed>"
        )
        return 2
    doc_type, key, status = args
    try:
        new_status = KeyStatus(status)
    except ValueError:
        print(f"Unknown status {status!r}; use approved, retired or proposed.")
        return 2
    if not DocumentStore().set_key_status(doc_type, key, new_status):
        print(f"No key {key!r} for doc_type {doc_type!r}.")
        return 1
    print(f"{doc_type}.{key} is now {new_status.value}.")
    return 0


def cmd_classify_documents(args: list[str]) -> int:
    """Classify the documents that have no type yet."""
    options = _parse(args, "Classify documents by type.", with_limit=True)
    store = DocumentStore()
    if not store.list_types():
        print("No document types. Run load-catalog first.")
        return 1

    def progress(done: int, total: int) -> None:
        print(f"  ... {done}/{total} documents", flush=True)

    runner = ClassificationRunner(
        documents=store,
        chunks=VectorStore(),
        classifier=LLMTypeClassifier(get_answer_driver()),
        on_progress=progress,
    )
    report = runner.run(limit=options.limit, seed=options.seed)
    print(
        f"\nclassified {report.documents} document(s): {report.classified} with an "
        f"approved type, {report.proposed} with a proposed type "
        f"({report.new_types} new type(s) proposed), {report.failed} left unclassified."
    )
    for reason, count in report.reasons:
        print(f"  {count} x {reason}")
    return cmd_types([])


def cmd_types(args: list[str]) -> int:
    """List the document types with their status and document counts."""
    counts = DocumentStore().count_by_type()
    types = DocumentStore().list_types()
    print(f"{'type':<24}{'status':<10}{'documents':>10}  name")
    for doc_type in types:
        print(
            f"{doc_type.type:<24}{doc_type.status.value:<10}"
            f"{counts.get(doc_type.type, 0):>10}  {doc_type.name}"
        )
    unclassified = counts.get(None, 0)
    shown = f"{_RED}{unclassified}{_RESET}" if unclassified else str(unclassified)
    print(f"\nno type yet: {shown} document(s)")
    return 0


def cmd_set_type_status(args: list[str]) -> int:
    """Approve, retire or re-propose one document type."""
    if len(args) != 2:
        print("Usage: meta_cli.py set-type-status <type> <approved|retired|proposed>")
        return 2
    type_name, status = args
    try:
        new_status = TypeStatus(status)
    except ValueError:
        print(f"Unknown status {status!r}; use approved, retired or proposed.")
        return 2
    if not DocumentStore().set_type_status(type_name, new_status):
        print(f"No document type {type_name!r}.")
        return 1
    print(f"{type_name} is now {new_status.value}.")
    return 0


def cmd_assign_type(args: list[str]) -> int:
    """Give every document without a type one approved type (a manual shortcut)."""
    confirmed = "--yes" in args
    names = [a for a in args if a != "--yes"]
    if len(names) != 1:
        print("Usage: meta_cli.py assign-type <type> [--yes]")
        return 2
    store = DocumentStore()
    doc_type = store.get_type(names[0])
    if doc_type is None or doc_type.status is not TypeStatus.APPROVED:
        print(
            f"{names[0]!r} is not an approved document type (see: meta_cli.py types)."
        )
        return 1
    waiting = store.count_by_type().get(None, 0)
    if not confirmed:
        print(
            f"{waiting} document(s) have no type; this would give them all "
            f"{doc_type.type!r}. Run again with --yes to do it."
        )
        return 0
    changed = store.assign_type_to_unclassified(doc_type.type)
    print(f"{changed} document(s) are now {doc_type.type!r}.")
    return 0


COMMANDS = {
    "sync-documents": lambda args: cmd_sync_documents(),
    "load-catalog": cmd_load_catalog,
    "extract-meta": cmd_extract_meta,
    "coverage": cmd_coverage,
    "keys": cmd_keys,
    "set-key-status": cmd_set_key_status,
    "classify-documents": cmd_classify_documents,
    "types": cmd_types,
    "set-type-status": cmd_set_type_status,
    "assign-type": cmd_assign_type,
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
