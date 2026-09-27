"""CLI for ingesting documents and directories into the knowledge base.

Accepts one or more file or directory paths. For files, ingests them individually
via :func:`ingestion.ingest.add_document`. For directories, batch-ingests them via
:func:`ingestion.ingest.add_directory`.

Usage:
    # Single document:
    uv run python scripts/ingest.py path/to/doc.pdf

    # Multiple documents:
    uv run python scripts/ingest.py doc1.pdf doc2.md doc3.pdf

    # Entire directory:
    uv run python scripts/ingest.py path/to/docs/

    # Directory with custom extension filter and force re-indexing:
    uv run python scripts/ingest.py path/to/docs/ --ext .md --force
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.ingest import add_directory, add_document


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents or directories into the docs-agent knowledge base.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more document file paths or directory paths to ingest.",
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
    return parser.parse_args(args)


logger = logging.getLogger(__name__)


def main(cli_args: list[str] | None = None) -> int:
    """Run the CLI ingestion workflow."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(cli_args)

    allowed_exts: list[str] | None = None
    if args.extensions:
        allowed_exts = [e.strip() for e in args.extensions.split(",") if e.strip()]

    has_errors = False

    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            logger.error("[ingest] Error: Path not found: %s", raw_path)
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
                logger.error("[ingest] Error processing directory '%s': %s", path, exc)
                has_errors = True
        elif path.is_file():
            try:
                add_document(path, force=args.force)
            except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
                logger.error("[ingest] Error ingesting file '%s': %s", path, exc)
                has_errors = True
        else:
            logger.error(
                "[ingest] Error: '%s' is neither a file nor a directory.", path
            )
            has_errors = True

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
