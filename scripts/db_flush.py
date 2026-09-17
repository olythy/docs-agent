"""Flush all rows from document_chunks — a fast reset for local development.

Does not drop the table and does not re-run migrations; use
``scripts/migrate.py`` separately if the schema itself needs to change.

WARNING: this truncates whatever database DATABASE_URL currently points to.
There is no separate "local" database for this project yet — double-check
your .env before running this against anything you care about.

Usage:
    uv run python scripts/db_flush.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db import get_connection


def flush_document_chunks() -> None:
    """Truncate the document_chunks table, resetting its identity sequence."""
    try:
        conn = get_connection()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE document_chunks RESTART IDENTITY;")
        conn.commit()
        print("document_chunks flushed.")
    finally:
        conn.close()


if __name__ == "__main__":
    flush_document_chunks()
