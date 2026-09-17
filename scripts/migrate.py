"""Migration runner with Laravel-artisan-style subcommands.

Migrations are Python files under ``migrations/`` (one ``Migration``
subclass per file, see ``migrations/base.py``), tracked by filename stem
(without extension) in a ``schema_migrations`` table in the same database
``DATABASE_URL`` points at.

Usage:
    uv run python scripts/migrate.py [up|status|install|rollback|reset|refresh|fresh]

    (no argument defaults to "up" — this is what `make db-migrate` runs)

Subcommands:
    up        Run all pending migrations (default).
    status    Show which migrations are applied vs. pending.
    install   Create the schema_migrations tracking table, nothing else.
    rollback  Revert the most recently applied *batch* of migrations.
    reset     Revert every applied migration, in reverse order.
    fresh     Revert every migration file (regardless of tracked state),
              drop the tracking table, then run everything from scratch.
    refresh   Shorthand for reset + up.
"""

import importlib.util
import inspect
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
from psycopg2.extensions import connection as PgConnection

from config import settings
from migrations.base import Migration

MIGRATIONS_DIR = PROJECT_ROOT / "migrations"


def get_connection() -> PgConnection:
    """Establish and return a database connection."""
    if not settings.DATABASE_URL:
        print("ERROR: DATABASE_URL is not set in environment or .env file.")
        sys.exit(1)
    return psycopg2.connect(settings.DATABASE_URL)


def ensure_migrations_table(conn: PgConnection) -> None:
    """Ensure that the migrations tracking table exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id SERIAL PRIMARY KEY,
                migration VARCHAR(255) UNIQUE NOT NULL,
                batch INT NOT NULL,
                executed_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
    conn.commit()


def discover_migration_files(migrations_dir: Path = MIGRATIONS_DIR) -> list[Path]:
    """Return every migration file in ``migrations_dir``, sorted by name.

    Excludes ``base.py`` (the ``Migration`` ABC, not a migration itself).
    """
    return sorted(
        f for f in migrations_dir.glob("*.py") if f.stem not in {"base", "__init__"}
    )


def load_migration_class(path: Path) -> type[Migration]:
    """Dynamically import a migration file and return its Migration subclass.

    Migration filenames (e.g. ``0001_create_x.py``) aren't valid dotted
    import names, so this loads by file path via ``importlib`` instead of a
    normal ``import`` statement. Each file must define exactly one concrete
    subclass of :class:`migrations.base.Migration`.

    Raises:
        ValueError: If the file defines zero or more than one such class.
    """
    spec = importlib.util.spec_from_file_location(f"migration_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Migration) and obj is not Migration
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{path.name} must define exactly one Migration subclass, "
            f"found {len(candidates)}."
        )
    return candidates[0]


def load_migration(path: Path) -> Migration:
    """Load and instantiate the Migration subclass defined in ``path``."""
    return load_migration_class(path)()


def get_applied_migrations(conn: PgConnection) -> list[tuple[str, int]]:
    """Return (migration, batch) for every applied migration, in application order."""
    with conn.cursor() as cur:
        cur.execute("SELECT migration, batch FROM schema_migrations ORDER BY id;")
        return list(cur.fetchall())


def get_next_batch_number(conn: PgConnection) -> int:
    """Calculate the next migration batch number."""
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(batch), 0) + 1 FROM schema_migrations;")
        return cur.fetchone()[0]


def compute_pending(files: list[Path], applied_stems: set[str]) -> list[Path]:
    """Return the subset of ``files`` not yet in ``applied_stems``, in order."""
    return [f for f in files if f.stem not in applied_stems]


def last_batch_stems_reversed(rows: list[tuple[str, int]]) -> list[str]:
    """Return the stems of the most recently applied batch, in reverse order.

    ``rows`` is (migration, batch) in application order, as returned by
    :func:`get_applied_migrations`. Reverse order matches how rollback must
    undo migrations: last applied, first reverted.
    """
    if not rows:
        return []
    last_batch = max(batch for _, batch in rows)
    return [stem for stem, batch in reversed(rows) if batch == last_batch]


def _find_file_by_stem(files: list[Path], stem: str) -> Path:
    for f in files:
        if f.stem == stem:
            return f
    raise FileNotFoundError(
        f"Migration file for '{stem}' not found in {MIGRATIONS_DIR} — "
        "it was applied before but its file is now missing."
    )


def _apply_up(conn: PgConnection, path: Path, batch: int) -> None:
    start = time.perf_counter()
    print(f"  Migrating: {path.stem} ...", end="", flush=True)
    try:
        migration = load_migration(path)
        migration.up(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO schema_migrations (migration, batch) VALUES (%s, %s);",
                (path.stem, batch),
            )
        conn.commit()
        print(f" DONE ({time.perf_counter() - start:.2f}s)")
    except (psycopg2.Error, OSError, ValueError) as e:
        conn.rollback()
        print(" FAILED!")
        print(f"Error applying {path.stem}: {e}")
        sys.exit(1)


def _apply_down(conn: PgConnection, path: Path) -> None:
    start = time.perf_counter()
    print(f"  Rolling back: {path.stem} ...", end="", flush=True)
    try:
        migration = load_migration(path)
        migration.down(conn)
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM schema_migrations WHERE migration = %s;", (path.stem,)
            )
        conn.commit()
        print(f" DONE ({time.perf_counter() - start:.2f}s)")
    except (psycopg2.Error, OSError, ValueError) as e:
        conn.rollback()
        print(" FAILED!")
        print(f"Error rolling back {path.stem}: {e}")
        sys.exit(1)


def cmd_up(conn: PgConnection) -> None:
    """Run all pending migrations."""
    files = discover_migration_files()
    applied = {stem for stem, _ in get_applied_migrations(conn)}
    pending = compute_pending(files, applied)

    if not pending:
        print("Nothing to migrate. Database schema is up to date.")
        return

    batch = get_next_batch_number(conn)
    print(f"Running migrations (Batch {batch}):")
    for path in pending:
        _apply_up(conn, path, batch)
    print("Migration complete!")


def cmd_status(conn: PgConnection) -> None:
    """Show which migrations are applied vs. pending."""
    files = discover_migration_files()
    applied = dict(get_applied_migrations(conn))

    if not files:
        print("No migration files found.")
        return

    for path in files:
        if path.stem in applied:
            print(f"  [applied, batch {applied[path.stem]}]  {path.stem}")
        else:
            print(f"  [pending]              {path.stem}")


def cmd_install(_conn: PgConnection) -> None:
    """Create the schema_migrations tracking table, nothing else.

    Takes an unused ``conn`` parameter so every command in ``COMMANDS``
    shares the same signature for uniform dispatch — ``main()`` already
    calls ``ensure_migrations_table`` before dispatching, which is this
    command's entire job.
    """
    print("Migration table ready (schema_migrations).")


def cmd_rollback(conn: PgConnection) -> None:
    """Revert the most recently applied batch of migrations."""
    files = discover_migration_files()
    rows = get_applied_migrations(conn)
    stems = last_batch_stems_reversed(rows)

    if not stems:
        print("Nothing to rollback.")
        return

    print(f"Rolling back batch {max(b for _, b in rows)}:")
    for stem in stems:
        _apply_down(conn, _find_file_by_stem(files, stem))
    print("Rollback complete!")


def cmd_reset(conn: PgConnection) -> None:
    """Revert every applied migration, in reverse order."""
    files = discover_migration_files()
    rows = get_applied_migrations(conn)

    if not rows:
        print("Nothing to reset.")
        return

    print("Reverting all migrations:")
    for stem, _ in reversed(rows):
        _apply_down(conn, _find_file_by_stem(files, stem))
    print("Reset complete!")


def cmd_refresh(conn: PgConnection) -> None:
    """Revert every applied migration, then re-run all migrations."""
    cmd_reset(conn)
    cmd_up(conn)


def cmd_fresh(conn: PgConnection) -> None:
    """Revert every migration file, drop the tracking table, then re-run everything.

    Unlike reset, this ignores what schema_migrations claims is applied and
    calls `down()` on every migration file unconditionally — each `down()`
    must be safe to call even if `up()` was never applied (see
    migrations/base.py). This guarantees a truly clean slate regardless of
    tracking-table drift.
    """
    files = discover_migration_files()
    print("Dropping everything:")
    for path in reversed(files):
        _apply_down(conn, path)

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS schema_migrations;")
    conn.commit()

    ensure_migrations_table(conn)
    cmd_up(conn)


COMMANDS = {
    "up": cmd_up,
    "status": cmd_status,
    "install": cmd_install,
    "rollback": cmd_rollback,
    "reset": cmd_reset,
    "refresh": cmd_refresh,
    "fresh": cmd_fresh,
}


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "up"
    if command not in COMMANDS:
        print(f"Unknown command: '{command}'")
        print(f"Available commands: {', '.join(COMMANDS)}")
        sys.exit(1)

    conn = get_connection()
    try:
        ensure_migrations_table(conn)
        COMMANDS[command](conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
