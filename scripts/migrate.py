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


def get_applied_migrations(conn: PgConnection) -> set[str]:
    """Return a set of already executed migration names."""
    with conn.cursor() as cur:
        cur.execute("SELECT migration FROM schema_migrations;")
        rows = cur.fetchall()
        return {row[0] for row in rows}


def get_next_batch_number(conn: PgConnection) -> int:
    """Calculate the next migration batch number."""
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(batch), 0) + 1 FROM schema_migrations;")
        return cur.fetchone()[0]


def run_migrations() -> None:
    """Find and execute pending SQL migrations in ascending order."""
    migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
    if not migrations_dir.exists():
        print(f"Migrations directory not found: {migrations_dir}")
        return

    sql_files = sorted(migrations_dir.glob("*.sql"))
    if not sql_files:
        print("No migration files found in migrations/ directory.")
        return

    conn = get_connection()
    try:
        ensure_migrations_table(conn)
        applied = get_applied_migrations(conn)

        pending = [f for f in sql_files if f.name not in applied]
        if not pending:
            print("Nothing to migrate. Database schema is up to date.")
            return

        batch = get_next_batch_number(conn)
        print(f"Running migrations (Batch {batch}):")

        for migration_file in pending:
            start_time = time.perf_counter()
            migration_name = migration_file.name
            print(f"  Migrating: {migration_name} ...", end="", flush=True)

            sql_content = migration_file.read_text(encoding="utf-8")

            try:
                with conn.cursor() as cur:
                    cur.execute(sql_content)
                    cur.execute(
                        """
                        INSERT INTO schema_migrations (migration, batch)
                        VALUES (%s, %s);
                        """,
                        (migration_name, batch),
                    )
                conn.commit()
                elapsed = time.perf_counter() - start_time
                print(f" DONE ({elapsed:.2f}s)")
            except (psycopg2.Error, OSError) as e:
                conn.rollback()
                print(" FAILED!")
                print(f"Error executing {migration_name}: {e}")
                sys.exit(1)

        print("Migration complete!")
    finally:
        conn.close()


if __name__ == "__main__":
    run_migrations()
