"""Scaffold a new, empty migration file under migrations/.

Usage:
    uv run python scripts/make_migration.py <snake_case_name>
    # or: make make-migration name=<snake_case_name>

Generates migrations/<NNNN>_<snake_case_name>.py, numbered one higher than
the highest existing migration, with an empty Migration subclass ready to
fill in.
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = PROJECT_ROOT / "migrations"

TEMPLATE = '''"""TODO: describe what this migration does and why."""

from psycopg2.extensions import connection as PgConnection

from migrations.base import Migration


class {class_name}(Migration):
    def up(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                -- TODO
            """)

    def down(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                -- TODO
            """)
'''


def next_migration_number(migrations_dir: Path) -> int:
    """Return one higher than the highest existing NNNN migration prefix."""
    numbers = []
    for f in migrations_dir.glob("*.py"):
        match = re.match(r"(\d+)_", f.stem)
        if match:
            numbers.append(int(match.group(1)))
    return (max(numbers) + 1) if numbers else 1


def to_class_name(snake_case_name: str) -> str:
    """Convert e.g. 'add_foo_column' to 'AddFooColumn'."""
    return "".join(word.capitalize() for word in snake_case_name.split("_"))


def make_migration(name: str, migrations_dir: Path = MIGRATIONS_DIR) -> Path:
    """Create a new numbered migration file and return its path."""
    number = next_migration_number(migrations_dir)
    filename = f"{number:04d}_{name}.py"
    path = migrations_dir / filename
    path.write_text(TEMPLATE.format(class_name=to_class_name(name)), encoding="utf-8")
    return path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/make_migration.py <snake_case_name>")
        print("   or: make make-migration name=<snake_case_name>")
        sys.exit(1)

    name = sys.argv[1]
    path = make_migration(name)
    print(f"Created {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
