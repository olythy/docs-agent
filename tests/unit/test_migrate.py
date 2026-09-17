"""Tests for scripts.migrate's pure logic (no DB required).

Anything that actually runs a migration's up()/down() against Postgres is
covered separately in tests/db/, gated on TEST_DATABASE_URL.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from migrations.base import Migration
from scripts.migrate import (
    compute_pending,
    discover_migration_files,
    last_batch_stems_reversed,
    load_migration,
)


def test_discover_migration_files_excludes_base(tmp_path: Path):
    (tmp_path / "base.py").write_text("")
    (tmp_path / "0002_second.py").write_text("")
    (tmp_path / "0001_first.py").write_text("")

    files = discover_migration_files(tmp_path)

    assert [f.stem for f in files] == ["0001_first", "0002_second"]


def test_compute_pending_excludes_applied():
    files = [Path("0001_a.py"), Path("0002_b.py"), Path("0003_c.py")]
    pending = compute_pending(files, applied_stems={"0001_a", "0003_c"})
    assert [f.stem for f in pending] == ["0002_b"]


def test_last_batch_stems_reversed_empty():
    assert last_batch_stems_reversed([]) == []


def test_last_batch_stems_reversed_returns_only_last_batch_in_reverse():
    rows = [
        ("0001_a", 1),
        ("0002_b", 2),
        ("0003_c", 2),
    ]
    assert last_batch_stems_reversed(rows) == ["0003_c", "0002_b"]


def test_load_migration_loads_the_concrete_subclass(tmp_path: Path):
    migration_file = tmp_path / "0001_dummy.py"
    migration_file.write_text(
        "from migrations.base import Migration\n\n"
        "class Dummy(Migration):\n"
        "    def up(self, conn): pass\n"
        "    def down(self, conn): pass\n"
    )

    instance = load_migration(migration_file)

    assert isinstance(instance, Migration)


def test_load_migration_rejects_file_without_exactly_one_subclass(tmp_path: Path):
    migration_file = tmp_path / "0001_empty.py"
    migration_file.write_text("from migrations.base import Migration\n")

    with pytest.raises(ValueError, match="exactly one Migration subclass"):
        load_migration(migration_file)


def test_real_migration_0001_up_and_down_use_only_cursor():
    """Sanity check on the actual shipped migration, not a fake — no real DB.

    A MagicMock connection is enough here because up()/down() only ever call
    conn.cursor() and use it as a context manager — nothing about it depends
    on a real Postgres connection.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "0001_create_document_chunks_table.py"
    )
    instance = load_migration(path)
    fake_conn = MagicMock()

    instance.up(fake_conn)
    instance.down(fake_conn)

    assert fake_conn.cursor.called
