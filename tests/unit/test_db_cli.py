"""Tests for scripts.db_cli's pure logic (no live DB required).

Anything that actually runs a migration's up()/down() against Postgres is
covered separately in tests/db/, gated on AGENT_ENV=test.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from migrations.base import Migration
from scripts.db_cli import (
    cmd_flush,
    compute_pending,
    discover_migration_files,
    last_batch_stems_reversed,
    load_migration,
    make_migration,
    next_migration_number,
    to_class_name,
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


def test_real_migration_0002_up_and_down_use_only_cursor():
    """Same sanity check as 0001, for the full-text-search migration."""
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "0002_add_fulltext_search.py"
    )
    instance = load_migration(path)
    fake_conn = MagicMock()

    instance.up(fake_conn)
    instance.down(fake_conn)

    assert fake_conn.cursor.called


def test_to_class_name():
    assert to_class_name("add_foo_column") == "AddFooColumn"
    assert to_class_name("create_users_table") == "CreateUsersTable"


def test_next_migration_number_empty(tmp_path: Path):
    assert next_migration_number(tmp_path) == 1


def test_next_migration_number_increment(tmp_path: Path):
    (tmp_path / "0001_first.py").write_text("")
    (tmp_path / "0002_second.py").write_text("")
    assert next_migration_number(tmp_path) == 3


def test_make_migration_creates_valid_file(tmp_path: Path):
    created = make_migration("add_bar_column", migrations_dir=tmp_path)
    assert created.name == "0001_add_bar_column.py"
    assert created.exists()
    content = created.read_text(encoding="utf-8")
    assert "class AddBarColumn(Migration):" in content
    assert "def up(" in content
    assert "def down(" in content


def test_cmd_flush_executes_truncate():
    fake_conn = MagicMock()
    cmd_flush(fake_conn)
    assert fake_conn.cursor.called
    assert fake_conn.commit.called
