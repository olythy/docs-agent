"""Base class for Python-based migrations.

Each migration is a single ``.py`` file under ``migrations/`` defining one
``Migration`` subclass named exactly ``Migration``. ``up``/``down`` run raw
SQL directly against a psycopg2 connection — there is no ORM/schema-builder
layer, by design (this project has exactly one table).
"""

from abc import ABC, abstractmethod

from psycopg2.extensions import connection as PgConnection


class Migration(ABC):
    """A single reversible database migration."""

    @abstractmethod
    def up(self, conn: PgConnection) -> None:
        """Apply this migration."""

    @abstractmethod
    def down(self, conn: PgConnection) -> None:
        """Revert this migration.

        Must be safe to call even if ``up`` was never applied (e.g. use
        ``DROP TABLE IF EXISTS``), since ``migrate fresh`` calls ``down`` on
        every migration file unconditionally.
        """
