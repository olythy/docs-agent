"""Lets a metadata key have the value type ``identifier``.

An identifier (a case number, an invoice number, ...) is stored exactly like text, in
``document_meta.value_text``, but the system compares it in a normalised form (see
``metadata/identifiers.py``): ignoring case, spaces and a trailing full stop, and
accepting a written suffix. The type tells it which keys those are, without any
corpus-specific key name in the code.

Only the ``CHECK`` on ``meta_keys.value_type`` has to widen; no data moves, so the
values already extracted for a key stay valid when the key is retyped.

``down`` turns every ``identifier`` key back into ``text`` first (the old constraint
would refuse the rows), so it is safe with data in place.
"""

from psycopg2.extensions import connection as PgConnection

from migrations.base import Migration

#: PostgreSQL's name for the unnamed CHECK that migration 0004 created.
_CONSTRAINT = "meta_keys_value_type_check"


class AllowIdentifierValueType(Migration):
    def up(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute(
                f"ALTER TABLE meta_keys DROP CONSTRAINT IF EXISTS {_CONSTRAINT};"
            )
            cur.execute(f"""
                ALTER TABLE meta_keys
                ADD CONSTRAINT {_CONSTRAINT}
                CHECK (value_type IN ('text', 'number', 'date', 'bool', 'identifier'));
            """)

    def down(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('meta_keys') IS NOT NULL;")
            row = cur.fetchone()
            if not (row and row[0]):
                return  # nothing to revert: the table was never created
            cur.execute(
                "UPDATE meta_keys SET value_type = 'text' WHERE value_type = 'identifier';"
            )
            cur.execute(
                f"ALTER TABLE meta_keys DROP CONSTRAINT IF EXISTS {_CONSTRAINT};"
            )
            cur.execute(f"""
                ALTER TABLE meta_keys
                ADD CONSTRAINT {_CONSTRAINT}
                CHECK (value_type IN ('text', 'number', 'date', 'bool'));
            """)
