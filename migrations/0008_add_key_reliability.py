"""Adds ``meta_keys.reliability``: what has been measured about how far a key can be trusted.

An exact answer (a count, a list, a breakdown) is only as good as the values it counts, and
for a key whose values a model extracted that is a measured, imperfect quality. The note is
data about the key, kept beside its definition, so the answer can say it without the code
knowing any key by name. It is free text and optional.

It is *not* part of a key's definition: changing it must not make the values already
extracted look stale, so the catalog import updates it without bumping ``version``.

``down`` drops the column (it can be re-written from the catalog file).
"""

from psycopg2.extensions import connection as PgConnection

from migrations.base import Migration


class AddKeyReliability(Migration):
    def up(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE meta_keys ADD COLUMN IF NOT EXISTS reliability TEXT;
            """)

    def down(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE meta_keys DROP COLUMN IF EXISTS reliability;
            """)
