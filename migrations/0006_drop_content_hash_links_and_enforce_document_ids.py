"""Makes the numeric document id the only link, and drops the old hash links (the "contract" half).

Migration 0005 *added* ``documents.id`` and a ``document_id`` on the chunks, the
values and the statuses, and left the old ``content_hash`` links beside them so the
code could be moved over. The code now uses only the ids, so this removes what is
left and turns the new invariant into a database rule. After it:

* ``document_chunks.document_id``, ``document_meta.document_id`` and
  ``document_meta_status.document_id`` are ``NOT NULL`` (a chunk, a value or a status
  cannot exist without its document) and cascade on delete;
* the ``content_hash`` column of those three tables is gone (the chunks' was a
  generated copy of a field inside the metadata JSON, with its index), and the
  uniqueness rules use ``document_id``;
* ``documents`` has ``id`` as its primary key and ``content_hash`` as a ``UNIQUE``
  natural key (it identifies duplicates, nothing refers to it any more).

It is a migration (not an edit of 0004/0005) so that a *new* database, which runs
every migration in order, ends in the same clean state and never keeps a dropped
column. ``up`` stops with a clear message if any row still lacks its document id.
Dropping a column is instant; the primary-key change rebuilds a small index.
"""

from psycopg2.extensions import connection as PgConnection

from migrations.base import Migration

#: Tables that refer to ``documents.id``, with the name of their foreign key.
_REFERRERS = {
    "document_meta": "document_meta_document_id_fkey",
    "document_meta_status": "document_meta_status_document_id_fkey",
    "document_chunks": "document_chunks_document_id_fkey",
}


class DropContentHashLinksAndEnforceDocumentIds(Migration):
    def up(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            for table in _REFERRERS:
                cur.execute(f"SELECT count(*) FROM {table} WHERE document_id IS NULL;")
                row = cur.fetchone()
                missing = row[0] if row else 0
                if missing:
                    raise ValueError(
                        f"{missing} row(s) of {table} have no document_id; "
                        "link them first (migration 0005 back-fills them), then migrate again."
                    )

            # The values: unique per document instead of per hash, then no hash.
            cur.execute(
                "ALTER TABLE document_meta ALTER COLUMN document_id SET NOT NULL;"
            )
            cur.execute(
                "ALTER TABLE document_meta DROP CONSTRAINT document_meta_unique_row;"
            )
            cur.execute("""
                ALTER TABLE document_meta
                ADD CONSTRAINT document_meta_unique_row UNIQUE (document_id, key, ordinal);
            """)
            cur.execute("ALTER TABLE document_meta DROP COLUMN content_hash;")

            # The statuses: keyed by document and key.
            cur.execute(
                "ALTER TABLE document_meta_status ALTER COLUMN document_id SET NOT NULL;"
            )
            cur.execute(
                "ALTER TABLE document_meta_status DROP CONSTRAINT document_meta_status_pkey;"
            )
            cur.execute(
                "ALTER TABLE document_meta_status ADD PRIMARY KEY (document_id, key);"
            )
            cur.execute("DROP INDEX IF EXISTS document_meta_status_document_id_idx;")
            cur.execute("ALTER TABLE document_meta_status DROP COLUMN content_hash;")

            # The chunks: the generated copy of the hash and its index go.
            cur.execute(
                "ALTER TABLE document_chunks ALTER COLUMN document_id SET NOT NULL;"
            )
            cur.execute("DROP INDEX IF EXISTS document_chunks_content_hash_idx;")
            cur.execute("ALTER TABLE document_chunks DROP COLUMN content_hash;")

            # documents: id becomes the primary key, the hash a unique natural key.
            # The foreign keys that rest on the old unique id are re-created.
            for table, constraint in _REFERRERS.items():
                cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT {constraint};")
            cur.execute("ALTER TABLE documents DROP CONSTRAINT documents_id_key;")
            cur.execute("ALTER TABLE documents DROP CONSTRAINT documents_pkey;")
            cur.execute(
                "ALTER TABLE documents ADD CONSTRAINT documents_pkey PRIMARY KEY (id);"
            )
            cur.execute("""
                ALTER TABLE documents
                ADD CONSTRAINT documents_content_hash_key UNIQUE (content_hash);
            """)
            for table, constraint in _REFERRERS.items():
                cur.execute(f"""
                    ALTER TABLE {table}
                    ADD CONSTRAINT {constraint}
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE;
                """)

    def down(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            # Nothing to revert unless this migration's result is in place: the
            # tables exist and the values no longer carry a content_hash.
            cur.execute("""
                SELECT to_regclass('document_meta') IS NOT NULL
                   AND to_regclass('document_meta_status') IS NOT NULL
                   AND to_regclass('document_chunks') IS NOT NULL
                   AND to_regclass('documents') IS NOT NULL
                   AND NOT EXISTS (
                       SELECT 1 FROM information_schema.columns
                       WHERE table_name = 'document_meta' AND column_name = 'content_hash'
                   );
            """)
            row = cur.fetchone()
            if not (row and row[0]):
                return

            for table, constraint in _REFERRERS.items():
                cur.execute(
                    f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint};"
                )
            cur.execute(
                "ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_content_hash_key;"
            )
            cur.execute(
                "ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_pkey;"
            )
            cur.execute(
                "ALTER TABLE documents ADD CONSTRAINT documents_pkey PRIMARY KEY (content_hash);"
            )
            cur.execute(
                "ALTER TABLE documents ADD CONSTRAINT documents_id_key UNIQUE (id);"
            )

            cur.execute("""
                ALTER TABLE document_chunks
                ADD COLUMN content_hash TEXT
                GENERATED ALWAYS AS (metadata->>'content_hash') STORED;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_content_hash_idx
                ON document_chunks (content_hash);
            """)
            cur.execute(
                "ALTER TABLE document_chunks ALTER COLUMN document_id DROP NOT NULL;"
            )

            for table in ("document_meta", "document_meta_status"):
                cur.execute(f"ALTER TABLE {table} ADD COLUMN content_hash TEXT;")
                cur.execute(f"""
                    UPDATE {table} t SET content_hash = d.content_hash
                    FROM documents d WHERE d.id = t.document_id;
                """)
                cur.execute(
                    f"ALTER TABLE {table} ALTER COLUMN content_hash SET NOT NULL;"
                )
                cur.execute(f"""
                    ALTER TABLE {table}
                    ADD CONSTRAINT {table}_content_hash_fkey
                    FOREIGN KEY (content_hash) REFERENCES documents (content_hash)
                    ON DELETE CASCADE;
                """)
            # The keys go back to the hash *before* document_id may be nullable
            # again: on the statuses it is part of the primary key.
            cur.execute(
                "ALTER TABLE document_meta DROP CONSTRAINT document_meta_unique_row;"
            )
            cur.execute("""
                ALTER TABLE document_meta
                ADD CONSTRAINT document_meta_unique_row UNIQUE (content_hash, key, ordinal);
            """)
            cur.execute(
                "ALTER TABLE document_meta_status DROP CONSTRAINT document_meta_status_pkey;"
            )
            cur.execute(
                "ALTER TABLE document_meta_status ADD PRIMARY KEY (content_hash, key);"
            )
            for table in ("document_meta", "document_meta_status"):
                cur.execute(
                    f"ALTER TABLE {table} ALTER COLUMN document_id DROP NOT NULL;"
                )
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_meta_status_document_id_idx
                ON document_meta_status (document_id);
            """)

            for table, constraint in _REFERRERS.items():
                cur.execute(f"""
                    ALTER TABLE {table}
                    ADD CONSTRAINT {constraint}
                    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE;
                """)
