"""Adds document types and numeric document ids (the "expand" half of a two-step change).

Two gaps in the structured-metadata layer (see docs/structured-metadata-design.md,
"Document types and numeric document ids"):

* The system does not know what *kind* of document a document is. This adds a
  ``document_types`` table (type, name, description, status) and a nullable
  ``documents.document_type`` that refers to it; ``meta_keys.doc_type`` now refers
  to it too. Types already named by an existing catalog are seeded as ``approved``
  with an empty description (the catalog loader fills it in).
* Chunks reach their document through a 64-character hash. This adds a numeric
  ``documents.id`` and a ``document_id`` on ``document_chunks``, ``document_meta``
  and ``document_meta_status``, back-filled from the hash, so a restricted search
  can be a sub-select on a small integer instead of a list of hashes.

Expand/contract: this migration only *adds*. The old ``content_hash`` links, keys
and the generated chunk column stay, so the code keeps working while it is moved to
the new columns; a later migration removes the old ones. ``document_id`` therefore
stays nullable for now (the ingest, which is not changed yet, does not set it).

Back-filling the chunks is an UPDATE of every row, which also writes new entries
into every index including the HNSW one, so it is slow on a large table: run it
before the corpus grows, and take a database dump first (``make db-dump``). A
chunk whose document has no ``documents`` row would stay without an id, so the
migration stops and says so (run ``make documents-sync`` first) rather than leave
holes behind.
"""

from psycopg2.extensions import connection as PgConnection

from migrations.base import Migration


class AddDocumentTypesAndDocumentIds(Migration):
    def up(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_types (
                    type        TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'proposed'
                        CHECK (status IN ('proposed', 'approved', 'retired')),
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            cur.execute("""
                INSERT INTO document_types (type, name, description, status)
                SELECT DISTINCT doc_type, doc_type, '', 'approved' FROM meta_keys
                ON CONFLICT (type) DO NOTHING;
            """)
            cur.execute("""
                ALTER TABLE meta_keys
                ADD CONSTRAINT meta_keys_doc_type_fkey
                FOREIGN KEY (doc_type) REFERENCES document_types (type);
            """)

            cur.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS id BIGINT GENERATED ALWAYS AS IDENTITY;
            """)
            cur.execute("""
                ALTER TABLE documents
                ADD CONSTRAINT documents_id_key UNIQUE (id);
            """)
            cur.execute("""
                ALTER TABLE documents
                ADD COLUMN IF NOT EXISTS document_type TEXT
                REFERENCES document_types (type);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_document_type_idx
                ON documents (document_type);
            """)

            for table in ("document_meta", "document_meta_status"):
                cur.execute(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS document_id BIGINT;"
                )
                cur.execute(f"""
                    UPDATE {table} t SET document_id = d.id
                    FROM documents d
                    WHERE d.content_hash = t.content_hash AND t.document_id IS NULL;
                """)
                cur.execute(f"""
                    ALTER TABLE {table}
                    ADD CONSTRAINT {table}_document_id_fkey
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                    ON DELETE CASCADE;
                """)
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table}_document_id_idx
                    ON {table} (document_id);
                """)

            cur.execute(
                "ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS document_id BIGINT;"
            )
            cur.execute("""
                UPDATE document_chunks c SET document_id = d.id
                FROM documents d
                WHERE d.content_hash = c.content_hash AND c.document_id IS NULL;
            """)
            cur.execute("""
                SELECT count(*) FROM document_chunks
                WHERE document_id IS NULL AND content_hash IS NOT NULL;
            """)
            row = cur.fetchone()
            orphans = row[0] if row else 0
            if orphans:
                raise ValueError(
                    f"{orphans} chunk(s) belong to a document with no 'documents' row; "
                    "run `make documents-sync` first, then migrate again."
                )
            cur.execute("""
                ALTER TABLE document_chunks
                ADD CONSTRAINT document_chunks_document_id_fkey
                FOREIGN KEY (document_id) REFERENCES documents (id)
                ON DELETE CASCADE NOT VALID;
            """)
            cur.execute("""
                ALTER TABLE document_chunks
                VALIDATE CONSTRAINT document_chunks_document_id_fkey;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx
                ON document_chunks (document_id);
            """)

    def down(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS document_chunks_document_id_idx;")
            cur.execute(
                "ALTER TABLE document_chunks "
                "DROP CONSTRAINT IF EXISTS document_chunks_document_id_fkey;"
            )
            cur.execute(
                "ALTER TABLE document_chunks DROP COLUMN IF EXISTS document_id;"
            )
            for table in ("document_meta_status", "document_meta"):
                cur.execute(f"DROP INDEX IF EXISTS {table}_document_id_idx;")
                cur.execute(
                    f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_document_id_fkey;"
                )
                cur.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS document_id;")
            cur.execute("DROP INDEX IF EXISTS documents_document_type_idx;")
            cur.execute("ALTER TABLE documents DROP COLUMN IF EXISTS document_type;")
            cur.execute(
                "ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_id_key;"
            )
            cur.execute("ALTER TABLE documents DROP COLUMN IF EXISTS id;")
            cur.execute(
                "ALTER TABLE meta_keys DROP CONSTRAINT IF EXISTS meta_keys_doc_type_fkey;"
            )
            cur.execute("DROP TABLE IF EXISTS document_types;")
