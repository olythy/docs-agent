"""Creates the structured-metadata tables and a queryable content_hash on chunks.

Top-k chunk retrieval cannot count or list ("how many judgments last
October"); that needs typed, per-document facts. This adds the storage for
them -- see docs/structured-metadata-design.md for the design and
docs/decisions.md for the measurements behind it.

    documents              identity of a document and the root of every cascade
                           (keyed by content_hash, because re-ingest deletes chunks
                           by hash, so a chunk-id foreign key would go stale)
    meta_keys              the key catalog: what keys exist, their type, description
                           and allowed values
    document_meta          the extracted values, each with its verbatim evidence
    document_meta_status   what is known about a (document, key) pair, including
                           the *absence* of a value, so a count can say "+K unknown"

It also adds a generated, indexed ``content_hash`` column to
``document_chunks`` (copied out of the metadata JSONB) so "search only inside
this set of documents" is an indexed lookup rather than a JSONB scan. The
column is generated, so no re-embedding and no change to how chunks are
written. Adding a stored generated column rewrites the table and rebuilds its
indexes (HNSW included), which takes a while on a large corpus.

Every identifier is English; only stored *values* keep the document's language.
"""

from psycopg2.extensions import connection as PgConnection

from migrations.base import Migration


class CreateStructuredMetadataTables(Migration):
    def up(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    content_hash TEXT PRIMARY KEY,
                    source_file  TEXT NOT NULL,
                    summary      TEXT,
                    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_source_file_idx
                ON documents (source_file);
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS meta_keys (
                    doc_type       TEXT NOT NULL,
                    key            TEXT NOT NULL,
                    value_type     TEXT NOT NULL
                        CHECK (value_type IN ('text', 'number', 'date', 'bool')),
                    description    TEXT NOT NULL,
                    example        TEXT,
                    allowed_values TEXT[],
                    multi_valued   BOOLEAN NOT NULL DEFAULT FALSE,
                    status         TEXT NOT NULL DEFAULT 'proposed'
                        CHECK (status IN ('proposed', 'approved', 'retired')),
                    version        INTEGER NOT NULL DEFAULT 1,
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (doc_type, key)
                );
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_meta (
                    id                   BIGSERIAL PRIMARY KEY,
                    content_hash         TEXT NOT NULL
                        REFERENCES documents (content_hash) ON DELETE CASCADE,
                    key                  TEXT NOT NULL,
                    key_version          INTEGER NOT NULL,
                    value_text           TEXT,
                    value_number         NUMERIC,
                    value_date           DATE,
                    value_bool           BOOLEAN,
                    unit                 TEXT,
                    ordinal              INTEGER NOT NULL DEFAULT 0,
                    qualifiers           JSONB NOT NULL DEFAULT '{}'::jsonb,
                    evidence             TEXT,
                    evidence_chunk_index INTEGER,
                    page                 INTEGER,
                    source               TEXT NOT NULL
                        CHECK (source IN ('deterministic', 'llm', 'sidecar')),
                    extracted_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
                    CONSTRAINT document_meta_one_value CHECK (
                        num_nonnulls(value_text, value_number, value_date, value_bool) = 1
                    ),
                    CONSTRAINT document_meta_unique_row
                        UNIQUE (content_hash, key, ordinal)
                );
            """)
            # One partial index per value type: the planner filters on a key and
            # a typed range, never across types.
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_meta_key_date_idx
                ON document_meta (key, value_date) WHERE value_date IS NOT NULL;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_meta_key_number_idx
                ON document_meta (key, value_number) WHERE value_number IS NOT NULL;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_meta_key_text_idx
                ON document_meta (key, value_text) WHERE value_text IS NOT NULL;
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_meta_status (
                    content_hash TEXT NOT NULL
                        REFERENCES documents (content_hash) ON DELETE CASCADE,
                    key          TEXT NOT NULL,
                    state        TEXT NOT NULL
                        CHECK (state IN (
                            'present', 'confirmed_absent', 'unverified', 'not_attempted'
                        )),
                    key_version  INTEGER NOT NULL,
                    attempted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (content_hash, key)
                );
            """)

            cur.execute("""
                ALTER TABLE document_chunks
                ADD COLUMN IF NOT EXISTS content_hash TEXT
                GENERATED ALWAYS AS (metadata->>'content_hash') STORED;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_content_hash_idx
                ON document_chunks (content_hash);
            """)

    def down(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS document_chunks_content_hash_idx;")
            cur.execute(
                "ALTER TABLE document_chunks DROP COLUMN IF EXISTS content_hash;"
            )
            cur.execute("DROP TABLE IF EXISTS document_meta_status;")
            cur.execute("DROP TABLE IF EXISTS document_meta;")
            cur.execute("DROP TABLE IF EXISTS meta_keys;")
            cur.execute("DROP TABLE IF EXISTS documents;")
