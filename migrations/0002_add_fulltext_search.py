"""Adds full-text search support to document_chunks, for hybrid retrieval.

A generated ``tsvector`` column (kept in sync automatically by Postgres,
never written to directly) plus a GIN index lets ``store.py`` run a
keyword-based search (``websearch_to_tsquery`` + ``ts_rank``) alongside the
existing pgvector cosine-similarity search — the two are fused with
Reciprocal Rank Fusion in ``query/retrieval.py``.

Uses the ``simple`` text search configuration (tokenizes and lowercases,
but does *not* stem or apply language-specific stopword lists) rather than
``english``/``hungarian``: the corpus is a mix of both languages, and a
single language-specific configuration would only tokenize one of them
well.
"""

from psycopg2.extensions import connection as PgConnection

from migrations.base import Migration


class AddFulltextSearch(Migration):
    def up(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE document_chunks
                ADD COLUMN content_tsv tsvector
                GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;
            """)
            cur.execute("""
                CREATE INDEX document_chunks_content_tsv_idx
                ON document_chunks
                USING GIN (content_tsv);
            """)

    def down(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS document_chunks_content_tsv_idx;")
            cur.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS content_tsv;")
