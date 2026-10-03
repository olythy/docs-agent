"""Switches content_tsv's text search config from 'simple' to 'hungarian'.

Revisits 0002_add_fulltext_search.py's deliberate choice of 'simple' (no
stemming, to tokenize mixed-language content evenly). Confirmed live (see
docs/decisions.md's 2026-10-03 entry) that 'simple' misses Hungarian
inflected word forms a question and its matching chunk use differently,
and that Postgres's built-in 'hungarian' (snowball stemmer) config
measurably improves real golden-question keyword-search rank -- on 5
test questions, average rank improved from 269 to 214, and one case
crossed the practical top-20 candidate-pool threshold entirely (334 -> 5).

Exact identifier matching (case numbers, invoice numbers, ...) is
unaffected either way: store.VectorStore.search_by_identifier() runs a
literal ILIKE match against chunk content, a separate code path from this
tsvector column, so this change carries no risk to that mechanism
regardless of document language.

Postgres doesn't support altering a generated column's expression in
place, so this drops and re-adds content_tsv (and its index) with the new
config, same shape as 0002's original up().
"""

from psycopg2.extensions import connection as PgConnection

from migrations.base import Migration


class HungarianFulltextSearchConfig(Migration):
    def up(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS document_chunks_content_tsv_idx;")
            cur.execute(
                "ALTER TABLE document_chunks DROP COLUMN IF EXISTS content_tsv;"
            )
            cur.execute("""
                ALTER TABLE document_chunks
                ADD COLUMN content_tsv tsvector
                GENERATED ALWAYS AS (to_tsvector('hungarian', content)) STORED;
            """)
            cur.execute("""
                CREATE INDEX document_chunks_content_tsv_idx
                ON document_chunks
                USING GIN (content_tsv);
            """)

    def down(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS document_chunks_content_tsv_idx;")
            cur.execute(
                "ALTER TABLE document_chunks DROP COLUMN IF EXISTS content_tsv;"
            )
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
