"""Creates the document_chunks table for pgvector-based RAG storage.

The embedding column's dimension comes directly from
``settings.EMBEDDING_DIMENSION`` at migration time, so a fresh database
matches whichever EMBEDDING_DRIVER is configured when this migration first
runs. This only affects the *first* run: once applied, the column's
dimension is fixed until a new migration changes it (switching drivers on an
existing database requires re-embedding all documents, not just a wider
column — embeddings from different models are not comparable regardless of
size).
"""

from psycopg2.extensions import connection as PgConnection

from config import settings
from migrations.base import Migration


class CreateDocumentChunksTable(Migration):
    def up(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding vector({settings.EMBEDDING_DIMENSION}),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops);
                """
            )

    def down(self, conn: PgConnection) -> None:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS document_chunks;")
