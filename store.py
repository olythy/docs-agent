"""Vector store: all document_chunks persistence (save + search).

Owns the only SQL that touches the document_chunks table. The orchestrator
functions (ingestion.ingest.add_document, query.retrieval.query_knowledge_base)
never build or execute SQL themselves — they call VectorStore.save()/.search().

Key exports:
    VectorStore  -- save() and search() against document_chunks.
"""

import json

from db import get_connection


def _to_pgvector_literal(embedding: list[float]) -> str:
    """Format a float vector as a pgvector literal, e.g. ``'[0.1,0.2,...]'``."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


class VectorStore:
    """Persistence layer for the document_chunks table.

    Opens its own connection per call, matching this project's existing
    short-lived-connection style (no connection pooling yet).
    """

    def save(self, chunks: list[dict], embeddings: list[list[float]]) -> int:
        """Insert chunk rows into document_chunks.

        Args:
            chunks: Chunk dicts from :func:`ingestion.chunker.chunk_pages`.
            embeddings: Parallel list of float vectors, one per chunk.

        Returns:
            The number of rows inserted.
        """
        insert_sql = """
            INSERT INTO document_chunks (content, metadata, embedding)
            VALUES (%s, %s, %s);
        """
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    cur.execute(
                        insert_sql,
                        (
                            chunk["content"],
                            json.dumps(chunk["metadata"]),
                            _to_pgvector_literal(embedding),
                        ),
                    )
            conn.commit()
            return len(chunks)
        finally:
            conn.close()

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[dict]:
        """Return the most similar chunks to ``query_embedding``, above ``min_score``.

        Uses pgvector's ``<=>`` operator, which computes cosine *distance*
        (0 = identical, 2 = opposite). Converted to similarity with
        ``1 - distance`` so that higher is better, matching the ``min_score``
        threshold convention.

        Args:
            query_embedding: The embedded question vector.
            top_k: Maximum number of results to consider before filtering.
            min_score: Minimum cosine similarity (0-1). Chunks below this are
                dropped.

        Returns:
            A list of chunk dicts ordered by descending similarity, each
            containing ``content``, ``metadata``, and ``score``.
        """
        vector_literal = _to_pgvector_literal(query_embedding)
        sql = """
            SELECT
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (vector_literal, vector_literal, top_k))
                rows = cur.fetchall()
        finally:
            conn.close()

        results = []
        for content, metadata, score in rows:
            if score < min_score:
                continue
            # metadata arrives as a dict when psycopg2 uses jsonb; ensure it is one
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append({"content": content, "metadata": metadata, "score": score})
        return results

    def get_embedding_dimension(self) -> int | None:
        """Read the declared dimension of the document_chunks.embedding column.

        pgvector encodes a ``vector(N)`` column's dimension directly in
        ``atttypmod``, with no offset (unlike e.g. ``varchar``'s typmod) —
        confirmed empirically against a real pgvector column.
        ``to_regclass`` returns NULL (not an error) for a table that
        doesn't exist yet, so this returns None cleanly if migrations
        haven't run.

        Returns:
            The column's declared dimension, or ``None`` if document_chunks
            doesn't exist yet.
        """
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT atttypmod FROM pg_attribute
                    WHERE attrelid = to_regclass('document_chunks')
                      AND attname = 'embedding'
                      AND NOT attisdropped;
                    """
                )
                row = cur.fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def assert_dimension_matches(self, expected_dimension: int) -> None:
        """Raise a clear error if ``expected_dimension`` disagrees with the DB column.

        A fresh database always gets a matching column (the dimension is
        embedded directly into the CREATE TABLE in
        ``migrations/0001_create_document_chunks_table.py``), so this only
        fires when someone switches ``EMBEDDING_DRIVER``/``EMBEDDING_DIMENSION``
        on a database that already has data from a different embedding model.
        Widening the column alone would not fix that case — embeddings from
        different models aren't comparable regardless of vector size, so the
        real fix is a new migration plus re-embedding every document.

        Args:
            expected_dimension: The active embedding driver's output dimension.

        Raises:
            RuntimeError: If document_chunks exists with a different dimension.
        """
        actual = self.get_embedding_dimension()
        if actual is not None and actual != expected_dimension:
            raise RuntimeError(
                f"Embedding dimension mismatch: the active embedding driver "
                f"produces {expected_dimension}-dim vectors, but "
                f"document_chunks.embedding is declared vector({actual}). "
                "Switching embedding drivers on an existing database requires "
                "re-embedding every document, not just widening the column — "
                "add a new migration (or run `migrate fresh` if you don't need "
                "the existing data) once you've decided which dimension to use."
            )
