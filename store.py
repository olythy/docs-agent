"""Vector store: all document_chunks persistence (save + search).

Owns the only SQL that touches the document_chunks table. The orchestrator
functions (ingestion.ingest.add_document, query.retrieval.query_knowledge_base)
never build or execute SQL themselves — they call VectorStore.save()/.search()/
.search_fulltext().

Key exports:
    VectorStore  -- save(), search() (vector), and search_fulltext()
                    (keyword) against document_chunks.
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
            containing ``id`` (the row's primary key — lets callers like
            :func:`query.hybrid.reciprocal_rank_fusion` identify the *same*
            chunk across a separate keyword-search result set, since two
            different rows could coincidentally share identical text),
            ``content``, ``metadata``, and ``score``.
        """
        vector_literal = _to_pgvector_literal(query_embedding)
        sql = """
            SELECT
                id,
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
        for chunk_id, content, metadata, score in rows:
            if score < min_score:
                continue
            # metadata arrives as a dict when psycopg2 uses jsonb; ensure it is one
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append(
                {"id": chunk_id, "content": content, "metadata": metadata, "score": score}
            )
        return results

    def search_fulltext(self, query_text: str, top_k: int) -> list[dict]:
        """Return chunks matching any word of ``query_text`` via full-text search.

        Uses ``websearch_to_tsquery('simple', ...)`` against the
        ``content_tsv`` generated column (``migrations/0002_add_fulltext_
        search.py``), ranked by ``ts_rank``. This is the keyword half of
        hybrid search; see :func:`query.hybrid.reciprocal_rank_fusion` for
        how it's combined with :meth:`search`'s vector results.

        ``query_text``'s words are joined with `` or `` before being
        passed to ``websearch_to_tsquery`` — confirmed empirically to be
        necessary, not optional: feeding a raw natural-language question
        straight in ANDs together every one of its words (a `` or ``-free
        input is plain-AND syntax, not OR), and the ``simple`` config has
        no stopword list (that's *why* it was chosen — see the migration's
        docstring — a Hungarian/English stopword list would only cover one
        language well). So a question like "Milyen technológiai stacket
        használ ...?" ANDed literally requires "milyen" and "használ" to
        also appear verbatim in the matching chunk — which they never will
        for an English-language chunk — and the search would silently
        return nothing for almost every real question. OR-joining instead
        matches chunks containing *any* of the question's words, ranked by
        how many/how prominently they matched — a reasonable keyword-recall
        signal to feed into RRF fusion, which is exactly what a keyword
        search component should contribute alongside vector search.

        Args:
            query_text: The raw question/query text.
            top_k: Maximum number of results.

        Returns:
            A list of chunk dicts ordered by descending ``ts_rank``, each
            containing ``id`` (see :meth:`search`'s docstring for why),
            ``content``, ``metadata``, and ``score``. This score is a
            ``ts_rank`` value, on a completely different scale than
            :meth:`search`'s cosine similarity — never compare the two
            directly, only their *ranks* (which is exactly what RRF does).
        """
        or_joined_query = " or ".join(query_text.split())
        sql = """
            SELECT
                id,
                content,
                metadata,
                ts_rank(content_tsv, websearch_to_tsquery('simple', %s)) AS score
            FROM document_chunks
            WHERE content_tsv @@ websearch_to_tsquery('simple', %s)
            ORDER BY score DESC
            LIMIT %s;
        """
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (or_joined_query, or_joined_query, top_k))
                rows = cur.fetchall()
        finally:
            conn.close()

        results = []
        for chunk_id, content, metadata, score in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append(
                {"id": chunk_id, "content": content, "metadata": metadata, "score": score}
            )
        return results

    def has_chunks_from_source(self, source_file: str) -> bool:
        """Return True if any stored chunk's metadata has this ``source_file``.

        Used by ``scripts/evaluate_retrieval.py`` to seed its known fixture
        documents idempotently — add them only if missing, never re-ingest
        (which would create duplicate rows) or delete anything first (which
        would make the script destructive against whatever database it's
        pointed at).

        Args:
            source_file: The basename to check for (e.g. ``"sample.pdf"``),
                matching :func:`ingestion.ingest.add_document`'s
                ``metadata.source_file``.

        Returns:
            True if at least one chunk with this ``source_file`` exists.
        """
        sql = "SELECT 1 FROM document_chunks WHERE metadata->>'source_file' = %s LIMIT 1;"
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (source_file,))
                return cur.fetchone() is not None
        finally:
            conn.close()

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
