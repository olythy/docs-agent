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
import re
from collections.abc import Callable
from contextlib import contextmanager
from typing import ClassVar, Self

from config import settings
from db import get_connection
from logger import get_logger
from models import Chunk, ChunkMetadata, RetrievedChunk

#: Technical acronyms and terms of 2-3 characters that should NOT be filtered out.
PRESERVED_SHORT_TERMS: frozenset[str] = frozenset(
    {
        "ai",
        "ui",
        "ux",
        "db",
        "ci",
        "cd",
        "go",
        "id",
        "ip",
        "os",
        "qa",
        "api",
        "rag",
        "sql",
        "llm",
    }
)

#: Combined bilingual (Hungarian + English) stop words for full-text query sanitization.
#: Prevents ranking chunks solely by occurrence of grammatical glue words.
BILINGUAL_STOPWORDS: frozenset[str] = frozenset(
    {
        # Hungarian stopwords / question words / particles
        "a",
        "az",
        "és",
        "vagy",
        "hogy",
        "van",
        "volt",
        "nem",
        "sem",
        "mint",
        "mert",
        "csak",
        "már",
        "még",
        "milyen",
        "mikor",
        "hol",
        "hova",
        "honnan",
        "ki",
        "kit",
        "kivel",
        "mi",
        "mit",
        "mivel",
        "miért",
        "hogyan",
        "melyik",
        "ez",
        "ezen",
        "azon",
        "itt",
        "ott",
        "egy",
        "egyik",
        "másik",
        "is",
        "se",
        "ne",
        "ha",
        "de",
        "te",
        "én",
        "ti",
        "ő",
        "ők",
        "ön",
        "önök",
        "lenne",
        "lesz",
        # English stopwords / pronouns / prepositions / auxiliaries
        "an",
        "the",
        "and",
        "or",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "from",
        "by",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "of",
        "off",
        "over",
        "under",
        "how",
        "what",
        "when",
        "where",
        "who",
        "which",
        "why",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "us",
        "our",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "i",
        "me",
        "my",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "could",
        "will",
        "would",
        "should",
        "not",
        "no",
        "nor",
        "so",
        "than",
        "too",
        "very",
        "just",
    }
)


def prepare_fulltext_query(query_text: str) -> tuple[str, list[str], list[str]]:
    """Clean and prepare a query for Postgres full-text search.

    Filters out grammatical stop words and short tokens (<= 2 chars) unless
    they are recognized technical acronyms (e.g. 'AI', 'UI'). Returns the
    OR-joined query along with lists of kept and dropped terms for telemetry.

    If all terms would be dropped (e.g. 'Who is it?'), falls back to all
    original terms to avoid returning an empty query.

    Args:
        query_text: The raw user query.

    Returns:
        A tuple of (or_joined_query, kept_terms, dropped_terms).
    """
    kept: list[str] = []
    dropped: list[str] = []

    for raw_word in query_text.split():
        clean = raw_word.strip(".,!?:;\"'()[]{}")
        if not clean:
            continue
        lower = clean.lower()

        if lower in PRESERVED_SHORT_TERMS:
            kept.append(clean)
        elif len(lower) <= 2 or lower in BILINGUAL_STOPWORDS:
            dropped.append(clean)
        else:
            kept.append(clean)

    if not kept:
        # Fallback if entire question was stop words
        fallback = [
            w.strip(".,!?:;\"'()[]{}")
            for w in query_text.split()
            if w.strip(".,!?:;\"'()[]{}")
        ]
        return " or ".join(fallback), fallback, []

    return " or ".join(kept), kept, dropped


#: SQL fragment: does any of the chunk's ``document_identifiers`` match the
#: regex parameter once separators are stripped? Both sides are reduced to
#: lower-case letters and digits (see :func:`_identifier_regex`), so
#: "P.20487.2020.221" and the stored "4.P.20.487/2020/221-ítélet" compare equal.
_NORMALIZED_IDENTIFIER_MATCH = (
    "EXISTS (SELECT 1 FROM jsonb_array_elements_text("
    "{metadata}->'document_identifiers') AS ident "
    "WHERE lower(regexp_replace(ident, '[^a-zA-Z0-9]', '', 'g')) ~ {regex})"
)
_MIN_NORMALIZED_IDENTIFIER_LENGTH = 6


def _identifier_regex(token: str) -> str | None:
    """Build a separator-insensitive regex for ``token``, or ``None`` if too weak.

    Strips everything but letters and digits and lower-cases, then requires a
    non-digit (or the string edge) after it, so ``P.20487.2020.22`` cannot
    match inside ``...2020.221``; and before it too, but only when the token
    itself *starts* with a digit (a letter-initial token like ``P...`` must
    still match inside the stored ``4p...``, where the digit ``4`` is the
    court-number prefix). Confirmed live: a user typing
    "P.20487.2020.221" never reached a document that writes the same case
    number "4.P.20.487/2020/221", because the literal ``ILIKE`` needs the
    exact punctuation.

    Returns:
        The regex, or ``None`` when the normalized token is too short or has
        no digit -- matching that loosely would flood the results.
    """
    normalized = re.sub(r"[^a-z0-9]", "", token.lower())
    if len(normalized) < _MIN_NORMALIZED_IDENTIFIER_LENGTH or not any(
        ch.isdigit() for ch in normalized
    ):
        return None
    lead = "(^|[^0-9])" if normalized[0].isdigit() else ""
    return f"{lead}{normalized}([^0-9]|$)"


def extract_identifier_tokens(query_text: str) -> list[str]:
    """Pull out code-like identifier tokens from a question (case numbers,
    invoice numbers, contract references, ...) that full-text search
    reliably loses track of.

    Confirmed empirically (see docs/decisions.md): ``prepare_fulltext_query()``
    OR-joins every kept word, and Postgres's ``ts_rank`` scores by
    frequency/coverage across the whole query -- a rare, exact identifier
    like a case number gets drowned out by common legal boilerplate words
    ("bíróság", "per tárgya") that also match, just far more often, across
    many unrelated documents. This function's job is only to *recognize*
    which tokens look like identifiers, document-type-agnostic (a court
    case number, an invoice number like "HU001", a contract reference —
    none of these should need their own hardcoded pattern); what the
    caller does with them (a direct, unranked text match --
    :meth:`VectorStore.search_by_identifier`) is what actually rescues them
    from the ranking problem.

    A token counts as identifier-like if it is not a plain word and not a
    short plain number:
        - contains a mix of letters and digits (e.g. "HU001"), or
        - contains a separator (``.``/``/``/``-``), even with only digits
          (e.g. "4.P.20.409/2023/4"), or
        - is purely digits but 5+ long (long enough to not just be a
          4-digit year).

    One explicit exclusion, confirmed live to matter (see docs/decisions.md):
    a bare number with a short Hungarian grammatical suffix attached via a
    hyphen (e.g. "2020-as" = "of 2020", "2023-ban" = "in 2023") matches the
    "digits + separator" rule above by accident, and being an extremely
    common way to mention a year in Hungarian, floods
    :meth:`VectorStore.search_by_identifier`'s result limit with irrelevant
    matches before the real, rare identifier tokens in the same question
    get a chance to appear. Excluded via a dedicated check rather than
    trying to special-case it in the main rule above, since it's a
    different kind of exception (a false positive to filter back out, not
    another way to recognize a true identifier).

    Args:
        query_text: The raw user question.

    Returns:
        The distinct identifier-like tokens found, in order of appearance.
    """
    tokens: list[str] = []
    seen: set[str] = set()

    for raw_word in query_text.split():
        token = raw_word.strip(".,!?:;\"'()[]{}")
        if not token:
            continue

        if _is_inflected_number(token):
            continue

        has_digit = any(c.isdigit() for c in token)
        has_letter = any(c.isalpha() for c in token)
        has_separator = any(c in "./-" for c in token)

        is_identifier = has_digit and (
            has_letter or has_separator or (token.isdigit() and len(token) >= 5)
        )

        if is_identifier and token not in seen:
            tokens.append(token)
            seen.add(token)

    return tokens


def _is_inflected_number(token: str) -> bool:
    """True for a bare number with a short Hungarian suffix, e.g. "2020-as".

    Digits, then a hyphen, then a short (1-3 letter) all-lowercase suffix
    and nothing else -- deliberately narrow, matching only this specific
    false-positive shape (see :func:`extract_identifier_tokens`'s
    docstring), not attempting general Hungarian morphology.
    """
    digits, sep, suffix = token.partition("-")
    return (
        bool(sep)
        and digits.isdigit()
        and 1 <= len(suffix) <= 3
        and suffix.isalpha()
        and suffix.islower()
    )


#: SQL fragment: the chunk's document_date (ISO ``YYYY-MM-DD``) is in one of
#: the given years (a ``text[]`` parameter).
_YEAR_CONDITION = "left(metadata->>'document_date', 4) = ANY(%s)"


def _to_pgvector_literal(embedding: list[float]) -> str:
    """Format a float vector as a pgvector literal, e.g. ``'[0.1,0.2,...]'``."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


class VectorStore:
    """Persistence layer for the document_chunks table.

    Can be initialized with an existing PostgreSQL connection to reuse across
    multiple operations (e.g. during hybrid retrieval or batched ingestion),
    or without one, in which case it opens and closes its own short-lived
    connection per call.

    Args:
        conn: Optional active psycopg connection. If provided, callers are
            responsible for closing it.
    """

    _validated_dimensions: ClassVar[set[int]] = set()

    @classmethod
    def clear_dimension_cache(cls) -> None:
        """Clear cached dimension validations. Useful in test fixtures."""
        cls._validated_dimensions.clear()

    def __init__(self, conn=None) -> None:
        self._conn = conn
        self._managed_conn = None
        self._conn_depth = 0

    def __enter__(self) -> Self:
        """Enter the connection context, opening a reusable connection if none exists."""
        if self._conn is None:
            self._managed_conn = get_connection()
            self._conn = self._managed_conn
        self._conn_depth += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the connection context, closing the connection when the outermost scope exits."""
        self._conn_depth -= 1
        if self._conn_depth <= 0:
            self._conn_depth = 0
            if self._managed_conn is not None:
                try:
                    self._managed_conn.close()
                finally:
                    self._conn = None
                    self._managed_conn = None

    @contextmanager
    def _connection(self):
        """Context manager yielding the active or a newly opened connection."""
        if self._conn is not None:
            yield self._conn
        else:
            conn = get_connection()
            try:
                yield conn
            finally:
                conn.close()

    def delete_chunks_by_hash(self, content_hash: str) -> int:
        """Delete all document_chunks rows matching the given SHA-256 content hash.

        All deletions in the storage engine are keyed by content_hash to ensure
        unambiguous, collision-free removal of document chunks.

        Args:
            content_hash: The 64-character hexadecimal SHA-256 digest.

        Returns:
            The number of rows deleted.
        """
        sql = "DELETE FROM document_chunks WHERE metadata->>'content_hash' = %s;"
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (content_hash,))
                deleted = cur.rowcount
            conn.commit()
            return deleted

    def delete_chunks_from_source(self, source: str) -> int:
        """Delete all document_chunks rows associated with the given source path.

        Finds the content_hash associated with ``source`` and delegates to
        :meth:`delete_chunks_by_hash`. Falls back to direct source-matching
        only if the stored chunks lack a content_hash (e.g. raw test fixtures).

        Args:
            source: Source path to delete (e.g. ``"finance/report.pdf"``).

        Returns:
            The number of rows deleted.
        """
        content_hash = self.get_hash_by_source(source)
        if content_hash:
            return self.delete_chunks_by_hash(content_hash)

        sql = """
            DELETE FROM document_chunks
            WHERE (metadata ? 'sources' AND metadata->'sources' ? %s)
               OR metadata->>'source_path' = %s
               OR metadata->>'source_file' = %s;
        """
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (source, source, source))
                deleted = cur.rowcount
            conn.commit()
            return deleted

    def has_content_hash(self, content_hash: str) -> bool:
        """Check whether chunks with the given content hash are already stored.

        Args:
            content_hash: The 64-character hexadecimal SHA-256 digest.

        Returns:
            True if at least one chunk exists with this content_hash.
        """
        if not content_hash:
            return False
        sql = "SELECT 1 FROM document_chunks WHERE metadata->>'content_hash' = %s LIMIT 1;"
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (content_hash,))
            return cur.fetchone() is not None

    def get_hash_by_source(self, source_path: str) -> str | None:
        """Find the active content_hash associated with a given source path.

        Checks both the ``sources`` array and ``source_path``/``source_file`` keys.

        Args:
            source_path: The file path to look up.

        Returns:
            The content_hash string if found, otherwise None.
        """
        sql = """
            SELECT metadata->>'content_hash'
            FROM document_chunks
            WHERE (metadata ? 'sources' AND metadata->'sources' ? %s)
               OR metadata->>'source_path' = %s
               OR metadata->>'source_file' = %s
            LIMIT 1;
        """
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (source_path, source_path, source_path))
            row = cur.fetchone()
            return row[0] if row else None

    def add_source_alias(self, content_hash: str, new_source_path: str) -> int:
        """Add a new source path alias to all chunks sharing an identical content hash.

        Allows multiple file paths or copies to reference the same embedded chunks
        without generating duplicate chunk rows or requiring redundant embedding.

        Args:
            content_hash: The SHA-256 hash identifying the chunks.
            new_source_path: The additional file path referencing this content.

        Returns:
            The number of chunk rows updated.
        """
        sql = """
            UPDATE document_chunks
            SET metadata = jsonb_set(
                metadata,
                '{sources}',
                CASE
                    WHEN metadata ? 'sources' AND metadata->'sources' ? %s THEN metadata->'sources'
                    WHEN metadata ? 'sources' THEN metadata->'sources' || to_jsonb(%s::text)
                    ELSE jsonb_build_array(%s::text)
                END
            )
            WHERE metadata->>'content_hash' = %s;
        """
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (new_source_path, new_source_path, new_source_path, content_hash),
                )
                updated = cur.rowcount
            conn.commit()
            return updated

    def save(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Insert chunk rows into document_chunks.

        Args:
            chunks: One :class:`models.Chunk` per row to insert.
            embeddings: Parallel list of float vectors, one per chunk.

        Returns:
            The number of rows inserted.
        """
        insert_sql = """
            INSERT INTO document_chunks (content, metadata, embedding)
            VALUES (%s, %s, %s);
        """
        with self._connection() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings, strict=True):
                    cur.execute(
                        insert_sql,
                        (
                            chunk.content,
                            json.dumps(chunk.metadata.to_dict()),
                            _to_pgvector_literal(embedding),
                        ),
                    )
            conn.commit()
            return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        min_score: float,
        metadata_filter: dict | None = None,
        years: list[int] | None = None,
    ) -> list[RetrievedChunk]:
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
            metadata_filter: Optional dict of key-value pairs that chunk metadata
                must contain (uses Postgres JSONB containment ``@>``).
            years: Optional years; only chunks whose ``document_date`` falls
                in one of them are considered (documents without a date are
                excluded from this call -- callers that want them back run
                an unfiltered search too, see ``query.retrieval``).

        Returns:
            A list of :class:`models.RetrievedChunk` ordered by descending
            similarity. Each carries ``id`` (the row's primary key — lets
            callers like :func:`query.hybrid.reciprocal_rank_fusion`
            identify the *same* chunk across a separate keyword-search
            result set, since two different rows could coincidentally
            share identical text).
        """
        vector_literal = _to_pgvector_literal(query_embedding)
        conditions: list[str] = []
        params: list = [vector_literal]
        if metadata_filter:
            conditions.append("metadata @> %s::jsonb")
            params.append(json.dumps(metadata_filter))
        if years:
            conditions.append(_YEAR_CONDITION)
            params.append([str(y) for y in years])
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([vector_literal, top_k])

        sql = f"""
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        with self._connection() as conn, conn.cursor() as cur:
            if years:
                # pgvector's HNSW index applies a WHERE clause *after* it has
                # found its ef_search nearest neighbours, so a selective filter
                # (a year with ~90 of ~2,200 documents) can leave few or no
                # rows. Iterative scan keeps scanning until LIMIT rows pass the
                # filter. LOCAL: only this transaction, not the whole session.
                cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results = []
        for chunk_id, content, metadata, score in rows:
            if score < min_score:
                continue
            # metadata arrives as a dict when psycopg2 uses jsonb; ensure it is one
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append(
                RetrievedChunk(
                    id=chunk_id,
                    content=content,
                    metadata=ChunkMetadata.from_dict(metadata),
                    score=score,
                )
            )
        if years:
            # relaxed_order can return rows slightly out of distance order.
            results.sort(key=lambda c: c.score, reverse=True)
        return results

    def compute_hub_scores(
        self,
        on_progress: Callable[[int, int], None] | None = None,
        commit_every: int = 200,
    ) -> int:
        """Compute and store each chunk's "hubness" (genericness) score.

        A post-ingest batch pass, not run during chunking (which has no
        visibility into the rest of the corpus). For every chunk, queries
        its ``settings.HUB_SCORE_NEIGHBOR_SAMPLE_SIZE`` nearest neighbors in
        the *whole corpus* by cosine similarity (reusing the same HNSW
        index :meth:`search` does) and stores the average similarity to
        them as ``metadata.hub_score`` -- a continuous measure of how
        "generic"/central this chunk's embedding is, independent of any
        specific query.

        Supersedes an earlier, rejected approach (see docs/decisions.md):
        a binary "boilerplate" flag that *excluded* chunks above a fixed
        cross-document similarity threshold. That flagged 40% of a real
        corpus, including most of a known-correct answer's own chunks,
        because genuinely distinct (but formulaically phrased) legal
        reasoning scored just as "similar to many other chunks" as actual
        copy-pasted boilerplate -- a hard threshold can't tell those
        apart. A continuous penalty applied at *query* time (CSLS-style,
        see :func:`query.retrieval._csls_rerank`) never excludes anything
        outright, so it can't repeat that failure mode; confirmed live on
        the same real test case that CSLS re-ranking alone (no exclusion)
        moved a known-correct document from rank 16 to rank 6 of 19 real
        near-duplicate competitors.

        Idempotent -- re-running it recomputes every chunk's score fresh,
        reflecting whatever's in the corpus at the time.

        Issues one nearest-neighbor query per chunk, so this is O(n) round
        trips, not O(n²) -- confirmed live to still take a while on a real
        corpus (tens of minutes for ~11,000 chunks), so progress is
        committed every ``commit_every`` chunks rather than in one
        transaction at the end.

        Args:
            on_progress: Optional callback invoked after each committed
                batch with ``(processed_count, total_count)`` -- the CLI
                wrapper uses this to print progress; tests can pass
                ``None`` (the default) and ignore it.
            commit_every: How many chunks to process between commits
                (default: 200).

        Returns:
            The number of chunks whose hub_score was updated.
        """
        sample_size = settings.HUB_SCORE_NEIGHBOR_SAMPLE_SIZE

        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, embedding::text FROM document_chunks;")
                chunks = cur.fetchall()

            total_updated = 0
            with conn.cursor() as cur:
                batch: list[tuple[int, float]] = []
                for processed, (chunk_id, embedding_text) in enumerate(chunks, start=1):
                    cur.execute(
                        """
                        SELECT 1 - (embedding <=> %s::vector) AS score
                        FROM document_chunks
                        WHERE id != %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (embedding_text, chunk_id, embedding_text, sample_size),
                    )
                    neighbor_scores = [r[0] for r in cur.fetchall()]
                    hub_score = (
                        sum(neighbor_scores) / len(neighbor_scores)
                        if neighbor_scores
                        else 0.0
                    )
                    batch.append((chunk_id, hub_score))

                    if processed % commit_every == 0 or processed == len(chunks):
                        for cid, score in batch:
                            cur.execute(
                                """
                                UPDATE document_chunks
                                SET metadata = jsonb_set(
                                    metadata, '{hub_score}', %s::jsonb
                                )
                                WHERE id = %s;
                                """,
                                (json.dumps(score), cid),
                            )
                        total_updated += len(batch)
                        conn.commit()
                        if on_progress:
                            on_progress(processed, len(chunks))
                        batch = []
        return total_updated

    def search_fulltext(
        self,
        query_text: str,
        top_k: int,
        metadata_filter: dict | None = None,
        years: list[int] | None = None,
    ) -> list[RetrievedChunk]:
        """Return chunks matching any word of ``query_text`` via full-text search.

        Uses ``websearch_to_tsquery('hungarian', ...)`` against the
        ``content_tsv`` generated column (originally ``simple`` in
        ``migrations/0002_add_fulltext_search.py``, switched to Postgres's
        built-in ``hungarian`` snowball-stemmer config by
        ``migrations/0003_hungarian_fulltext_search_config.py`` — confirmed
        live to measurably improve keyword-search rank on this
        all-Hungarian legal corpus, see docs/decisions.md's 2026-10-03
        entry), ranked by ``ts_rank``. This is the keyword half of
        hybrid search; see :func:`query.hybrid.reciprocal_rank_fusion` for
        how it's combined with :meth:`search`'s vector results. The
        tsquery-side config must always match ``content_tsv``'s own
        config — a query built with a different config would tokenize
        into different (e.g. unstemmed) lexemes than what's actually
        stored, breaking matches silently rather than erroring.

        ``query_text``'s words are joined with `` or `` before being
        passed to ``websearch_to_tsquery`` — confirmed empirically to be
        necessary, not optional: feeding a raw natural-language question
        straight in ANDs together every one of its words (a `` or ``-free
        input is plain-AND syntax, not OR). ``prepare_fulltext_query()``'s
        own bilingual stopword list still does useful work even with the
        ``hungarian`` config's built-in Hungarian stopword list, since it
        also drops common *English* stopwords that config has no notion
        of. So a question like "Milyen technológiai stacket használ
        ...?" ANDed literally requires "milyen" and "használ" to
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
            metadata_filter: Optional dict of key-value pairs that chunk metadata
                must contain (uses Postgres JSONB containment ``@>``).
            years: Optional years; same meaning as in :meth:`search`.

        Returns:
            A list of :class:`models.RetrievedChunk` ordered by descending
            ``ts_rank`` (see :meth:`search`'s docstring for why ``id``
            matters). This score is a ``ts_rank`` value, on a completely
            different scale than :meth:`search`'s cosine similarity — never
            compare the two directly, only their *ranks* (which is exactly
            what RRF does).
        """
        or_joined_query, kept_terms, dropped_terms = prepare_fulltext_query(query_text)
        get_logger().log_fts_query_filtered(
            original_query=query_text,
            kept_terms=kept_terms,
            dropped_terms=dropped_terms,
        )
        where_filter = ""
        params: list = [or_joined_query, or_joined_query]
        if metadata_filter:
            where_filter += " AND metadata @> %s::jsonb"
            params.append(json.dumps(metadata_filter))
        if years:
            where_filter += f" AND {_YEAR_CONDITION}"
            params.append([str(y) for y in years])
        params.append(top_k)

        sql = f"""
            SELECT
                id,
                content,
                metadata,
                ts_rank(content_tsv, websearch_to_tsquery('hungarian', %s)) AS score
            FROM document_chunks
            WHERE content_tsv @@ websearch_to_tsquery('hungarian', %s)
            {where_filter}
            ORDER BY score DESC
            LIMIT %s;
        """
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        results = []
        for chunk_id, content, metadata, score in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append(
                RetrievedChunk(
                    id=chunk_id,
                    content=content,
                    metadata=ChunkMetadata.from_dict(metadata),
                    score=score,
                )
            )
        return results

    def search_by_identifier(
        self, tokens: list[str], top_k: int, per_token: bool = False
    ) -> list[RetrievedChunk]:
        """Return chunks whose content, or stored case-number, matches any of ``tokens``.

        A direct ``ILIKE`` substring match on the content, **or** a
        separator-insensitive match against the chunk's ``document_identifiers``
        (see :func:`_identifier_regex`) so a case number typed with different
        punctuation than the document uses still finds it. Deliberately *not* ranked by
        ``ts_rank`` like :meth:`search_fulltext` — see
        :func:`extract_identifier_tokens`'s docstring for why an exact
        identifier match (a case number, invoice number, ...) needs to
        bypass frequency-based ranking entirely rather than compete with
        common words for score.

        Args:
            tokens: Identifier-like tokens from :func:`extract_identifier_tokens`.
            top_k: Maximum number of results.
            per_token: If ``True``, give every token its own bounded share
                (``ceil(top_k / len(tokens))`` rows, in ``id`` order)
                instead of one shared ``LIMIT``. Without it, a token whose
                document has many chunks (the identifier is embedded in
                every chunk) can take all ``top_k`` rows and starve the
                other tokens' documents -- and since there is no
                ``ORDER BY``, which rows win is arbitrary. A chunk that
                matches several tokens is returned once.

        Returns:
            Matching chunks, each with a placeholder ``score`` (1.0) — the
            caller (:class:`query.retrieval.HybridRetrievalStrategy`)
            doesn't rank these against the vector/full-text results, it
            merges them in directly, and the reranker re-scores everything
            downstream anyway.
        """
        if not tokens:
            return []

        if per_token:
            patterns = [f"%{token}%" for token in tokens]
            regexes = [_identifier_regex(token) for token in tokens]
            per_token_limit = max(1, -(-top_k // len(tokens)))
            normalized_match = _NORMALIZED_IDENTIFIER_MATCH.format(
                metadata="c.metadata", regex="t.r"
            )
            sql = f"""
                SELECT id, content, metadata FROM (
                    SELECT c.id, c.content, c.metadata,
                           ROW_NUMBER() OVER (PARTITION BY t.p ORDER BY c.id) AS rn
                    FROM document_chunks c,
                         unnest(%s::text[], %s::text[]) AS t(p, r)
                    WHERE c.content ILIKE t.p
                       OR (t.r IS NOT NULL AND {normalized_match})
                ) matched
                WHERE rn <= %s
                ORDER BY id;
            """
            query_params: tuple = (patterns, regexes, per_token_limit)
        else:
            params: list = []
            condition_parts = []
            for token in tokens:
                regex = _identifier_regex(token)
                part = "content ILIKE %s"
                params.append(f"%{token}%")
                if regex is not None:
                    part += " OR " + _NORMALIZED_IDENTIFIER_MATCH.format(
                        metadata="metadata", regex="%s"
                    )
                    params.append(regex)
                condition_parts.append(f"({part})")
            conditions = " OR ".join(condition_parts)
            params.append(top_k)
            sql = f"""
                SELECT id, content, metadata
                FROM document_chunks
                WHERE {conditions}
                LIMIT %s;
            """
            query_params = tuple(params)
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, query_params)
            rows = cur.fetchall()
        if per_token:
            rows = list({row[0]: row for row in rows}.values())

        results = []
        for chunk_id, content, metadata in rows:
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            results.append(
                RetrievedChunk(
                    id=chunk_id,
                    content=content,
                    metadata=ChunkMetadata.from_dict(metadata),
                    score=1.0,
                )
            )
        return results

    def has_chunks_from_source(self, source: str) -> bool:
        """Return True if any stored chunk's metadata matches this source identifier.

        Matches against either ``source_path`` (logical path) or ``source_file``
        (basename) for compatibility.

        Used by ``scripts/evaluate_retrieval.py`` to seed its known fixture
        documents idempotently — add them only if missing, never re-ingest
        (which would create duplicate rows) or delete anything first (which
        would make the script destructive against whatever database it's
        pointed at).

        Args:
            source: The source path or basename to check for (e.g. ``"sample.pdf"``).

        Returns:
            True if at least one chunk with this source identifier exists.
        """
        sql = """
            SELECT 1 FROM document_chunks
            WHERE metadata->>'source_path' = %s
               OR metadata->>'source_file' = %s
            LIMIT 1;
        """
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(sql, (source, source))
            return cur.fetchone() is not None

    def get_all_source_files(self) -> set[str]:
        """Return every distinct ``source_file`` currently in document_chunks.

        Used by ``corpus.commands.coverage`` to check, during a growing or
        partial ingest, which golden questions' cited documents are
        already ingested -- cheap enough to run before a full `eval` pass
        on a corpus that isn't fully loaded yet.

        Returns:
            The set of distinct source_file values across all chunks.
        """
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT metadata->>'source_file' FROM document_chunks;"
            )
            return {row[0] for row in cur.fetchall()}

    def count_hub_scored_chunks(self) -> tuple[int, int]:
        """Count how many chunks already carry a ``metadata.hub_score``.

        Used by ``corpus.commands.coverage`` to flag a stale or missing
        :meth:`compute_hub_scores` pass -- chunks ingested after the last
        run (or before the first) have no score, so CSLS re-ranking
        silently has nothing to work with for them.

        Returns:
            ``(scored, total)`` chunk counts.
        """
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FILTER (WHERE metadata ? 'hub_score'), count(*) "
                "FROM document_chunks;"
            )
            row = cur.fetchone()
            assert row is not None  # an aggregate query always returns one row
            return row[0], row[1]

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
        with self._connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                    SELECT atttypmod FROM pg_attribute
                    WHERE attrelid = to_regclass('document_chunks')
                      AND attname = 'embedding'
                      AND NOT attisdropped;
                    """
            )
            row = cur.fetchone()
        return row[0] if row else None

    def assert_dimension_matches(self, expected_dimension: int) -> None:
        """Raise a clear error if ``expected_dimension`` disagrees with the DB column.

        Caches successfully verified dimensions in-memory to prevent redundant
        catalog queries against pg_attribute on every subsequent query.

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
        if expected_dimension in self._validated_dimensions:
            return

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

        self._validated_dimensions.add(expected_dimension)
