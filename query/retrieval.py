"""Knowledge base retrieval and answer generation.

This module implements the ``query_knowledge_base`` tool — the second of the two
tools the agent can call (the first being ``add_document`` in ``ingestion/ingest.py``).

Flow:
    1. Embed the user's question with the same driver used during ingestion.
    2. Run a cosine-similarity vector search against ``document_chunks`` in Postgres.
    3. Filter results below the minimum relevance threshold.
    4. If no relevant chunks remain, return a polite "I don't know" message.
    5. Otherwise, pass the chunks to the LLM driver and return its grounded answer.

Usage::

    from query.retrieval import query_knowledge_base
    answer = query_knowledge_base("Mennyi az SZJA tartozásom?")
    print(answer)
"""

import json

import psycopg2

from config import settings
from drivers.embedding import get_embedding_driver
from drivers.llm import get_answer_driver

# Returned when no chunk clears the relevance threshold.
# Using a constant avoids scatter: every caller sees the same wording,
# and the agent layer (step 6) can test for this exact string if needed.
NO_RESULTS_MESSAGE = (
    "I could not find relevant information about this in the provided documents."
)


def _get_connection():
    """Open and return a new Postgres connection using settings.DATABASE_URL.

    Returns:
        A ``psycopg2`` connection object.

    Raises:
        RuntimeError: If ``DATABASE_URL`` is not configured.
    """
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not configured. Set DATABASE_URL in your .env file."
        )
    return psycopg2.connect(settings.DATABASE_URL)


def _retrieve_chunks(
    query_embedding: list[float],
    top_k: int,
    min_score: float,
) -> list[dict]:
    """Query the vector store for the most similar chunks.

    Uses pgvector's ``<=>`` operator, which computes cosine *distance*
    (0 = identical, 2 = opposite). We convert to similarity with ``1 - distance``
    so that higher is better, matching the ``min_score`` threshold convention.

    Args:
        query_embedding: The embedded question vector (length = EMBEDDING_DIMENSION).
        top_k: Maximum number of results to consider before filtering.
        min_score: Minimum cosine similarity (0–1). Chunks below this are dropped.

    Returns:
        A list of chunk dicts ordered by descending similarity, each containing:
            - ``content`` (str): The chunk text.
            - ``metadata`` (dict): ``source_file``, ``page_number``, ``chunk_index``.
            - ``score`` (float): Cosine similarity score (0–1).
    """
    # Format vector as pgvector literal: '[0.1, 0.2, ...]'
    vector_literal = "[" + ",".join(str(v) for v in query_embedding) + "]"

    sql = """
        SELECT
            content,
            metadata,
            1 - (embedding <=> %s::vector) AS score
        FROM document_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (vector_literal, vector_literal, top_k))
            rows = cur.fetchall()
    finally:
        conn.close()

    chunks = []
    for content, metadata, score in rows:
        # Skip chunks that are too distant to be relevant
        if score < min_score:
            continue
        # metadata arrives as a dict when psycopg2 uses jsonb; ensure it is one
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        chunks.append({"content": content, "metadata": metadata, "score": score})

    return chunks


def query_knowledge_base(
    question: str,
    top_k: int | None = None,
    min_score: float | None = None,
) -> str:
    """Answer a question using the RAG knowledge base.

    This is the main tool exposed to the agent. It embeds the question, retrieves
    the most relevant document chunks, and generates a grounded answer. If no
    chunk clears the relevance threshold the agent receives an honest "I don't
    know" response instead of a hallucinated answer.

    Args:
        question: The user's natural-language question.
        top_k: Override for ``settings.RETRIEVAL_TOP_K``. Maximum chunks to fetch.
        min_score: Override for ``settings.RETRIEVAL_MIN_SCORE``. Similarity
            threshold (0–1) below which chunks are discarded.

    Returns:
        A string answer grounded in the retrieved chunks, or
        :data:`NO_RESULTS_MESSAGE` if no relevant chunks were found.
    """
    k = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
    threshold = min_score if min_score is not None else settings.RETRIEVAL_MIN_SCORE

    print("[query] Embedding question ...")
    embedding_driver = get_embedding_driver()
    query_vector = embedding_driver.embed_text(question)

    print(f"[query] Searching top-{k} chunks (min_score={threshold}) ...")
    chunks = _retrieve_chunks(query_vector, top_k=k, min_score=threshold)

    if not chunks:
        print("[query] No relevant chunks found — returning fallback message.")
        return NO_RESULTS_MESSAGE

    scores_str = ", ".join(f"{c['score']:.3f}" for c in chunks)
    print(f"[query] Found {len(chunks)} relevant chunk(s). Scores: {scores_str}")

    print(f"[query] Generating answer with LLM driver='{settings.LLM_DRIVER}' ...")
    answer_driver = get_answer_driver()
    answer = answer_driver.answer(question=question, context_chunks=chunks)

    print("[query] Done.")
    return answer
