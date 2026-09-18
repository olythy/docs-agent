"""Hybrid search: fuses vector and keyword search results by rank.

Key exports:
    reciprocal_rank_fusion  -- Merge two ranked chunk lists into one.
"""


def reciprocal_rank_fusion(
    vector_results: list[dict],
    fulltext_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """Fuse two ranked chunk lists into one, via Reciprocal Rank Fusion (RRF).

    Combines :meth:`store.VectorStore.search`'s (cosine similarity) and
    :meth:`store.VectorStore.search_fulltext`'s (``ts_rank``) result lists.
    RRF is deliberately *rank*-based, not score-based: cosine similarity
    and ``ts_rank`` live on completely different, incomparable numeric
    scales (0-1 vs. an unbounded text-relevance score), so averaging or
    weighting the raw numbers would be comparing apples to oranges. A
    chunk's *position* (1st, 2nd, ...) means the same thing in either
    list, so RRF sums ``1 / (rank + k)`` for each list a chunk appears in
    — this is the same technique Elasticsearch/OpenSearch's built-in
    hybrid search uses, and needs no extra model or training.

    Args:
        vector_results: :meth:`store.VectorStore.search`'s output, already
            sorted by descending similarity.
        fulltext_results: :meth:`store.VectorStore.search_fulltext`'s
            output, already sorted by descending ``ts_rank``.
        k: RRF's smoothing constant. The standard default (60, from the
            original RRF paper and widely reused since) keeps a single
            very-high individual rank from completely dominating the
            fused score.

    Returns:
        The union of both input lists, deduplicated by ``id`` (two
        different rows could coincidentally have identical text, so
        matching on content would be wrong), sorted by descending fused
        score. Each dict keeps its ``id``/``content``/``metadata``, with
        ``score`` replaced by the fused RRF value — the original cosine
        similarity / ``ts_rank`` numbers aren't meaningful anymore once
        merged, so keeping the same key name (rather than adding a new
        ``rrf_score`` key) avoids implying two different "real" scores
        exist for the same chunk.
    """
    fused: dict[int, dict] = {}
    for results in (vector_results, fulltext_results):
        for rank, chunk in enumerate(results, start=1):
            entry = fused.setdefault(
                chunk["id"],
                {"id": chunk["id"], "content": chunk["content"], "metadata": chunk["metadata"], "score": 0.0},
            )
            entry["score"] += 1 / (rank + k)

    return sorted(fused.values(), key=lambda c: c["score"], reverse=True)
