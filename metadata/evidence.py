"""Picks which parts of a long document an extractor should read.

A court decision runs to ~20 chunks, and the facts are scattered (the court at
the top, the date in the signature line at the end). Sending everything is
costly and dilutes the extractor, so for every catalog key the document's chunks
are reranked using the key's *description* as the query, and the best few are
kept. This is retrieval inside one document, with no corpus-specific rule: the
description alone tells the reranker where to look.

Key exports:
    EvidenceSelector -- Selects the chunks to show the extractor.
"""

from drivers.reranker import RerankerDriver
from models import MetaKey, RetrievedChunk


class EvidenceSelector:
    """Selects the chunks of one document that best match a set of catalog keys.

    Args:
        reranker: Scores chunks against a query.
        per_key: How many chunks to keep for each key.
        max_chunks: Upper bound on the chunks returned overall (the prompt budget).
    """

    def __init__(
        self, reranker: RerankerDriver, per_key: int = 2, max_chunks: int = 6
    ) -> None:
        self._reranker = reranker
        self._per_key = per_key
        self._max_chunks = max_chunks

    def select(
        self, chunks: list[RetrievedChunk], keys: list[MetaKey]
    ) -> list[RetrievedChunk]:
        """Return the chunks to show the extractor, in document order.

        Args:
            chunks: One document's chunks (their ``content`` should already be
                the body text, without embedded prefixes).
            keys: The keys about to be extracted.

        Returns:
            At most ``max_chunks`` chunks: the union of each key's best
            ``per_key`` chunks, ordered by ``chunk_index``. A document that is
            short enough is returned whole.
        """
        if len(chunks) <= self._max_chunks:
            return list(chunks)
        chosen: dict[int, RetrievedChunk] = {}
        for key in keys:
            ranked = self._reranker.rerank(f"{key.key}: {key.description}", chunks)
            for chunk in ranked[: self._per_key]:
                chosen[chunk.id] = chunk
        ordered = sorted(chosen.values(), key=lambda c: c.metadata.chunk_index)
        return ordered[: self._max_chunks]
