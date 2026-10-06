"""Steps that produce candidate chunks: embedding the question and searching.

Key exports:
    EmbedQueryStep  -- Embeds the question (unless the caller supplied a vector).
    DenseSearchStep -- The vector (cosine) search over the candidate pool.
"""

from drivers.embedding import EmbeddingDriver
from query.context import RetrievalContext, Slot
from query.step import Continue, RetrievalStep, StepName, StepResult
from store import VectorStore


class EmbedQueryStep(RetrievalStep):
    """Embeds the question, unless the context already carries a query vector.

    Args:
        embedding: The driver used to embed (the one used at ingestion).
    """

    name = StepName.EMBED_QUERY
    provides = frozenset({Slot.QUERY_VECTOR})

    def __init__(self, embedding: EmbeddingDriver) -> None:
        self._embedding = embedding

    def run(self, context: RetrievalContext) -> StepResult:
        if context.query_vector is not None:
            return Continue(context)
        vector = self._embedding.embed_query(context.facts.question)
        return Continue(context.with_slots(query_vector=tuple(vector)))


class DenseSearchStep(RetrievalStep):
    """Cosine-similarity search for a pool of candidates, with no score threshold.

    Nothing is filtered by score here: the relevance gate and the later steps decide.

    Args:
        store: The (possibly document-restricted) store to search.
        pool_size: How many candidates to fetch.
    """

    name = StepName.DENSE_SEARCH
    requires = frozenset({Slot.QUERY_VECTOR})
    provides = frozenset({Slot.DENSE_POOL})

    def __init__(self, store: VectorStore, pool_size: int) -> None:
        self._store = store
        self._pool_size = pool_size

    def run(self, context: RetrievalContext) -> StepResult:
        assert context.query_vector is not None  # guaranteed by `requires`
        vector = list(context.query_vector)
        if context.metadata_filter:
            pool = self._store.search(
                vector,
                top_k=self._pool_size,
                min_score=0.0,
                metadata_filter=dict(context.metadata_filter),
            )
        else:
            pool = self._store.search(vector, top_k=self._pool_size, min_score=0.0)
        return Continue(context.with_slots(dense_pool=tuple(pool)))
