"""Steps that produce candidate chunks: embedding the question and searching.

Key exports:
    EmbedQueryStep     -- Embeds the question (unless the caller supplied a vector).
    DenseSearchStep    -- The vector (cosine) search over the candidate pool.
    CslsReorderStep    -- Re-orders the pool by a hubness-corrected score.
    KeywordSearchStep  -- The full-text search over a pool as large as the vector one.
    YearDenseWideningStep, YearKeywordWideningStep
                       -- Add candidates from the question's years to the two pools.
"""

from drivers.embedding import EmbeddingDriver
from models import RetrievedChunk
from query.context import RetrievalContext, Slot
from query.step import AuxRecord, Continue, RetrievalStep, StepName, StepResult
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


class CslsReorderStep(RetrievalStep):
    """Re-sorts the vector pool by ``2 * similarity - hub score`` (CSLS).

    Some chunks sit in a generic region of the embedding space and score deceptively
    high against almost any question; subtracting the chunk's hub score demotes them.
    Only the *order* changes, never the set, and each chunk keeps its raw similarity
    (the fusion only uses rank positions). A chunk without a hub score keeps its raw
    score, so this is safe on a partly scored corpus. The sort is stable.
    """

    name = StepName.CSLS_REORDER
    requires = frozenset({Slot.DENSE_POOL})
    provides = frozenset({Slot.DENSE_POOL})

    def run(self, context: RetrievalContext) -> StepResult:
        def adjusted(chunk: RetrievedChunk) -> float:
            hub = chunk.metadata.hub_score
            return 2 * chunk.score - hub if hub is not None else chunk.score

        pool = sorted(context.dense_pool or (), key=adjusted, reverse=True)
        return Continue(context.with_slots(dense_pool=tuple(pool)))


class KeywordSearchStep(RetrievalStep):
    """Full-text search, fetching as many candidates as the vector pool holds.

    Args:
        store: The (possibly document-restricted) store to search.
    """

    name = StepName.KEYWORD_SEARCH
    requires = frozenset({Slot.DENSE_POOL})
    provides = frozenset({Slot.KEYWORD_POOL})

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def run(self, context: RetrievalContext) -> StepResult:
        limit = len(context.dense_pool or ())
        question = context.facts.question
        if context.metadata_filter:
            pool = self._store.search_fulltext(
                question,
                top_k=limit,
                metadata_filter=dict(context.metadata_filter),
            )
        else:
            pool = self._store.search_fulltext(question, top_k=limit)
        return Continue(context.with_slots(keyword_pool=tuple(pool)))


def _merge_unique(
    primary: tuple[RetrievedChunk, ...], extra: tuple[RetrievedChunk, ...]
) -> tuple[RetrievedChunk, ...]:
    """Append the ``extra`` chunks whose id is not already in ``primary``."""
    seen = {c.id for c in primary}
    return primary + tuple(c for c in extra if c.id not in seen)


class YearDenseWideningStep(RetrievalStep):
    """Adds a second, year-restricted vector pool when the question names years.

    Embeddings are weak at telling years apart, so the top-N most similar chunks can
    hold none from the year asked about. The year pool is *added* to the unrestricted
    one (a wrongly read year only adds candidates, never removes any), the union is
    sorted by raw similarity again (ties keep the original order), and the years go
    into the notes. Without years in the question nothing changes.

    Args:
        store: The (possibly document-restricted) store to search.
        pool_size: How many year candidates to fetch.
    """

    name = StepName.YEAR_DENSE_WIDENING
    requires = frozenset({Slot.QUERY_VECTOR, Slot.DENSE_POOL})
    provides = frozenset({Slot.DENSE_POOL})

    def __init__(self, store: VectorStore, pool_size: int) -> None:
        self._store = store
        self._pool_size = pool_size

    def run(self, context: RetrievalContext) -> StepResult:
        years = list(context.facts.years)
        if not years:
            return Continue(context)
        assert context.query_vector is not None  # guaranteed by `requires`
        vector = list(context.query_vector)
        if context.metadata_filter:
            found = self._store.search(
                vector,
                top_k=self._pool_size,
                min_score=0.0,
                years=years,
                metadata_filter=dict(context.metadata_filter),
            )
        else:
            found = self._store.search(
                vector, top_k=self._pool_size, min_score=0.0, years=years
            )
        merged = sorted(
            _merge_unique(context.dense_pool or (), tuple(found)),
            key=lambda c: c.score,
            reverse=True,
        )
        return Continue(
            context.with_slots(dense_pool=tuple(merged)),
            records=(AuxRecord("year_pool", tuple(found)),),
            notes={"years": years},
        )


class YearKeywordWideningStep(RetrievalStep):
    """Adds a year-restricted full-text pool when the question names years.

    The keyword side of :class:`YearDenseWideningStep`: the year candidates are
    appended after the unrestricted ones (no re-sorting; the fusion that follows only
    uses rank positions). The limit is the size of the (widened) vector pool.

    Args:
        store: The (possibly document-restricted) store to search.
    """

    name = StepName.YEAR_KEYWORD_WIDENING
    requires = frozenset({Slot.DENSE_POOL, Slot.KEYWORD_POOL})
    provides = frozenset({Slot.KEYWORD_POOL})

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def run(self, context: RetrievalContext) -> StepResult:
        years = list(context.facts.years)
        if not years:
            return Continue(context)
        limit = len(context.dense_pool or ())
        question = context.facts.question
        if context.metadata_filter:
            found = self._store.search_fulltext(
                question,
                top_k=limit,
                years=years,
                metadata_filter=dict(context.metadata_filter),
            )
        else:
            found = self._store.search_fulltext(question, top_k=limit, years=years)
        return Continue(
            context.with_slots(
                keyword_pool=_merge_unique(context.keyword_pool or (), tuple(found))
            ),
            records=(AuxRecord("year_pool", tuple(found)),),
        )
